from typing import Callable, List, Dict, Iterable

import torch

from .GenericPSO import clone_param_groups, _initialize_param_groups, GenericParticle, GenericPSO


class SCAParticle(GenericParticle):
    def __init__(self,
                 param_groups,
                 max_param_value: float,
                 min_param_value: float):

        self.param_groups = param_groups
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value

        self.position = _initialize_param_groups(param_groups, max_param_value, min_param_value)

        self.best_known_position = clone_param_groups(self.position)
        self.best_known_loss_value = torch.inf

    def step(self, closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict],
             r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor, use_sine: bool) -> torch.Tensor:
        """
        Particle will take one step.
        :param closure: A callable that reevaluates the model and returns the loss.
        :param global_best_param_groups: List of param_groups that yield the best found loss globally
        :return:
        """
        # Because our parameters are not a single tensor, we have to iterate over each group, and then each param in
        # each group.
        for position_group, global_best, master in zip(self.position,
                                                       global_best_param_groups,
                                                       self.param_groups):
            position_group_params = position_group['params']
            global_best_params = global_best['params']
            master_params = master['params']

            new_position_params = []
            for p, gb, m in zip(position_group_params,
                                global_best_params, master_params):
                func = torch.sin if use_sine else torch.cos
                new_position = p + r1 * func(r2) * abs(r3 * gb - p)
                new_position_params.append(new_position)
                m.data = new_position.data  # Update the model, so we can use it for calculating loss
            position_group['params'] = new_position_params

        # Really crummy way to update the parameter weights in the original model.
        # Simply changing self.param_groups doesn't update the model.
        # Nor does changing its elements or the raw values of 'param' of the elements.
        # We have to change the underlying tensor data to point to the new positions
        for i in range(len(self.position)):
            for j in range(len(self.param_groups[i]['params'])):
                self.param_groups[i]['params'][j].data = self.param_groups[i]['params'][j].data

        # Calculate new loss after moving and update the best known position if we're in a better spot
        new_loss = closure()
        if new_loss < self.best_known_loss_value:
            self.best_known_position = clone_param_groups(self.position)
            self.best_known_loss_value = new_loss
        return new_loss


class SineCosineAlgorithm(GenericPSO):
    """
    The Sine Cosine Algorithm is an algorithm, conceived by Seyedali Mirjalili, similar to PSO, although the author
    does not use particle terminology in the paper. Each particle moves towards a destination (the best candidate
    solution) in a step size determined by several random parameters. The "sine" and "cosine" in the name come from the
    use of sine and cosine functions to determine whether the particle's step will explore or exploit the currently
    best solution.

    Original Paper:
    Seyedali Mirjalili , SCA: A Sine Cosine Algorithm for Solving Optimization Problems, Knowledge-Based Systems (2016),
    doi: 10.1016/j.knosys.2015.12.022

    https://dl.programstore.ir/files/Uploades/Lib/PDF/SCA.pdf
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 num_particles: int = 100,
                 max_movement_radius: float = 2,
                 max_param_value: float = -10,
                 min_param_value: float = 10):
        particle_kwargs = {
            'max_param_value': max_param_value,
            'min_param_value': min_param_value,
        }
        super().__init__(params, num_particles, particle_class=SCAParticle, particle_kwargs=particle_kwargs)
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value
        self.max_movement_radius = max_movement_radius
        self.magnitude = max_param_value-min_param_value
        self.initial_movement_radius = max_movement_radius


    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single optimization step.

        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        r3 = 2 * torch.rand((1,))
        r2 = 2*torch.pi*torch.rand((1,))
        use_sine = torch.rand((1,)).item() < .5
        r1 = self.max_movement_radius*torch.rand((1,))
        # self.max_movement_radius *= .99
        return super().step(closure, particle_step_kwargs={'r1': r1, 'r2': r2, 'r3': r3, 'use_sine': use_sine})
