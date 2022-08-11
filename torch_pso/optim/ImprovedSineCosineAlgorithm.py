import math
from typing import Callable, List, Dict, Iterable

import torch

from .GenericPSO import clone_param_groups, GenericPSO
from .SineCosineAlgorithm import SCAParticle


class ImprovedSCAParticle(SCAParticle):
    def __init__(self,
                 param_groups,
                 max_param_value: float,
                 min_param_value: float):

        super().__init__(param_groups, max_param_value, min_param_value)

    def step(self, closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict],
             r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor, use_sine: bool, w: float) -> torch.Tensor:
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
                new_position = w * p + r1 * func(r2) * abs(r3 * gb - p)
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


class ImprovedSineCosineAlgorithm(GenericPSO):
    """
    The Improved Sine Cosine Algorithm is a variation on the original Sine Cosine Algorithm, specifically attempting to
    improve its resilience against local optima and unbalanced exploitation. It also demonstrates better behavior in
    high-dimensional optimization problems.

    Original Paper:
    Long, W., Wu, T., Liang, X., & Xu, S. (2019). Solving high-dimensional global optimization problems using an
    improved sine cosine algorithm. Expert systems with applications, 123, 108-126.

    https://e-tarjome.com/storage/panel/fileuploads/2019-08-22/1566462251_E11587-e-tarjome.pdf
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 num_particles: int = 100,
                 a_start: float = 2,
                 a_end: float = 0,
                 max_time_steps: int = 1000,
                 w_end: float = 0,
                 w_start: float = .1,
                 k: float = 15.,
                 max_param_value: float = -10,
                 min_param_value: float = 10):
        particle_kwargs = {
            'max_param_value': max_param_value,
            'min_param_value': min_param_value,
        }
        super().__init__(params, num_particles, particle_class=ImprovedSCAParticle, particle_kwargs=particle_kwargs)
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value
        self.magnitude = max_param_value - min_param_value
        self.w_end = w_end
        self.w_start = w_start
        self.a_start = a_start
        self.a_end = a_end
        self.k = k

        self.max_time_steps = max_time_steps
        self.current_time_step = 0

    def w(self) -> float:
        """
        Calculate the inertial weight of the particle at the current time step.
        """
        return self.w_end + (self.w_start - self.w_end) * (
                    self.max_time_steps - self.current_time_step) / self.max_time_steps

    def r1(self):
        """Calculate the r1 value for the current timestep"""
        return (self.a_start - self.a_end) * math.exp(
            -self.current_time_step ** 2 / (self.k * self.max_time_steps) ** 2) + self.a_end

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single optimization step.

        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        r3 = 2 * torch.rand((1,))
        r2 = 2 * torch.pi * torch.rand((1,))
        use_sine = torch.rand((1,)).item() < .5
        r1 = self.r1()
        w = self.w()
        self.current_time_step += 1
        return super().step(closure, particle_step_kwargs={'r1': r1,
                                                           'r2': r2,
                                                           'r3': r3,
                                                           'use_sine': use_sine,
                                                           'w': w
                                                           })
