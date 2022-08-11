from typing import Callable, List, Dict, Iterable

import torch

from .GenericPSO import clone_param_groups, _initialize_param_groups, GenericParticle, GenericPSO


class AcceleratedParticle(GenericParticle):
    def __init__(self,
                 param_groups,
                 alpha: float,
                 beta: float,
                 decay_parameter:float,
                 max_param_value: float,
                 min_param_value: float):
        if not 0 < beta < 1:
            raise ValueError(f'Beta must be a positive constant, not of value {beta}.')
        if not 0 < decay_parameter < 1:
            raise ValueError(f'Decay parameter must be a positive constant, not of value {decay_parameter}.')

        self.param_groups = param_groups
        self.alpha = alpha
        self.beta = beta
        self.decay_parameter = decay_parameter
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value

        self.position = _initialize_param_groups(param_groups, max_param_value, min_param_value)

        self.best_known_position = clone_param_groups(self.position)
        self.best_known_loss_value = torch.inf

    def step(self, closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict]) -> torch.Tensor:
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
                rand_group = torch.rand_like(p)
                new_position = (1-self.beta)*p + self.beta*gb + self.alpha*rand_group
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


class AcceleratedPSO(GenericPSO):
    """
    ChaoticPSO is a variation on the original Particle Swarm Optimization developed to quickly train SVMs. It simplifies
    the original PSO algorithm--most notably eschewing the need for velocity states.

    Original Paper:
    Yang, X. S., Deb, S., and Fong, S., (2011), Accelerated Particle Swarm Optimization and Support Vector Machine for
    Business Optimization and Applications, in: NDT2011, CCIS 136, Springer, pp. 53-66 (2011)

    https://arxiv.org/abs/1203.6577
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 num_particles: int = 100,
                 alpha: float = 0.3,
                 beta: float = 0.45,
                 decay_parameter: float = 0.7,
                 max_param_value: float = -10,
                 min_param_value: float = 10):
        particle_kwargs = {'alpha': alpha,
                           'beta': beta,
                           'decay_parameter': decay_parameter,
                           'max_param_value': max_param_value,
                           'min_param_value': min_param_value,
                           }
        super().__init__(params, num_particles, particle_class=AcceleratedParticle, particle_kwargs=particle_kwargs)
