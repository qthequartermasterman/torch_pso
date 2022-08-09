from typing import Callable, List, Dict, Iterable

import torch

from .GenericPSO import clone_param_groups, _initialize_param_groups, GenericParticle, GenericPSO


class ChaoticParticle(GenericParticle):
    def __init__(self,
                 param_groups,
                 a: float,
                 b: float,
                 c: float,
                 beta: float,
                 k: float,
                 epsilon: float,
                 i0: float,
                 max_param_value: float,
                 min_param_value: float):
        if a <= 0:
            raise ValueError(f'A must be a positive constant, not of value {a}.')
        if b <= 0:
            raise ValueError(f'B must be a positive constant, not of value {b}.')
        if c <= 0:
            raise ValueError(f'C must be a positive constant, not of value {c}.')
        if not 0 < beta < 1:
            raise ValueError(f'Beta must be a positive constant, not of value {beta}.')

        self.param_groups = param_groups
        self.a = a
        self.b = b
        self.c = c
        self.beta = beta
        self.k = k
        self.i0 = i0
        self.epsilon = epsilon
        self._z = 1  # This value is never specified in the paper.
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value

        magnitude = abs(max_param_value - min_param_value)
        self.position = _initialize_param_groups(param_groups, max_param_value, min_param_value)
        self.velocity = _initialize_param_groups(param_groups, magnitude, -magnitude)
        # I'm not sure what x_ip and u_ip represent, but their iterations are defined mathematically
        # Their initial values are not defined, however
        self.x_ip = _initialize_param_groups(param_groups, max_param_value, min_param_value)
        self.u_ip = _initialize_param_groups(param_groups, magnitude, -magnitude)

        self.best_known_position = clone_param_groups(self.position)
        self.best_known_loss_value = torch.inf

    def _unit_interval_to_min_max(self, x):
        return self.min_param_value + x * (self.max_param_value - self.min_param_value)

    def _min_max_to_unit_interval(self, x):
        return (x - self.min_param_value) / (self.max_param_value - self.min_param_value)

    def step(self, closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict]) -> torch.Tensor:
        """
        Particle will take one step.
        :param closure: A callable that reevaluates the model and returns the loss.
        :param global_best_param_groups: List of param_groups that yield the best found loss globally
        :return:
        """
        # Because our parameters are not a single tensor, we have to iterate over each group, and then each param in
        # each group.
        for (position_group,
             velocity_group,
             x_ip_group,
             u_ip_group,
             personal_best,
             global_best,
             master) in zip(self.position,
                            self.velocity,
                            self.u_ip,
                            self.x_ip,
                            self.best_known_position,
                            global_best_param_groups,
                            self.param_groups):

            position_group_params = position_group['params']
            velocity_group_params = velocity_group['params']
            x_ip_group_params = x_ip_group['params']
            u_ip_group_params = u_ip_group['params']
            personal_best_params = personal_best['params']
            global_best_params = global_best['params']
            master_params = master['params']

            new_position_params = []
            new_velocity_params = []
            new_x_ip_params = []
            new_u_ip_params = []
            for x, u, x_ip, u_ip, pb, gb, m in zip(position_group_params,
                                                   velocity_group_params,
                                                   x_ip_group_params,
                                                   u_ip_group_params,
                                                   personal_best_params,
                                                   global_best_params,
                                                   master_params):
                # rand_personal = torch.rand_like(u)
                # rand_group = torch.rand_like(u)
                # new_velocity = (self.inertial_weight * u
                #                 + self.cognitive_coefficient * rand_personal * (pb - x)
                #                 + self.social_coefficient * rand_group * (gb - x)
                #                 )
                # new_position = x + new_velocity
                # All the below calculations assume x is between 0 and 1
                x = self._min_max_to_unit_interval(x)
                x_ip = self._min_max_to_unit_interval(x_ip)

                delta_u = -(2 * self.a * (x - gb) + 2 * self.c * (x - x_ip)) / (1 + self.k * (self.a + self.b))
                new_velocity = u + delta_u - self._z * (x - self.i0)
                new_position = torch.clamp(self.k * new_velocity, min=0, max=1)
                new_position = self._unit_interval_to_min_max(new_position)

                delta_u_ip = -(2 * self.b * (x - pb) + 2 * self.c * (x - x_ip)) / (1 + self.k * (self.b + self.c))
                new_u_ip = u_ip + delta_u_ip - self._z * (x_ip - self.i0)
                new_x_ip = torch.clamp(self.k * new_u_ip, min=0, max=1)
                new_x_ip = self._unit_interval_to_min_max(new_x_ip)

                self._z *= 1 - self.beta

                new_velocity_params.append(new_velocity)
                new_position_params.append(new_position)
                new_x_ip_params.append(new_x_ip)
                new_u_ip_params.append(new_u_ip)
                m.data = new_position.data  # Update the model, so we can use it for calculating loss
            position_group['params'] = new_position_params
            velocity_group['params'] = new_velocity_params
            x_ip_group['params'] = new_x_ip_params
            u_ip_group['params'] = new_u_ip_params

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


class ChaoticPSO(GenericPSO):
    """
    ChaoticPSO is a variation on the original Particle Swarm Optimization inspired by techniques in training
    Hopfield Networks. It introduces some chaos-like mechanics into the optimization, theoretically improving
    convergence speed in some contexts.

    Original Paper:
    Sun, Yanxia & Qi, Guoyuan & Wang, Zenghui & Van Wyk, Barend & Hamam, Yskandar. (2009). Chaotic particle
    swarm optimization. 505-510. 10.1145/1543834.1543902.

    https://www.researchgate.net/publication/220741402_Chaotic_particle_swarm_optimization
    """
    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 num_particles: int = 100,
                 a: float = 0.02,
                 b: float = 0.01,
                 c: float = 0.01,
                 beta: float = .001,
                 k: float = 15,
                 epsilon: float = 1.,
                 i0: float = 0.2,
                 max_param_value: float = -10,
                 min_param_value: float = 10):
        particle_kwargs = {'a': a,
                           'b': b,
                           'c': c,
                           'beta': beta,
                           'k': k,
                           'epsilon': epsilon,
                           'i0': i0,
                           'max_param_value': max_param_value,
                           'min_param_value': min_param_value,
                           }
        super().__init__(params, num_particles, particle_class=ChaoticParticle, particle_kwargs=particle_kwargs)
