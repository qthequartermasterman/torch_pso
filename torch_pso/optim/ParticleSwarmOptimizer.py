from typing import List, Dict, Callable, Iterable

import torch
from torch.optim import Optimizer


def clone_param_group(param_group: Dict) -> Dict:
    """
    Clone each param in a param_group and return a new dict containing the clones
    :param param_group: Dict containing param_groups
    :return: cloned param_group dict
    """
    new_group = {key: value for key, value in param_group.items() if key != 'params'}
    new_group['params'] = [param.detach().clone() for param in param_group['params']]
    return new_group


def clone_param_groups(param_groups: List[Dict]) -> List[Dict]:
    """
    Make a list of clones for each param_group in a param_groups list.
    :param param_groups: List of dicts containing param_groups
    :return: cloned list of param_groups
    """
    return [clone_param_group(param_group) for param_group in param_groups]


def _initialize_param_groups(param_groups: List[Dict], max_param_value, min_param_value) -> List[Dict]:
    """
    Take a list of param_groups, clone it, and then randomly initialize its parameters with values between
    max_param_value and min_param_value.

    :param param_groups: List of dicts containing param_groups
    :param max_param_value: Maximum value of the parameters in the search space
    :param min_param_value: Minimum value of the parameters in the search space
    :return the new, initialized param_groups
    """
    magnitude = max_param_value - min_param_value
    mean_value = (max_param_value + min_param_value) / 2

    def _initialize_params(param):
        return magnitude * torch.rand_like(param) - magnitude / 2 + mean_value

    # Make sure we get a clone, so we don't overwrite the original params in the module
    param_groups = clone_param_groups(param_groups)
    for group in param_groups:
        group['params'] = [_initialize_params(p) for p in group['params']]

    return param_groups


class Particle:
    r"""
    Algorithm from Wikipedia: https://en.wikipedia.org/wiki/Particle_swarm_optimization
    Let S be the number of particles in the swarm, each having a position xi ∈ ℝn in the search-space
    and a velocity vi ∈ ℝn. Let pi be the best known position of particle i and let g be the best known
    position of the entire swarm.
    The values blo and bup represent the lower and upper boundaries of the search-space respectively.
    The w parameter is the inertia weight. The parameters φp and φg are often called cognitive coefficient and
    social coefficient.

    The termination criterion can be the number of iterations performed, or a solution where the adequate
    objective function value is found. The parameters w, φp, and φg are selected by the practitioner and control
    the behaviour and efficacy of the PSO method.

    for each particle i = 1, ..., S do
        Initialize the particle's position with a uniformly distributed random vector: xi ~ U(blo, bup)
        Initialize the particle's best known position to its initial position: pi ← xi
        if f(pi) < f(g) then
            update the swarm's best known position: g ← pi
        Initialize the particle's velocity: vi ~ U(-|bup-blo|, |bup-blo|)
    while a termination criterion is not met do:
        for each particle i = 1, ..., S do
            for each dimension d = 1, ..., n do
                Pick random numbers: rp, rg ~ U(0,1)
                Update the particle's velocity: vi,d ← w vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)
            Update the particle's position: xi ← xi + vi
            if f(xi) < f(pi) then
                Update the particle's best known position: pi ← xi
                if f(pi) < f(g) then
                    Update the swarm's best known position: g ← pi


    :param param_groups: list of dict containing parameters
    :param inertial_weight: float representing inertial weight of the particles
    :param cognitive_coefficient: float representing cognitive coefficient of the particles
    :param social_coefficient: float representing social coefficient of the particles
    :param max_param_value: Maximum value of the parameters in the search space
    :param min_param_value: Minimum value of the parameters in the search space
    """

    def __init__(self,
                 param_groups: List[Dict],
                 inertial_weight: float,
                 cognitive_coefficient: float,
                 social_coefficient: float,
                 max_param_value: float = 10.,
                 min_param_value: float = -10.):
        magnitude = abs(max_param_value - min_param_value)
        self.param_groups = param_groups
        self.position = _initialize_param_groups(param_groups, max_param_value, min_param_value)
        self.velocity = _initialize_param_groups(param_groups, magnitude, -magnitude)
        self.best_known_position = clone_param_groups(self.position)
        self.best_known_loss_value = torch.inf

        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient

    def step(self, closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict]) -> torch.Tensor:
        """
        Particle will take one step.
        :param closure: A callable that reevaluates the model and returns the loss.
        :param global_best_param_groups: List of param_groups that yield the best found loss globally
        :return:
        """
        # Because our parameters are not a single tensor, we have to iterate over each group, and then each param in
        # each group.
        for position_group, velocity_group, personal_best, global_best, master in zip(self.position, self.velocity,
                                                                                      self.best_known_position,
                                                                                      global_best_param_groups,
                                                                                      self.param_groups):
            position_group_params = position_group['params']
            velocity_group_params = velocity_group['params']
            personal_best_params = personal_best['params']
            global_best_params = global_best['params']
            master_params = master['params']

            new_position_params = []
            new_velocity_params = []
            for p, v, pb, gb, m in zip(position_group_params, velocity_group_params, personal_best_params,
                                       global_best_params, master_params):
                rand_personal = torch.rand_like(v)
                rand_group = torch.rand_like(v)
                new_velocity = (self.inertial_weight * v
                                + self.cognitive_coefficient * rand_personal * (pb - p)
                                + self.social_coefficient * rand_group * (gb - p)
                                )
                new_velocity_params.append(new_velocity)
                new_position = p + new_velocity
                new_position_params.append(new_position)
                m.data = new_position.data  # Update the model, so we can use it for calculating loss
            position_group['params'] = new_position_params
            velocity_group['params'] = new_velocity_params

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


class ParticleSwarmOptimizer(Optimizer):
    r"""
    Algorithm from Wikipedia: https://en.wikipedia.org/wiki/Particle_swarm_optimization
    Let S be the number of particles in the swarm, each having a position xi ∈ ℝn in the search-space
    and a velocity vi ∈ ℝn. Let pi be the best known position of particle i and let g be the best known
    position of the entire swarm.
    The values blo and bup represent the lower and upper boundaries of the search-space respectively.
    The w parameter is the inertia weight. The parameters φp and φg are often called cognitive coefficient and
    social coefficient.

    The termination criterion can be the number of iterations performed, or a solution where the adequate
    objective function value is found. The parameters w, φp, and φg are selected by the practitioner and control
    the behaviour and efficacy of the PSO method.

    for each particle i = 1, ..., S do
        Initialize the particle's position with a uniformly distributed random vector: xi ~ U(blo, bup)
        Initialize the particle's best known position to its initial position: pi ← xi
        if f(pi) < f(g) then
            update the swarm's best known position: g ← pi
        Initialize the particle's velocity: vi ~ U(-|bup-blo|, |bup-blo|)
    while a termination criterion is not met do:
        for each particle i = 1, ..., S do
            for each dimension d = 1, ..., n do
                Pick random numbers: rp, rg ~ U(0,1)
                Update the particle's velocity: vi,d ← w vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)
            Update the particle's position: xi ← xi + vi
            if f(xi) < f(pi) then
                Update the particle's best known position: pi ← xi
                if f(pi) < f(g) then
                    Update the swarm's best known position: g ← pi


    :param params:iterable of parameters to optimize or dicts defining parameter groups
    :param inertial_weight: float representing inertial weight of the particles
    :param cognitive_coefficient: float representing cognitive coefficient of the particles
    :param social_coefficient: float representing social coefficient of the particles
    :param num_particles: int representing the number of particles in the swarm
    :param max_param_value: Maximum value of the parameters in the search space
    :param min_param_value: Minimum value of the parameters in the search space
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 inertial_weight: float = 1.,
                 cognitive_coefficient: float = 1.,
                 social_coefficient: float = 1.,
                 num_particles: int = 100,
                 max_param_value: float = 10.,
                 min_param_value: float = -10.):
        self.num_particles = num_particles
        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient

        defaults = {}
        super().__init__(params, defaults)
        # print('self.param_groups', self.param_groups)
        self.particles = [Particle(self.param_groups,
                                   self.inertial_weight,
                                   self.cognitive_coefficient,
                                   self.social_coefficient,
                                   max_param_value=max_param_value,
                                   min_param_value=min_param_value)
                          for _ in range(self.num_particles)]

        self.best_known_global_param_groups = clone_param_groups(self.param_groups)
        self.best_known_global_loss_value = torch.inf

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single optimization step.

        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        for particle in self.particles:
            particle_loss = particle.step(closure, self.best_known_global_param_groups)
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss

        # set the module's parameters to be the best performing ones
        for master_group, best_group in zip(self.param_groups, self.best_known_global_param_groups):
            clone = clone_param_group(best_group)['params']
            for i in range(len(clone)):
                master_group['params'][i].data = clone[i].data

        return closure()  # loss = closure()
