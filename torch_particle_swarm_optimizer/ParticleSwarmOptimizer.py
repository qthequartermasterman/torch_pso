from typing import List, Dict

import torch
from torch.optim import Optimizer


def clone_param_group(param_group: Dict) -> Dict:
    new_group = {key: value for key, value in param_group.items() if key != 'params'}
    new_group['params'] = [param.detach().clone() for param in param_group['params']]
    return new_group


def clone_param_groups(param_groups: List[Dict]) -> List[Dict]:
    return [clone_param_group(param_group) for param_group in param_groups]


class Particle:
    def __init__(self, param_groups: List[Dict], inertial_weight, cognitive_coefficient, social_coefficient,
                 max_param_value=10, min_param_value=-10):
        magnitude = abs(max_param_value - min_param_value)
        self.param_groups = param_groups
        self.position = self._initialize_position(param_groups, max_param_value, min_param_value)
        self.velocity = self._initialize_position(param_groups, magnitude, -magnitude)
        self.best_known_position = clone_param_groups(self.position)
        self.best_known_loss_value = torch.inf

        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient

    def _initialize_position(self, param_groups: List[Dict], max_param_value, min_param_value) -> List[Dict]:
        magnitude = max_param_value - min_param_value
        mean_value = (max_param_value + min_param_value) / 2

        def _initialize_params(param):
            return magnitude * torch.rand_like(param) - magnitude / 2 + mean_value

        # Make sure we get a clone, so we don't overwrite the original params in the module
        param_groups = clone_param_groups(param_groups)
        for group in param_groups:
            group['params'] = [_initialize_params(p) for p in group['params']]

        return param_groups

    def step(self, closure, global_best_param_groups):
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
                # print('position group before', position_group, self.position)
                rand_personal = torch.rand_like(v)
                rand_group = torch.rand_like(v)
                new_velocity = (self.inertial_weight * v
                                + self.cognitive_coefficient * rand_personal * (pb - p)
                                + self.social_coefficient * rand_group * (gb - p)
                                )
                # print('old velocity', v, 'new_velocity', new_velocity)
                new_velocity_params.append(new_velocity)
                new_position = p + new_velocity
                new_position_params.append(new_position)
                m.data = new_position.data  # Update the model, so we can use it for calculating loss
            position_group['params'] = new_position_params
            velocity_group['params'] = new_velocity_params
            # print('position group after', position_group, self.position)

        for i in range(len(self.position)):
            # print(self.param_groups, self.position)
            for j in range(len(self.param_groups[i]['params'] )):
                self.param_groups[i]['params'][j].data = self.param_groups[i]['params'][j].data

        new_loss = closure()
        # print('new loss', new_loss)
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
    """

    def __init__(self, params, inertial_weight=1., cognitive_coefficient=1., social_coefficient=1., num_particles=100,
                 max_param_value=10, min_param_value=-10):
        self.num_particles = num_particles
        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient

        defaults = {}
        super().__init__(params, defaults)
        print('self.param_groups', self.param_groups)
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
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
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

        print()
        print('best loss', [(i, f'{particle.best_known_loss_value.item():.3f}') for i, particle in enumerate(self.particles)])
        print('best position', [(i, f'{particle.best_known_position[0]["params"][0].item():.3f}') for i, particle in enumerate(self.particles)])
        print('current position', [(i, f'{particle.position[0]["params"][0].item():.3f}') for i, particle in enumerate(self.particles)])
        print('current velocity', [(i, f'{particle.velocity[0]["params"][0].item():.3f}') for i, particle in enumerate(self.particles)])
        print('global', self.best_known_global_loss_value.item(), self.best_known_global_param_groups)

        return closure()  # loss = closure()
