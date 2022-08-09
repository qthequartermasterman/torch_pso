import random
from typing import Callable, Iterable

import torch

from .ParticleSwarmOptimizer import ParticleSwarmOptimizer, Particle
from .GenericPSO import clone_param_group, clone_param_groups


class GenerationalPSO(ParticleSwarmOptimizer):
    """
    Generational PSO is a modification of the naive Particle Swarm Optimization Algorithm, where a certain percentage of
    randomly-chosen, low-performing particles are re-initialized after each step.

    This is a sample algorithm designed by Andrew Sansom with the sole purpose of demonstrating an example of
    alternate PSO algorithms.
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 inertial_weight: float = .9,
                 cognitive_coefficient: float = 1.,
                 social_coefficient: float = 1.,
                 num_particles: int = 100,
                 max_param_value: float = 10.,
                 min_param_value: float = -10.,
                 generational_turnover_ratio: float = .3,
                 keep_top_performers: int = 10):
        super().__init__(params, inertial_weight, cognitive_coefficient, social_coefficient, num_particles,
                         max_param_value, min_param_value)
        self.generational_turnover_ratio = generational_turnover_ratio
        self.keep_top_performers = keep_top_performers

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single optimization step.

        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        losses = {}
        for i, particle in enumerate(self.particles):
            particle_loss = particle.step(closure, self.best_known_global_param_groups)
            losses[i] = particle_loss
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss

        # set the module's parameters to be the best performing ones
        for master_group, best_group in zip(self.param_groups, self.best_known_global_param_groups):
            clone = clone_param_group(best_group)['params']
            for i in range(len(clone)):
                master_group['params'][i].data = clone[i].data

        # Respawn a certain proportion of the worst performing particles, chosen at random
        best_performers_indices = list(sorted(losses, key=losses.get, reverse=True))
        bottom_performers = best_performers_indices[self.keep_top_performers:]
        indices_to_respawn = random.sample(bottom_performers, round(self.generational_turnover_ratio * len(losses)))
        for index in indices_to_respawn:
            self.particles[index] = Particle(self.param_groups,
                                             self.inertial_weight,
                                             self.cognitive_coefficient,
                                             self.social_coefficient,
                                             max_param_value=self.max_param_value,
                                             min_param_value=self.min_param_value)

        return closure()  # loss = closure()
