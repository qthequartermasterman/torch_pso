from typing import Callable, Iterable, List, Dict

import torch

from .ParticleSwarmOptimizer import ParticleSwarmOptimizer
from .GenericPSO import clone_param_group, clone_param_groups


class RingTopologyPSO(ParticleSwarmOptimizer):
    """
    Ring Topology PSO is a modification of the naive Particle Swarm Optimization Algorithm, where instead of feeding
    each particle the global optimum, each particle only receives the minimum of itself and two other particles. These
    two particles are the same through the entire iteration.

    This is called a "ring" because the network graph showing these connections has a single cycle.
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 num_neighbors: int = 2,
                 inertial_weight: float = .9,
                 cognitive_coefficient: float = 1.,
                 social_coefficient: float = 1.,
                 num_particles: int = 100,
                 max_param_value: float = 10.,
                 min_param_value: float = -10.):
        super().__init__(params, inertial_weight, cognitive_coefficient, social_coefficient, num_particles,
                         max_param_value, min_param_value)
        self.losses = {i: (particle.position, torch.inf) for i, particle in enumerate(self.particles)}
        self.num_neighbors = num_neighbors

    def _find_minimum_of_neighbors(self, particle_index: int) -> List[Dict]:
        neighbors = [(particle_index + i) % len(self.particles) for i in range(self.num_neighbors)]

        best = sorted([self.losses[n] for n in neighbors],
                      key=lambda x: x[1],
                      reverse=True)[0]
        return clone_param_groups(best[0])

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single optimization step.

        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        losses = {}
        for i, particle in enumerate(self.particles):
            particle_loss = particle.step(closure, self._find_minimum_of_neighbors(i))
            losses[i] = (particle.position, particle_loss)
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss

        # set the module's parameters to be the best performing ones
        for master_group, best_group in zip(self.param_groups, self.best_known_global_param_groups):
            clone = clone_param_group(best_group)['params']
            for i in range(len(clone)):
                master_group['params'][i].data = clone[i].data
        self.losses = losses
        return closure()  # loss = closure()
