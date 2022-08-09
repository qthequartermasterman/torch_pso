from typing import Callable, Iterable, Optional

import torch

from .ParticleSwarmOptimizer import ParticleSwarmOptimizer
from .GenericPSO import clone_param_group, clone_param_groups


class AutotuningPSO(ParticleSwarmOptimizer):
    """
    Autotuning Particle Swarm Optimization is a modification of Particle Swarm Optimization where the coefficients
    change over time, as prescribed by Axel Thevenot in his Medium Article entitled ["Particle Swarm Optimization
    Visually Explained"](https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14).
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 num_total_iterations: int = 1000,
                 inertial_weight: float = .9,
                 cognitive_coefficient: float = 1.,
                 social_coefficient: float = 1.,
                 num_particles: int = 100,
                 max_param_value: float = 10.,
                 min_param_value: float = -10.,
                 ):
        super().__init__(params, inertial_weight, cognitive_coefficient, social_coefficient, num_particles,
                         max_param_value, min_param_value)
        self.num_total_iterations = num_total_iterations
        self.current_step = 0

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], n: Optional[int] = None) -> torch.Tensor:
        """
        Performs a single optimization step. This is a standard PSO step, followed by a weight adjustment.

        :param n: integer representing which "step number" this step should be treated as in calculating
            the weight decays
        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        # Calculate the new coefficients for the swarm
        n = n if n is not None else self.current_step
        w_t = 0.4 * (n - self.num_total_iterations) / self.num_total_iterations ** 2 + 0.4
        cognitive_t = -3 * n / self.num_total_iterations + 3.5
        social_t = 3 * n / self.num_total_iterations + 0.5

        self.inertial_weight = w_t
        self.cognitive_coefficient = cognitive_t
        self.social_coefficient = social_t

        # Update the step number

        for particle in self.particles:
            particle_loss = particle.step(closure, self.best_known_global_param_groups)
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss

            # Update the particle coefficients
            particle.inertial_weight = w_t
            particle.cognitive_coefficient = cognitive_t
            particle.social_coefficient = social_t

        # set the module's parameters to be the best performing ones
        for master_group, best_group in zip(self.param_groups, self.best_known_global_param_groups):
            clone = clone_param_group(best_group)['params']
            for i in range(len(clone)):
                master_group['params'][i].data = clone[i].data

        return closure()  # loss = closure()
