from functools import reduce
from typing import Callable, List, Dict, Iterable, Union

import torch

from .GenericPSO import clone_param_groups, _initialize_param_groups, GenericParticle, GenericPSO


def unit_difference_of_two_param_groups_lists(b: List[Dict], x: List[Dict]) -> List[Dict]:
    diff = difference_of_two_param_groups_lists(b, x)
    mag = magnitude_param_groups_list(diff)
    return multiply_param_groups_by_scalar(diff, mag)


def sum_of_two_param_groups_lists(b: List[Dict], x: List[Dict]) -> List[Dict]:
    new_param_groups = []
    for b_group, x_group in zip(b, x):
        b_params = b_group['params']
        x_params = x_group['params']

        new_params = [b_param + x_param for b_param, x_param in zip(b_params, x_params)]

        new_group = {'params': new_params}
        new_param_groups.append(new_group)

    return new_param_groups


def difference_of_two_param_groups_lists(b: List[Dict], x: List[Dict]) -> List[Dict]:
    new_param_groups = []
    for b_group, x_group in zip(b, x):
        b_params = b_group['params']
        x_params = x_group['params']

        new_params = [b_param - x_param for b_param, x_param in zip(b_params, x_params)]
        new_group = {'params': new_params}
        new_param_groups.append(new_group)

    return new_param_groups


def magnitude_param_groups_list(x: List[Dict]) -> torch.Tensor:
    """
    Calculate the euclidean magnitude of a param_group_list as if it were a single vector in R^n.
    We can iterate over every parameter, get that parameter's norm squared, sum all of those, and then take the square
    root.
    """
    running_total = torch.tensor(0.)
    for x_group in x:
        x_params = x_group['params']
        running_total += sum(torch.linalg.norm(x_param) ** 2 for x_param in x_params)

    return running_total.sqrt()


def multiply_param_groups_by_scalar(x: List[Dict], scalar: Union[float, torch.Tensor]) -> List[Dict]:
    new_param_groups = []
    for x_group in x:
        x_params = x_group['params']

        new_params = [scalar * x_param for x_param in x_params]

        new_group = {'params': new_params}
        new_param_groups.append(new_group)

    return new_param_groups


class DolphinPodParticle(GenericParticle):
    def __init__(self,
                 param_groups,
                 time_step: float,
                 xi: float,
                 k: float,
                 h: float,
                 max_param_value: float,
                 min_param_value: float):

        self.xi = xi
        self.k = k
        self.h = h
        self.param_groups = param_groups
        self.max_param_value = max_param_value
        self.min_param_value = min_param_value
        # TODO Assert timestep is small enough
        self.time_step = time_step

        # TODO: Initialize params according to paper
        self.position = _initialize_param_groups(param_groups, max_param_value, min_param_value)
        self.velocity = _initialize_param_groups(param_groups, 0, 0)
        self.current_loss_value = torch.inf

        self.best_known_position = clone_param_groups(self.position)
        self.best_known_loss_value = torch.inf

        self.worst_known_position = clone_param_groups(self.position)
        self.worst_known_loss_value = -torch.inf

    def step(self,
             closure: Callable[[], torch.Tensor],
             global_best_param_groups: List[Dict],
             pod_attractive_force: List[Dict],
             food_attractive_force: List[Dict],
             ) -> torch.Tensor:
        """
        Particle will take one step.
        :param closure: A callable that reevaluates the model and returns the loss.
        :param global_best_param_groups: List of param_groups that yield the best found loss globally
        :return:
        """

        # TODO: Update velocity and position

        for (position_group,
             velocity_group,
             pod_attractive_group,
             food_attractive_group,
             master) in zip(self.position,
                            self.velocity,
                            pod_attractive_force,
                            food_attractive_force,
                            self.param_groups):

            position_group_params = position_group['params']
            velocity_group_params = velocity_group['params']
            pod_params = pod_attractive_group['params']
            food_params = food_attractive_group['params']
            master_params = master['params']

            new_position_params = []
            new_velocity_params = []
            for p, v, pod, food, m in zip(position_group_params, velocity_group_params, pod_params, food_params,
                                          master_params):
                new_velocity = (1 - self.xi * self.time_step) * v + self.time_step * (-self.k * pod + self.h * food)
                new_velocity_params.append(new_velocity)
                new_position = p + new_velocity * self.time_step
                new_position_params.append(new_position)
                m.data = new_position.data  # Update the model, so we can use it for calculating loss
            position_group['params'] = new_position_params
            velocity_group['params'] = new_velocity_params

        self._update_params()

        # Calculate new loss after moving and update the best known position if we're in a better spot
        new_loss = closure()
        if new_loss < self.best_known_loss_value:
            self.best_known_position = clone_param_groups(self.position)
            self.best_known_loss_value = new_loss
        if new_loss > self.worst_known_loss_value:
            self.worst_known_position = clone_param_groups(self.position)
            self.worst_known_loss_value = new_loss
        self.current_loss_value = new_loss
        return new_loss

    def evaluate(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        for (position_group,
             master) in zip(self.position,
                            self.param_groups):

            position_group_params = position_group['params']
            master_params = master['params']

            for p, m in zip(position_group_params,
                            master_params):
                m.data = p.data  # Update the model, so we can use it for calculating loss

        self._update_params()

        # Calculate new loss after moving and update the best known position if we're in a better spot
        new_loss = closure()
        if new_loss < self.best_known_loss_value:
            self.best_known_position = clone_param_groups(self.position)
            self.best_known_loss_value = new_loss
        if new_loss > self.worst_known_loss_value:
            self.worst_known_position = clone_param_groups(self.position)
            self.worst_known_loss_value = new_loss
        self.current_loss_value = new_loss
        return new_loss


class DolphinPodOptimizer(GenericPSO):
    """
    Dolphin Pod Optimization is a nature-inspired, deterministic, global, and derivative-free optimization method,
    inspired by the hunting methods of dolphins. Although the paper does not use particle swarm terminology, the method
    can be implemented as a PSO variation.

    Original Paper:
    Serani, A., & Diez, M. (2017, September). Dolphin pod optimization. In International Workshop on Machine Learning,
    Optimization, and Big Data (pp. 50-62). Springer, Cham.

    https://www.researchgate.net/profile/Andrea-Serani/publication/319750174_Dolphin_Pod_Optimization_A_Nature-Inspired_
    Deterministic_Algorithm_for_Simulation-Based_Design/links/59bb98cda6fdcca8e5614dcc/Dolphin-Pod-Optimization-A-Nature
    -Inspired-Deterministic-Algorithm-for-Simulation-Based-Design.pdf
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 num_particles: int = 10,
                 alpha: float = 0.5,
                 time_step: float = .001,
                 xi: float = 2.,
                 k: float = .2,
                 h: float = .2,
                 max_param_value: float = -10,
                 min_param_value: float = 10):
        # TODO: Fix hyperparameter initialization, provide p and q instead of k h and timestep
        self.alpha = alpha
        particle_kwargs = {'xi': xi, 'k': k, 'h': h, 'time_step': time_step,
                           'max_param_value': max_param_value,
                           'min_param_value': min_param_value,
                           }
        super().__init__(params, num_particles, particle_class=DolphinPodParticle, particle_kwargs=particle_kwargs)
        self.worst_current_global_param_groups = clone_param_groups(self.param_groups)
        self.worst_current_global_loss_value = -torch.inf

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Performs a single optimization step.

        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        self.populate_best_known_values(closure)

        for particle in self.particles:
            # TODO: Evaluate the attraction forces delta_j and phi_j
            self.worst_current_global_loss_value = -torch.inf  # Reset the worst particle loss for each step
            pod_attraction_force: List[Dict] = self._calculate_pod_attraction(particle)
            food_attraction_force: List[Dict] = self._calculate_food_attraction(particle)

            particle_loss = particle.step(closure,
                                          self.best_known_global_param_groups,
                                          pod_attraction_force,
                                          food_attraction_force)
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss
            if particle_loss > self.worst_current_global_loss_value:
                self.worst_current_global_param_groups = clone_param_groups(particle.position)
                self.worst_current_global_loss_value = particle_loss

        self._update_master_parms()
        print([particle.position for particle in self.particles])
        print(self.param_groups)
        print()
        return closure()  # loss = closure()

    def _calculate_pod_attraction(self, particle: DolphinPodParticle) -> List[Dict]:
        """Sum of (x_j-x_i) for given particle position x_j and all other particles x_i"""
        x_j = particle.position
        x_i_list = [p.position for p in self.particles]
        new_param_groups = []
        for x_j_group, *x_i_groups in zip(x_j, *x_i_list):

            x_j_params = x_j_group['params']
            x_i_params_list = [x_i_group['params'] for x_i_group in x_i_groups]

            new_params = []
            for x_j, *x_i_param_list in zip(x_j_params, *x_i_params_list):
                new_sum_param = sum(x_j - x_i for x_i in x_i_param_list)
                new_params.append(new_sum_param)

            new_group = {'params': new_params}
            new_param_groups.append(new_group)

        return new_param_groups

    def _calculate_food_attraction(self, particle: DolphinPodParticle) -> List[Dict]:
        addends = []
        for particle_i in self.particles:
            f_hat = self.f_hat(particle, particle_i)
            direction = unit_difference_of_two_param_groups_lists(particle_i.best_known_position, particle.position)
            magnitude_of_diff = magnitude_param_groups_list(
                difference_of_two_param_groups_lists(particle_i.best_known_position, particle.position))
            coefficient = 2 * f_hat / (1 + magnitude_of_diff ** self.alpha)
            addend = multiply_param_groups_by_scalar(direction, coefficient)
            addends.append(addend)
        return reduce(sum_of_two_param_groups_lists, addends)

    def f_hat(self, particle1: DolphinPodParticle, particle2: DolphinPodParticle) -> float:
        """
        F-hat roughly measures how small is the difference between particle 1's position and particle2's best
        position compared to the absolute best and worst
        """
        dynamic_norm_term = self.worst_current_global_loss_value - self.best_known_global_loss_value
        calculation = (particle1.current_loss_value - particle2.best_known_loss_value) / dynamic_norm_term
        # If calculation is nan (i.e. one of the best known values is inf), then downstream computation will break
        if torch.isnan(calculation):
            # return torch.clamp(torch.tensor(calculation), -1e6, 1e6).item()
            raise ValueError(f'F-hat calculation is nan {calculation}' )
        return calculation

    def populate_best_known_values(self, closure: Callable[[], torch.Tensor]) -> None:
        for particle in self.particles:
            particle_loss = particle.evaluate(closure)
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss
            if particle_loss > self.worst_current_global_loss_value:
                self.worst_current_global_param_groups = clone_param_groups(particle.position)
                self.worst_current_global_loss_value = particle_loss
        self._update_master_parms()
