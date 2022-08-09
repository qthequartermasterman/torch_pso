from abc import ABC, abstractmethod, ABCMeta
from typing import Dict, List, Callable, Type, Iterable, Optional

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


class GenericParticle(ABC):
    def __init__(self, *args, **kwargs):
        self.position: List[Dict] = []

    @abstractmethod
    def step(self, closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict]) -> torch.Tensor:
        """
        Particle will take one step.
        :param closure: A callable that reevaluates the model and returns the loss.
        :param global_best_param_groups: List of param_groups that yield the best found loss globally
        :return:
        """
        pass


class GenericPSO(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], num_particles: int, particle_class: Type[GenericParticle],
                 particle_args: Optional[List] = None, particle_kwargs: Optional[Dict] = None):
        defaults = {}
        super().__init__(params, defaults)
        if particle_args is None:
            particle_args = []
        if particle_kwargs is None:
            particle_kwargs = {}
        self.particles = [particle_class(self.param_groups, *particle_args, **particle_kwargs)
                          for _ in range(num_particles)]

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

    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)
