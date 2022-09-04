from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Type, Iterable, Optional, Any, TypeVar, Sequence, ClassVar

import torch
from torch import Tensor
from torch.optim import Optimizer


def clamp_param_groups(param_groups: List[Dict],
                       max_param_value: float | Tensor,
                       min_param_value: float | Tensor,
                       inplace=True) -> List[Dict]:
    if not inplace:
        param_groups = clone_param_groups(param_groups)
    for group in param_groups:
        group['params'] = [torch.clamp(p, min=min_param_value, max=max_param_value) for p in group['params']]
    return param_groups


def compare_param_groups(param_groups_left: List[Dict], param_groups_right: List[Dict]) -> bool:
    """
    Check that each param in two param_groups are equal
    :param param_groups_left: Dict containing param_groups
    :param param_groups_right: Dict containing param_groups
    :return: True if param_groups equal else False
    """
    for group_left, group_right in zip(param_groups_left, param_groups_right):
        params_left, params_right = group_left['params'], group_right['params']
        if any(not torch.allclose(left, right) for left, right in zip(params_left, params_right)):
            return False
    return True


def clone_param_group(param_group: Dict) -> Dict:
    """
    Clone each param in a param_group and return a new dict containing the clones
    :param param_group: Dict containing param_groups
    :return: cloned param_group dict
    """
    new_group = {key: value for key, value in param_group.items() if key != 'params'}
    new_group['params'] = [param.clone() for param in param_group['params']]
    # new_group['params'] = [param.detach().clone() for param in param_group['params']]
    return new_group


def clone_param_groups(param_groups: List[Dict]) -> List[Dict]:
    """
    Make a list of clones for each param_group in a param_groups list.
    :param param_groups: List of dicts containing param_groups
    :return: cloned list of param_groups
    """
    return [clone_param_group(param_group) for param_group in param_groups]


def _initialize_param_groups(param_groups: List[Dict], max_param_value: float, min_param_value: float) -> List[Dict]:
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

    def _initialize_params(param: torch.nn.Parameter) -> torch.nn.Parameter:
        return magnitude * torch.rand_like(param) - magnitude / 2 + mean_value

    # Make sure we get a clone, so we don't overwrite the original params in the module
    param_groups = clone_param_groups(param_groups)
    for group in param_groups:
        group['params'] = [_initialize_params(p) for p in group['params']]

    return param_groups


class GenericParticle(ABC):
    """
    Generic particle class that contains functionality to (almost) all particle types
    """

    def __init__(self, *args, **kwargs):
        self.position: List[Dict] = []
        self.param_groups: List[Dict] = []
        self.max_param_value: float | Tensor = 1
        self.min_param_value: float | Tensor = -1

    @abstractmethod
    def step(self, closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict]) -> torch.Tensor:
        """
        Particle will take one step.
        :param closure: A callable that reevaluates the model and returns the loss.
        :param global_best_param_groups: List of param_groups that yield the best found loss globally
        :return:
        """
        pass

    def _update_params(self) -> None:
        # Really crummy way to update the parameter weights in the original model.
        # Simply changing self.param_groups doesn't update the model.
        # Nor does changing its elements or the raw values of 'param' of the elements.
        # We have to change the underlying tensor data to point to the new positions
        # for i in range(len(self.position)):
        #     for j in range(len(self.param_groups[i]['params'])):
        #         self.param_groups[i]['params'][j].data = self.param_groups[i]['params'][j].data
        # self.param_groups = clamp_param_groups(self.param_groups, self.max_param_value, self.min_param_value)
        pass

    def set_params(self, params: List[Dict]):
        self.position = clone_param_groups(params)


ParticleType = TypeVar('ParticleType', bound=Type[GenericParticle])


class GenericPSO(Optimizer):
    """
    Generic PSO contains functionality common to (almost) all particle swarm optimization algorithms.
    """

    subclasses: ClassVar[List[Type['GenericPSO']]] = []

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            num_particles: int = 100,
            particle_class: Type[GenericParticle] = GenericParticle,
            # particle_class: Particle = GenericParticle,
            particle_args: Optional[List] = None,
            particle_kwargs: Optional[Dict] = None,
    ):
        defaults: Dict[str, Any] = {}
        super().__init__(params, defaults)
        if particle_args is None:
            particle_args = []
        if particle_kwargs is None:
            particle_kwargs = {}
        self.particles: Sequence[GenericParticle] = [
            particle_class(self.param_groups, *particle_args, **particle_kwargs) for _ in range(num_particles)
        ]
        # We always want the initial params to be one of the particles
        self.particles[0].set_params(self.param_groups)

        self.best_known_global_param_groups = clone_param_groups(self.param_groups)
        self.best_known_global_loss_value: torch.Tensor = torch.tensor(torch.inf)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """
        Performs a single optimization step.

        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        if closure is None:
            raise TypeError('Closures are required for Particle Swarm Optimizers')
        for particle in self.particles:
            particle_loss = particle.step(closure, self.best_known_global_param_groups)
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss

        self._update_master_parms()

        return closure()  # loss = closure()

    def _update_master_parms(self, new_param_groups: Optional[List[Dict]] = None) -> None:
        """Set the module's parameters to be the best performing ones."""
        new_param_groups = new_param_groups or self.best_known_global_param_groups
        for master_group, best_group in zip(self.param_groups, new_param_groups):
            clone = clone_param_group(best_group)['params']
            for i in range(len(clone)):
                master_group['params'][i].data = clone[i].data

    def __init_subclass__(cls, **kwargs) -> None:
        """Register all subclasses, so we can easily run the same test benchmarks on every subclass."""
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)
