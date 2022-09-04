from .GenericPSO import GenericPSO, GenericParticle, _initialize_param_groups, compare_param_groups
from torch import Tensor, tensor, inf
from typing import Callable, List, Sequence, Tuple


def sample_space_and_choose_best_spawn(
        optimizer: GenericPSO,
        closure: Callable[[], Tensor],
        num_samples: int,
        max_param_value: float,
        min_param_value: float,
        initializer_func=_initialize_param_groups
) -> Sequence[GenericParticle]:
    """
    Given a particle swarm optimizer, sample its parameter space, and set its particle to the best performers in that
    sample. This is basically initializing the optimizer's particles with a monte carlo search.

    :param optimizer: A particle swarm optimizer (subclassed from GenericPSO)
    :param closure: function to determine the loss of the optimizer's parameters
    :param num_samples: number of samples to take
    :param max_param_value: Maximum value of the parameters in the search space
    :param min_param_value: Minimum value of the parameters in the search space
    :return:
    """
    sampled_param_groups = [optimizer.param_groups] + [initializer_func(optimizer.param_groups,
                                                                        max_param_value,
                                                                        min_param_value
                                                                        )
                                                       for _ in range(num_samples)]
    # print()
    # print('sampled params groups', [p[0]['params'][0][0] for p in sampled_param_groups])
    losses: List[Tensor] = [tensor(-inf) for _ in range(len(sampled_param_groups))]
    # print('losses', losses)
    for i, param_groups in enumerate(sampled_param_groups):
        optimizer._update_master_parms(param_groups)
        assert compare_param_groups(optimizer.param_groups, param_groups)
        losses[i] = closure()
    # print('losses', losses)

    num_particles = len(optimizer.particles)
    best_losses_indices: List[Tuple[int, Tensor]] = sorted(enumerate(losses),
                                                           key=lambda x: x[
                                                               1])  # Figure out why this type is having a fit

    # print('best_losses', best_losses_indices)
    # select the best num_particles number of losses from best_losses_indices
    for i in range(num_particles):
        position_index, loss = best_losses_indices[i]
        # print('best losses i', i, position_index, loss, sampled_param_groups[position_index])
        optimizer.particles[i].position = sampled_param_groups[position_index]

    best_position_index, best_loss = best_losses_indices[0]
    if best_loss < optimizer.best_known_global_loss_value:
        # print('best losses 0', best_loss, optimizer.best_known_global_loss_value, sampled_param_groups[best_position_index])
        optimizer.best_known_global_loss_value = best_loss
        optimizer.best_known_global_param_groups = sampled_param_groups[best_position_index]
        optimizer._update_master_parms(optimizer.best_known_global_param_groups)

    # print('optimizer.best_known_global_loss_value',optimizer.best_known_global_loss_value)
    # print('optimizer.best_known_global_param_groups',optimizer.best_known_global_param_groups)
    # print('param_groups', optimizer.param_groups)

    # if the best loss is lower than the optimizer's best known global loss, update the best known global
    # loss and its position

    return optimizer.particles
