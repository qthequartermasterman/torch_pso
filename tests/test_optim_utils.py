from typing import Type

import torch
from torch import nn
from torch.nn import Sequential, Linear, MSELoss
from torch.optim import SGD

from torch_pso import GenericPSO
from torch_pso.optim.GenericPSO import compare_param_groups
from torch_pso.optim.optim_utils import sample_space_and_choose_best_spawn
from .test_utils import optimizer_tests

def test_compare_params():
    """Ensure that our param_groups comparison is correct"""
    torch.manual_seed(0)
    net1: nn.Module = Sequential(Linear(3, 16), Linear(16, 1))
    net2: nn.Module = Sequential(Linear(3, 16), Linear(16, 1))
    optim1 = SGD(net1.parameters(), lr=1e-3)
    optim2 = SGD(net2.parameters(), lr=1e-3)
    assert compare_param_groups(optim1.param_groups, optim1.param_groups)
    assert compare_param_groups(optim2.param_groups, optim2.param_groups)
    assert not compare_param_groups(optim1.param_groups, optim2.param_groups)


@optimizer_tests()
def test_sample_space_and_choose_best_spawn(optimizer_type: Type[GenericPSO]):
    torch.manual_seed(0)
    net: nn.Module = Sequential(Linear(3, 16), Linear(16, 1))
    x = torch.tensor((0., 0., 1.))
    target = torch.tensor((1.,))
    mse = MSELoss()
    num_samples = 100

    def closure():
        # print('param', next(net.parameters())[0])
        loss = mse(net(x), target)
        # print('loss', loss)
        return loss

    parameters = net.parameters()

    optimizer = optimizer_type(parameters, num_particles=1)
    original_particle_position = optimizer.particles[0].position
    original_loss = closure()
    sample_space_and_choose_best_spawn(optimizer, closure, num_samples, max_param_value=1, min_param_value=-1)
    new_particle_position = optimizer.particles[0].position
    new_loss = closure()

    # Make sure the parameters changed
    assert not compare_param_groups(original_particle_position, new_particle_position)

    # Make sure our loss is improved
    assert new_loss <= original_loss

