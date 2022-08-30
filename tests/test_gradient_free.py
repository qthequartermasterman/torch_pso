from copy import copy, deepcopy
import random
from typing import Type

import torch
from torch import nn
from torch.nn import Sequential, Linear, MSELoss
from torch.optim import Optimizer

from tests.test_utils import optimizer_tests
from torch_pso import GenericPSO
from torch_pso.optim.GenericPSO import clone_param_groups


def test_closure_function():
    """Ensure that Evaluating a closure function after changing model parameters yields a different result"""
    torch.manual_seed(0)
    net: nn.Module = Sequential(Linear(3, 16), Linear(16, 1))
    x = torch.rand((3,))

    def closure():
        print('param', next(net.parameters())[0])
        loss = net(x)
        print('loss', loss)
        return loss

    loss1 = closure()
    net = Sequential(Linear(3, 16), Linear(16, 1))
    loss2 = closure()
    net = Sequential(Linear(3, 16), Linear(16, 1))
    loss3 = closure()
    params = list(net.parameters())
    params[0].data = torch.rand_like(params[0].data).detach()
    loss4 = closure()

    assert not torch.allclose(loss1, loss2)
    assert not torch.allclose(loss1, loss3)
    assert not torch.allclose(loss2, loss3)
    assert not torch.allclose(loss3, loss4)


@optimizer_tests()
def test_grad_agnostic(optimizer_type: Type[GenericPSO]) -> None:
    """Ensure that evaluating the grad of the loss doesn't change the result of the PSO training"""
    def train_net(use_grad=False) -> nn.Module:
        torch.manual_seed(0)
        random.seed(0)
        net: nn.Module = Sequential(Linear(3, 16), Linear(16, 3))
        optim: GenericPSO = optimizer_type(net.parameters(), num_particles=3)

        criterion = MSELoss()
        target = torch.rand((3,)).round()
        x = torch.rand((3,))

        def closure():
            # Clear any grads from before the optimization step, since we will be changing the parameters
            loss = criterion(net(x), target)
            print('loss', loss)
            return loss

        if use_grad:
            net.train()
            x.requires_grad_(True)

        for i in range(15):
            print(f'{i} before')
            if use_grad:
                closure().backward()
            optim.step(closure)

        return net

    net = train_net(False)
    net_grad = train_net(True)

    for p, p_grad in zip(net.parameters(), net_grad.parameters()):
        print(p, p_grad)
        assert torch.allclose(p, p_grad)
