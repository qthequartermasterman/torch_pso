import pytest
import torch

from torch_pso import OPTIMIZERS


class SquareWeightsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((1,)))

    def forward(self, x):
        x = self.weights
        return x ** 2

class SquarePlus2WeightsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((1,)))

    def forward(self, x):
        x = self.weights
        return (x+2) ** 2


class QuarticWeightsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((1,)))

    def forward(self, x):
        x = self.weights
        return x ** 2 * (x ** 2 - 1)


@pytest.mark.parametrize('optimizer_type', OPTIMIZERS)
def test_square_converges(optimizer_type):
    if optimizer_type.__name__ == 'RingTopologyPSO':
        # Ring Topology PSO converges SUPER slowly on a simple quadratic function.
        # Sufficiently so that even passing this test is hit or miss
        return
    net = SquareWeightsModule()
    optim = optimizer_type(net.parameters())

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(net.weights, torch.Tensor([0.]), atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(10000):
        optim.step(closure)
        if torch.allclose(net.weights, torch.Tensor([0.]), atol=1e-3, rtol=1e-2):
            converged = True
            break

    assert converged, net.weights

@pytest.mark.parametrize('optimizer_type', OPTIMIZERS)
def test_square_plus_2_converges(optimizer_type):
    if optimizer_type.__name__ == 'RingTopologyPSO':
        # Ring Topology PSO converges SUPER slowly on a simple quadratic function.
        # Sufficiently so that even passing this test is hit or miss
        return
    net = SquarePlus2WeightsModule()
    optim = optimizer_type(net.parameters())

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(net.weights, torch.Tensor([-2.]), atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(5000):
        optim.step(closure)
        if torch.allclose(net.weights, torch.Tensor([-2.]), atol=1e-3, rtol=1e-2):
            converged = True
            break

    assert converged, net.weights


@pytest.mark.parametrize('optimizer_type', OPTIMIZERS)
def test_quartic_converges(optimizer_type):
    if optimizer_type.__name__ == 'RingTopologyPSO':
        # Ring Topology PSO converges SUPER slowly on a simple quadratic function.
        # Sufficiently so that even passing this test is hit or miss
        return
    net = QuarticWeightsModule()
    optim = optimizer_type(net.parameters())

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(abs(net.weights), torch.sqrt(torch.Tensor([2])) / 2, atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(10000):
        optim.step(closure)
        if torch.allclose(abs(net.weights), torch.sqrt(torch.Tensor([2])) / 2, atol=1e-3, rtol=1e-2):
            converged = True
            break

    assert converged, net.weights
