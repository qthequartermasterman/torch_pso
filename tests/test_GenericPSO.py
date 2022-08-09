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


class QuarticWeightsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((1,)))

    def forward(self, x):
        x = self.weights
        return x ** 2 * (x ** 2 - 1)


@pytest.mark.parametrize('optimizer_type', OPTIMIZERS)
def test_square_converges(optimizer_type):
    net = SquareWeightsModule()
    optim = optimizer_type(net.parameters())

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(net.weights, torch.Tensor([0.]), atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(1000):
        optim.step(closure)
        if torch.allclose(net.weights, torch.Tensor([0.]), atol=1e-4, rtol=1e-3):
            converged = True
            break

    assert converged, net.weights


@pytest.mark.parametrize('optimizer_type', OPTIMIZERS)
def test_quartic_converges(optimizer_type):
    net = QuarticWeightsModule()
    optim = optimizer_type(net.parameters())

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(abs(net.weights), torch.sqrt(torch.Tensor([2])) / 2, atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(1000):
        optim.step(closure)
        if torch.allclose(abs(net.weights), torch.sqrt(torch.Tensor([2])) / 2, atol=1e-4, rtol=1e-3):
            converged = True
            break

    assert converged, net.weights
