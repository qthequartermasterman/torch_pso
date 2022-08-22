from typing import List

import pytest
import torch
from torch import Tensor

from tests.test_utils import optimizer_tests
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
        return (x + 2) ** 2


class QuarticWeightsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((1,)))

    def forward(self, x):
        x = self.weights
        return x ** 2 * (x ** 2 - 1)


class RastriginModule(torch.nn.Module):
    def __init__(self, num_dimensions, a=10.):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((num_dimensions,)))
        self.num_dimensions = num_dimensions
        self.A = a

    def forward(self, x):
        x = self.weights
        return self.A * self.num_dimensions + (x ** 2 - self.A * torch.cos(2 * torch.pi * x)).sum()


class HimmelblauModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((2,)))

    def forward(self, x):
        x, y = self.weights
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


class GoldsteinPriceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((2,)))

    def forward(self, x):
        x, y = self.weights
        return (1 + (x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30 + (2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_square_converges(optimizer_type):
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
        if torch.allclose(net.weights, torch.Tensor([0.]), atol=1e-2, rtol=1e-1):
            converged = True
            break

    assert converged, net.weights


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_square_plus_2_converges(optimizer_type):
    net = SquarePlus2WeightsModule()
    optim = optimizer_type(net.parameters())

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(net.weights, torch.Tensor([-2.]), atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(1000):
        optim.step(closure)
        if torch.allclose(net.weights, torch.Tensor([-2.]), atol=1e-2, rtol=1e-1):
            converged = True
            break

    assert converged, net.weights


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_quartic_converges(optimizer_type):
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
        if torch.allclose(abs(net.weights), torch.sqrt(torch.Tensor([2])) / 2, atol=1e-2, rtol=1e-1):
            converged = True
            break

    assert converged, net.weights


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_rastrigin_converges(optimizer_type):
    net = RastriginModule(num_dimensions=1)
    optim = optimizer_type(net.parameters())

    global_minimum = torch.zeros_like(net.weights)

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(net.weights, global_minimum, atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(10000):
        optim.step(closure)
        if torch.allclose(net.weights, global_minimum, atol=1e-2, rtol=1e-1):
            converged = True
            break

    assert converged, net.weights


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO', 'GenerationalPSO', 'ChaoticPSO', 'AcceleratedPSO'])
def test_rastrigin3_converges(optimizer_type):
    net = RastriginModule(num_dimensions=3)
    optim = optimizer_type(net.parameters())

    global_minimum = torch.zeros_like(net.weights)

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(net.weights, global_minimum, atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(1000):
        optim.step(closure)
        if torch.allclose(net.weights, global_minimum, atol=1e-2, rtol=1e-1):
            converged = True
            break

    assert converged, net.weights


# All of the algorithms converge really slowly on this test. Interestingly, there is always a single dimension
# That converges more slowly than the rest.
@pytest.mark.skip('Difficult convergence test.')
@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_rastrigin10_converges(optimizer_type):
    net = RastriginModule(num_dimensions=10)
    optim = optimizer_type(net.parameters())

    global_minimum = torch.zeros_like(net.weights)

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not torch.allclose(net.weights, global_minimum, atol=1e-4, rtol=1e-3)

    converged = False
    for _ in range(10000):
        optim.step(closure)
        if torch.allclose(net.weights, global_minimum, atol=1e-2, rtol=1e-1):
            converged = True
            break

    assert converged, net.weights


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_himmelblau_converges(optimizer_type):
    net = HimmelblauModule()
    optim = optimizer_type(net.parameters())

    global_minima = [torch.Tensor([3., 2.]),
                     torch.Tensor([-2.805118, 3.131312]),
                     torch.Tensor([-3.779310, -3.283186]),
                     torch.Tensor([3.584428, -1.848126]),
                     ]

    def close_to_a_minimum(x, minima: List[Tensor]) -> bool:
        return any(torch.allclose(x, m, atol=1e-2, rtol=1e-1) for m in minima)

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not close_to_a_minimum(net.weights, global_minima)

    converged = False
    for _ in range(4000):
        optim.step(closure)
        if close_to_a_minimum(net.weights, global_minima):
            converged = True
            break

    assert converged, net.weights


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_goldstein_price_converges(optimizer_type):
    net = GoldsteinPriceModule()
    optim = optimizer_type(net.parameters())

    global_minima = [torch.Tensor([0, -1]),
                     ]

    def close_to_a_minimum(x, minima: List[Tensor]) -> bool:
        return any(torch.allclose(x, m, atol=1e-1, rtol=1e-1) for m in minima)

    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not close_to_a_minimum(net.weights, global_minima)

    converged = False
    for _ in range(5000):
        optim.step(closure)
        if close_to_a_minimum(net.weights, global_minima):
            converged = True
            break

    assert converged, net.weights
