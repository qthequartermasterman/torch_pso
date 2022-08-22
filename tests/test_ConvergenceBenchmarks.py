import pytest
import torch
from torch import Tensor

from tests.test_utils import optimizer_tests, close_to_a_minimum


def generic_convergence_test(optimizer_type, net: torch.nn.Module, atol: float, rtol: float, max_iterations: int):
    optim = optimizer_type(net.parameters())

    global_minima = net.global_minima

    @torch.no_grad()
    def closure():
        optim.zero_grad()
        return net(None)

    # Make sure we don't start out satisfying the condition
    assert not close_to_a_minimum(net.weights, global_minima, atol / 100, rtol / 100), \
        f'Convergence test started too close: {optimizer_type}, {net}'

    converged = False
    for _ in range(max_iterations):
        optim.step(closure)
        if close_to_a_minimum(net.weights, global_minima, atol, rtol):
            converged = True
            break

    assert converged, 'Test benchmark did not converge.' \
                      f'\nNet={net}' \
                      f'\nWeights={net.weights}' \
                      f'\nOptimizer={optimizer_type}'


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_square_converges(optimizer_type):
    class SquareWeightsModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.rand((1,)))
            self.global_minima = [torch.Tensor([0.])]

        def forward(self, x):
            x = self.weights
            return x ** 2

    return generic_convergence_test(optimizer_type=optimizer_type,
                                    net=SquareWeightsModule(),
                                    atol=1e-1,
                                    rtol=1e-1,
                                    max_iterations=3000)


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_square_plus_2_converges(optimizer_type):
    class SquarePlus2WeightsModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.rand((1,)))
            self.global_minima = [torch.Tensor([-2.])]

        def forward(self, x):
            x = self.weights
            return (x + 2) ** 2

    return generic_convergence_test(optimizer_type=optimizer_type,
                                    net=SquarePlus2WeightsModule(),
                                    atol=1e-1,
                                    rtol=1e-1,
                                    max_iterations=3000)


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_quartic_converges(optimizer_type):
    class QuarticWeightsModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.rand((1,)))
            self.global_minima = [torch.sqrt(Tensor([2])) / 2,
                                  -torch.sqrt(Tensor([2])) / 2,
                                  ]

        def forward(self, x):
            x = self.weights
            return x ** 2 * (x ** 2 - 1)

    return generic_convergence_test(optimizer_type=optimizer_type,
                                    net=QuarticWeightsModule(),
                                    atol=1e-1,
                                    rtol=1e-1,
                                    max_iterations=3000)


class RastriginModule(torch.nn.Module):
    def __init__(self, num_dimensions, a=10.):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((num_dimensions,)))
        self.num_dimensions = num_dimensions
        self.A = a
        # Rastrigin has a global minimum at the origin, regardless of dimension
        self.global_minima = [torch.zeros_like(self.weights)]

    def forward(self, x):
        x = self.weights
        return self.A * self.num_dimensions + (x ** 2 - self.A * torch.cos(2 * torch.pi * x)).sum()


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_rastrigin1_converges(optimizer_type):
    return generic_convergence_test(optimizer_type=optimizer_type,
                                    net=RastriginModule(num_dimensions=1),
                                    atol=1e-1,
                                    rtol=1e-1,
                                    max_iterations=3000)


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO', 'GenerationalPSO', 'ChaoticPSO', 'AcceleratedPSO'])
def test_rastrigin3_converges(optimizer_type):
    return generic_convergence_test(optimizer_type=optimizer_type,
                                    net=RastriginModule(num_dimensions=3),
                                    atol=1e-1,
                                    rtol=1e-1,
                                    max_iterations=3000)


# All of the algorithms converge really slowly on this test. Interestingly, there is always a single dimension
# That converges more slowly than the rest.
@pytest.mark.skip('Difficult convergence test.')
@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO', 'GenerationalPSO', 'ChaoticPSO', 'AcceleratedPSO'])
def test_rastrigin10_converges(optimizer_type):
    return generic_convergence_test(optimizer_type=optimizer_type,
                                    net=RastriginModule(num_dimensions=10),
                                    atol=1e-1,
                                    rtol=1e-1,
                                    max_iterations=3000)


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_himmelblau_converges(optimizer_type):
    class HimmelblauModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.rand((2,)))
            self.global_minima = [torch.Tensor([3., 2.]),
                                  torch.Tensor([-2.805118, 3.131312]),
                                  torch.Tensor([-3.779310, -3.283186]),
                                  torch.Tensor([3.584428, -1.848126]),
                                  ]

        def forward(self, x):
            x, y = self.weights
            return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    return generic_convergence_test(optimizer_type=optimizer_type,
                                    net=HimmelblauModule(),
                                    atol=1e-1,
                                    rtol=1e-1,
                                    max_iterations=3000)


@optimizer_tests(ignore=['DolphinPodOptimizer', 'RingTopologyPSO'])
def test_goldstein_price_converges(optimizer_type):
    class GoldsteinPriceModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.rand((2,)))
            self.global_minima = [torch.Tensor([0, -1])]

        def forward(self, x):
            x, y = self.weights
            return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

    return generic_convergence_test(optimizer_type=optimizer_type,
                                    net=GoldsteinPriceModule(),
                                    atol=1e-1,
                                    rtol=1e-1,
                                    max_iterations=5000)
