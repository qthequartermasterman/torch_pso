from unittest import TestCase

import torch

from torch_pso import ParticleSwarmOptimizer


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
        return x**2 * (x**2-1)


class TestParticleSwarmOptimizer(TestCase):
    def test_square_converges(self):
        self.net = SquareWeightsModule()
        self.optim = ParticleSwarmOptimizer(self.net.parameters(),
                                            inertial_weight=5e-2,
                                            cognitive_coefficient=1e-2,
                                            social_coefficient=1e-2,
                                            num_particles=100,
                                            max_param_value=1,
                                            min_param_value=-1)
        def closure():
            self.optim.zero_grad()
            return self.net(None)

        for _ in range(1000):
            self.optim.step(closure)

        assert torch.allclose(self.net.weights, torch.Tensor([0.]), atol=1e-4, rtol=1e-3), self.net.weights

    def test_quartic_converges(self):
        self.net = QuarticWeightsModule()
        self.optim = ParticleSwarmOptimizer(self.net.parameters(),
                                            inertial_weight=5e-2,
                                            cognitive_coefficient=1e-2,
                                            social_coefficient=1e-2,
                                            num_particles=100,
                                            max_param_value=1,
                                            min_param_value=-1)
        def closure():
            self.optim.zero_grad()
            return self.net(None)

        for _ in range(1000):
            self.optim.step(closure)

        assert torch.allclose(abs(self.net.weights), torch.sqrt(torch.Tensor([2]))/2,
                              atol=1e-4, rtol=1e-3), self.net.weights
