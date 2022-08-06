# Torch PSO

Particle Swarm Optimization is an optimization technique that iteratively attempts to improve a list of candidate
solutions. Each candidate solution is called a "particle", and collectively they are called a "swarm". In each step of
the optimization, each particle moves in a random directly while simultaneously being pulled towards the other particles
in the swarm. A simple introduction to the algorithm can be found on 
[its Wikipedia article](https://en.wikipedia.org/wiki/Particle_swarm_optimization).

This package implements the Particle Swarm Optimization using the PyTorch Optimizer API, making it compatible with 
most pre-existing Torch training loops.

## Installation
To install Torch PSO using PyPI, run the following command:

    $ pip install torch-pso

## Getting Started
To use the ParticleSwarmOptimizer, simply import it, and use it as with any other PyTorch Optimizer. 
Hyperparameters of the optimizer can also be specified. In practice, most PyTorch tutorials could be used to create 
a use-case, simply substituting the ParticleSwarmOptimizer for any other optimizer. 
A simplified use-case can be seen below, which trains a simple neural network to match its output to a target.

```python
import torch
from torch.nn import Sequential, Linear, MSELoss
from torch_pso import ParticleSwarmOptimizer

net = Sequential(Linear(10,100), Linear(100,100), Linear(100,10))
optim = ParticleSwarmOptimizer(net.parameters(),
                               inertial_weight=0.5,
                               num_particles=100,
                               max_param_value=1,
                               min_param_value=-1)
criterion = MSELoss()
target = torch.rand((10,)).round()

x = torch.rand((10,))
for _ in range(100):
    
    def closure():
        # Clear any grads from before the optimization step, since we will be changing the parameters
        optim.zero_grad()  
        return criterion(net(x), target)
    
    optim.step(closure)
    print('Prediciton', net(x))
    print('Target    ', target)
```