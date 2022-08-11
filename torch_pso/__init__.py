from .optim.ParticleSwarmOptimizer import ParticleSwarmOptimizer
from .optim.GenerationalPSO import GenerationalPSO
from .optim.AutotuningPSO import AutotuningPSO
from .optim.RingTopologyPSO import RingTopologyPSO
from .optim.ChaoticPSO import ChaoticPSO
from .optim.GenericPSO import GenericPSO
from .optim.AcceleratedPSO import AcceleratedPSO
from .optim.SineCosineAlgorithm import SineCosineAlgorithm

OPTIMIZERS = list(GenericPSO.subclasses)