from .optim.ParticleSwarmOptimizer import ParticleSwarmOptimizer
from .optim.GenerationalPSO import GenerationalPSO
from .optim.AutotuningPSO import AutotuningPSO
from .optim.RingTopologyPSO import RingTopologyPSO
from .optim.ChaoticPSO import ChaoticPSO
from .optim.GenericPSO import GenericPSO

OPTIMIZERS = list(GenericPSO.subclasses)