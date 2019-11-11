"""
:mod:`torch.distributed.optim` exposes DistributedOptimizer, which takes a list
of remote parameters and runs the optimizer locally on the workers where the
parameters live.
"""
from .optimizer import DistributedOptimizer
