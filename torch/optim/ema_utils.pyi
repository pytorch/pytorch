import torch
from .optimizer import Optimizer

class EMAOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer: Optimizer, device: torch.device, decay: float=...) -> None:...