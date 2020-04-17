import torch
from torch.optim.optimizer import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator


def _combined_found_inf_helper(
    optimizer: Optimizer, grad_scaler: GradScaler, device: torch.Device) -> _MultiDeviceReplicator
