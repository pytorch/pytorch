#!/usr/bin/env python3

"""Model execution logic for TorchBench benchmark suite."""

import torch
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs


class TorchBenchModelRunner:
    """Handles execution of TorchBench models (forward and backward passes)."""
    
    def __init__(self, autocast, autocast_arg, grad_scaler):
        self.autocast = autocast
        self.autocast_arg = autocast_arg
        self.grad_scaler = grad_scaler
        self._optimizer_zero_grad = None
        self._optimizer_step = None
    
    def set_optimizer_methods(self, optimizer_zero_grad, optimizer_step):
        """Set optimizer methods from the main runner."""
        self._optimizer_zero_grad = optimizer_zero_grad
        self._optimizer_step = optimizer_step
    
    def pick_grad(self, name, is_training):
        """Choose appropriate gradient context manager."""
        if is_training or name in ("maml",):
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def compute_loss(self, pred):
        """Compute scalar loss from model prediction."""
        return reduce_to_scalar_loss(pred)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        """Execute forward pass through model."""
        with self.autocast(**self.autocast_arg):
            if isinstance(inputs, dict):
                return mod(**inputs)
            else:
                return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        """Execute forward and backward pass through model."""
        cloned_inputs = clone_inputs(inputs)
        self._optimizer_zero_grad(mod)
        with self.autocast(**self.autocast_arg):
            if isinstance(cloned_inputs, dict):
                pred = mod(**cloned_inputs)
            else:
                pred = mod(*cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self._optimizer_step()
        if collect_outputs:
            return collect_results(mod, None, loss, cloned_inputs)
        return None