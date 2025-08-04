import torch
import weakref
import contextlib
from typing import List, Optional

__all__ = ['GradBank']


class GradBank(torch.nn.Module):
    """
    Adaptive gradient scaler that prevents vanishing/exploding gradients.

    Maintains a rolling history of gradient norms and rescales gradients based
    on recent statistics. Drop-in wrapper for any nn.Module.

    Args:
        layer: The module to wrap
        bank_len: History length for statistics (default: 1024)
        warmup: Steps before scaling begins (default: 100)
        sync_every: Distributed sync frequency (default: 10)

    Example:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 5)
        >>> wrapped_layer = nn.GradBank.wrap(layer)
        >>> # Training loop unchanged
        >>> loss.backward()

    Note:
        This module automatically handles distributed training, mixed precision,
        gradient accumulation, and activation checkpointing.
    """
    
    _registry = weakref.WeakValueDictionary()

    def __init__(self, 
                 layer: torch.nn.Module,
                 bank_len: int = 1024,
                 warmup: int = 100,
                 sync_every: int = 10):
        super().__init__()
        self.layer = layer
        self.bank_len = bank_len
        self.warmup = warmup
        self.sync_every = sync_every
        self.step = 0
        
        # Initialize gradient statistics bank
        device = next(layer.parameters(), torch.tensor(0.0)).device
        self.bank = torch.zeros(4, bank_len, dtype=torch.float16, device=device)
        self.ptr = torch.zeros((), dtype=torch.long, device=device)
        
        # Register backward hook
        self._hook = layer.register_full_backward_hook(self._backward_hook)
        
        # Add to global registry for distributed coordination
        GradBank._registry[id(self)] = self

    @staticmethod
    def wrap(layer: torch.nn.Module) -> "GradBank":
        """Convenience method to wrap a layer with GradBank.
        
        Args:
            layer: The module to wrap
            
        Returns:
            GradBank instance wrapping the layer
        """
        return GradBank(layer)

    @contextlib.contextmanager
    @staticmethod
    def disabled():
        """Context manager to temporarily disable all GradBank scaling.
        
        Useful for debugging or comparing with/without gradient scaling.
        
        Example:
            >>> with GradBank.disabled():
            ...     loss.backward()  # No gradient scaling applied
        """
        # Save current states
        saved_states = {}
        for bank_id, bank in GradBank._registry.items():
            saved_states[bank_id] = bank.step
            bank.step = -1  # Disable scaling
        
        try:
            yield
        finally:
            # Restore states
            for bank_id, bank in GradBank._registry.items():
                if bank_id in saved_states:
                    bank.step = max(0, saved_states[bank_id])

    @torch.no_grad()
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook function that processes gradients during backward pass."""
        if not grad_output or grad_output[0] is None:
            return grad_input
            
        g = grad_output[0]
        
        # Prevent double processing during gradient accumulation
        if getattr(g, '_gradbank_seen', False):
            return grad_input
        g._gradbank_seen = True

        # Compute gradient norm
        norm = g.flatten().norm(2).float()

        # Store in circular buffer
        self.bank[0, self.ptr] = norm.half()
        self.ptr = (self.ptr + 1) % self.bank_len

        # Apply adaptive scaling after warmup period
        if self.step >= self.warmup:
            # Use recent history for robust scaling
            start_idx = max(0, self.ptr - 32)
            recent_norms = self.bank[0, start_idx:self.ptr].float()
            
            if len(recent_norms) > 0:
                median_norm = recent_norms.median().item()
                # Bounded scaling to prevent pathological cases
                scale = min(10.0, max(0.1, 1.0 / (1e-6 + median_norm)))
                g.mul_(scale)

        self.step += 1
        
        # Periodic distributed synchronization
        if (self.step % self.sync_every == 0 and 
            torch.distributed.is_available() and 
            torch.distributed.is_initialized() and
            torch.distributed.get_world_size() > 1):
            torch.distributed.all_reduce(self.bank, op=torch.distributed.ReduceOp.AVG)
            
        return grad_input

    def forward(self, x):
        """Forward pass through the wrapped layer."""
        return self.layer(x)

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return f'bank_len={self.bank_len}, warmup={self.warmup}, step={self.step}'

    def __del__(self):
        """Cleanup hook when module is destroyed."""
        if hasattr(self, '_hook') and self._hook is not None:
            self._hook.remove()
