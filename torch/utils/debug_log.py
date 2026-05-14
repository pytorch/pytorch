"""Compile-safe backward gradient logging for multiple tensors.

``debug_grad_log`` logs gradient norms during backward for one or more tensors.
It is a leaf function with a ``register_multi_grad_hook`` that fires exactly
once when all requires_grad tensor inputs have their gradients computed.

Example::

    import torch
    from torch.utils.debug_log import debug_grad_log

    x = torch.randn(4, requires_grad=True)
    y = torch.randn(4, requires_grad=True)
    z = x * 2 + y * 3

    debug_grad_log(x, y)

    z.sum().backward()
    # Logs: [rank 0][bwd] t0_grad_norm=... t1_grad_norm=...
"""

import logging

import torch
from torch._dynamo.decorators import leaf_function


__all__ = ["debug_grad_log"]

log = logging.getLogger(__name__)


def _get_rank() -> int:
    if not torch.distributed.is_available():
        return 0
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


@leaf_function
def debug_grad_log(*tensors):
    """Log gradient norms of multiple tensors during backward.

    This is a no-op in the forward pass. During backward, the hook fires
    exactly once when all requires_grad tensor inputs have their gradients
    computed, and logs ``[rank R][bwd] t0_grad_norm=... t1_grad_norm=...``.

    Args:
        *tensors: One or more tensors to monitor.

    Returns:
        None. Call without assignment: ``debug_grad_log(x, y)``.
    """
    return None


@debug_grad_log.register_fake  # pyrefly: ignore[missing-attribute]
def _debug_grad_log_fake(*tensors):
    return None


@debug_grad_log.register_multi_grad_hook  # pyrefly: ignore[missing-attribute]
def _debug_grad_log_hook(*grads):
    norms = " ".join(
        f"t{i}_grad_norm={g.norm().item():.4f}" for i, g in enumerate(grads)
    )
    log.info("[rank %d][bwd] %s", _get_rank(), norms)
