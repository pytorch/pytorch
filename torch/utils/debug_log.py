"""Compile-safe backward gradient logging for multiple tensors.

``debug_grad_log`` logs gradient norms during backward for one or more tensors.
It is a leaf function with a ``register_multi_grad_hook`` that fires exactly
once when all requires_grad tensor inputs have their gradients computed.

For **forward** logging, use ``torch._higher_order_ops.print`` directly —
it already supports format strings, DTensor, and Inductor codegen.

Works with eager, torch.compile's aot_eager backend, and make_fx tracing.

Example::

    import torch
    import torch._higher_order_ops
    from torch.utils.debug_log import debug_grad_log

    x = torch.randn(4, requires_grad=True)
    y = torch.randn(4, requires_grad=True)
    z = x * 2 + y * 3

    # Forward logging — use HOP print directly
    torch._higher_order_ops.print("fwd: x_norm={} y_norm={}", x.norm(), y.norm())

    # Backward gradient logging — fires once when all grads are ready
    debug_grad_log("layer1", x, y)

    z.sum().backward()
    # Logs: [rank 0][layer1][bwd] t0_grad_norm=... t1_grad_norm=...
"""

import logging

import torch.distributed as dist
from torch._dynamo.decorators import leaf_function


__all__ = ["debug_grad_log"]

log = logging.getLogger(__name__)


def _get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _should_log(ranks: int | list[int] | None) -> bool:
    if ranks is None:
        return True
    if not dist.is_initialized():
        return True
    current_rank = dist.get_rank()
    if isinstance(ranks, int):
        return current_rank == ranks
    return current_rank in ranks


@leaf_function
def debug_grad_log(tag, *tensors, ranks=None):
    """Log gradient norms of multiple tensors during backward.

    This is a no-op in the forward pass. During backward, the hook fires
    exactly once when all requires_grad tensor inputs have their gradients
    computed, and logs ``[rank R][tag][bwd] t0_grad_norm=... t1_grad_norm=...``.

    For forward logging, use ``torch._higher_order_ops.print`` directly.

    Supported backends: eager, ``aot_eager``, and ``torch.compile`` with
    ``aot_eager``. Inductor backend is not yet supported.

    Args:
        tag: String label included in the log output.
        *tensors: One or more tensors to monitor. Each must require grad and
            be on the path from input to loss for its gradient to be available.
        ranks: Which distributed ranks should log. ``None`` logs on all
            ranks, an ``int`` logs on that rank only, and a collection of
            ints logs on those ranks. When ``torch.distributed`` is not
            initialized, logging always occurs.

    Returns:
        None. Call without assignment: ``debug_grad_log("fc1", x, y)``.

    Example::

        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.0], requires_grad=True)
        >>> y = torch.tensor([1.0], requires_grad=True)
        >>> z = x * 2 + y * 3
        >>> debug_grad_log("check", x, y)
        >>> z.sum().backward()
        # Logs: [rank 0][check][bwd] t0_grad_norm=2.0000 t1_grad_norm=3.0000
    """
    return None


@debug_grad_log.register_fake  # pyrefly: ignore[missing-attribute]
def _debug_grad_log_fake(tag, *tensors, ranks=None):
    return None


@debug_grad_log.register_multi_grad_hook  # pyrefly: ignore[missing-attribute]
def _debug_grad_log_hook(tag, *grads, ranks=None):
    if _should_log(ranks):
        norms = " ".join(
            f"t{i}_grad_norm={g.norm().item():.4f}" for i, g in enumerate(grads)
        )
        log.info("[rank %d][%s][bwd] %s", _get_rank(), tag, norms)
