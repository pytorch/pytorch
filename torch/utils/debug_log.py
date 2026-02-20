"""Example logging utilities using leaf_function's register_hook.

``debug_log`` and ``debug_log_rank`` log tensor norms during forward and
gradients during backward. Both are leaf functions, so they are opaque to
the compiler and work with eager, torch.compile, and aot_function.

Backward logging is implemented via ``register_hook`` on the leaf function,
which registers autograd hooks on the input tensors so the hook fires when
their gradients are computed.

Works with eager, torch.compile's aot_eager backend, and make_fx tracing.

Writing custom logging with the same pattern
=============================================

To create your own compile-safe logging function, follow this three-step
pattern:

1. Define the forward function with ``@leaf_function``. It receives the
   tensor and any extra arguments, performs the logging, and returns ``None``.
2. Register a fake implementation with ``@fn.register_fake`` that returns
   ``None`` (no tensor output to trace).
3. Register a backward hook with ``@fn.register_hook``. The hook has the
   same signature, but each tensor argument receives the gradient instead
   of the original value. The hook must return ``None``.

Example::

    from torch._dynamo.decorators import leaf_function


    @leaf_function
    def my_log(t, label):
        print(f"[{label}][fwd] mean={t.mean().item():.4f}")
        return None


    @my_log.register_fake
    def my_log_fake(t, label):
        return None


    @my_log.register_hook
    def my_log_hook(t_grad, label):
        print(f"[{label}][bwd] mean={t_grad.mean().item():.4f}")


    # Usage (works in eager, torch.compile, and aot_function):
    y = x * 2
    my_log(y, "after_linear")  # logs fwd on call, bwd when y's grad is computed

The tensor passed to the logging function must have its gradient computed
during backward for the hook to fire. This means it must be on the path
from the input to the loss.
"""

import logging

import torch
import torch._higher_order_ops
import torch.distributed as dist
from torch._dynamo.decorators import leaf_function


log = logging.getLogger(__name__)


@leaf_function
def debug_log(t, tag):
    """Log tensor norm to stdout during forward and backward.

    Prints ``[tag][fwd] norm=...`` on the forward call and
    ``[tag][bwd] norm=...`` when the tensor's gradient is computed.

    Args:
        t: Tensor to inspect. Must be on the path from input to loss
            for the backward hook to fire.
        tag: String label included in the log output.

    Returns:
        None. Call without assignment: ``debug_log(y, "my_tag")``.

    Example::

        >>> x = torch.randn(4, requires_grad=True)
        >>> y = x * 2
        >>> debug_log(y, "after_mul")
        [after_mul][fwd] norm=...
        >>> y.sum().backward()
        [after_mul][bwd] norm=...
    """
    torch._higher_order_ops.print("[{}][fwd] norm={}", tag, t.norm())
    return None


@debug_log.register_fake  # pyrefly: ignore[missing-attribute]
def debug_log_fake(t, tag):
    return None


@debug_log.register_hook  # pyrefly: ignore[missing-attribute]
def debug_log_hook(t_grad, tag):
    torch._higher_order_ops.print("[{}][bwd] norm={}", tag, t_grad.norm())


def _should_log(ranks):
    if ranks is None:
        return True
    if not dist.is_initialized():
        return True
    current_rank = dist.get_rank()
    if isinstance(ranks, int):
        return current_rank == ranks
    return current_rank in ranks


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


@leaf_function
def debug_log_rank(t, tag, ranks=None):
    """Log tensor norm via Python logging with optional rank filtering.

    Logs ``[rank R][tag][fwd] norm=...`` on the forward call and
    ``[rank R][tag][bwd] norm=...`` when the tensor's gradient is computed.
    Uses the ``torch.utils.debug_log`` logger at INFO level.

    Args:
        t: Tensor to inspect. Must be on the path from input to loss
            for the backward hook to fire.
        tag: String label included in the log output.
        ranks: Which distributed ranks should log. ``None`` logs on all
            ranks, an ``int`` logs on that rank only, and a collection of
            ints logs on those ranks. When ``torch.distributed`` is not
            initialized, logging always occurs.

    Returns:
        None. Call without assignment: ``debug_log_rank(y, "fc1", ranks=0)``.

    Example::

        >>> x = torch.randn(4, requires_grad=True)
        >>> y = x * 2
        >>> debug_log_rank(y, "fc1", ranks=0)  # logs only on rank 0
        >>> y.sum().backward()
    """
    if _should_log(ranks):
        log.info("[rank %d][%s][fwd] norm=%s", _get_rank(), tag, t.norm().item())
    return None


@debug_log_rank.register_fake  # pyrefly: ignore[missing-attribute]
def debug_log_rank_fake(t, tag, ranks=None):
    return None


@debug_log_rank.register_hook  # pyrefly: ignore[missing-attribute]
def debug_log_rank_hook(t_grad, tag, ranks=None):
    if _should_log(ranks):
        log.info("[rank %d][%s][bwd] norm=%s", _get_rank(), tag, t_grad.norm().item())
