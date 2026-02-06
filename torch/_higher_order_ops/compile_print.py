from collections.abc import Callable

import torch
from torch._dynamo.decorators import leaf_function


def make_compile_print(
    fwd_f: Callable[[torch.Tensor], None],
    bwd_f: Callable[[torch.Tensor], None],
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """
    Create a compile_print function with the given forward and backward callbacks.

    This factory function creates a properly configured compile_print that can be used
    inside torch.compile regions. The returned function must be called with tensors
    and will run fwd_f(tensor) on each tensor in forward, and bwd_f(grad) on each
    gradient in backward.

    Example:
        >>> def print_fwd(t):
        ...     print(f"Forward: shape={t.shape}, mean={t.mean():.4f}")
        >>> def print_bwd(t):
        ...     print(f"Backward: shape={t.shape}, mean={t.mean():.4f}")
        >>> compile_print = make_compile_print(print_fwd, print_bwd)
        >>> @torch.compile
        ... def fn(x):
        ...     out = compile_print(x)
        ...     return out[0].sum()
        >>> x = torch.randn(3, 3, requires_grad=True)
        >>> fn(x).backward()

    Args:
        fwd_f: Function to call on each tensor in forward pass. Signature: (Tensor) -> None.
        bwd_f: Function to call on each gradient in backward pass. Signature: (Tensor) -> None.

    Returns:
        A function that takes tensors and returns them unchanged after calling the callbacks.
    """

    @leaf_function
    def fwd_leaf_fn(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                fwd_f(t)
        return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in tensors)

    @fwd_leaf_fn.register_fake
    def fwd_fake_fn(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in tensors)

    @leaf_function
    def bwd_leaf_fn(*grad_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        for g in grad_tensors:
            if isinstance(g, torch.Tensor):
                bwd_f(g)
        return tuple(grad_tensors)

    @bwd_leaf_fn.register_fake
    def bwd_fake_fn(*grad_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(grad_tensors)

    def compile_print_impl(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        fwd_outputs = fwd_leaf_fn(*args)
        return _CompilePrintBackward.apply(bwd_leaf_fn, *fwd_outputs)

    return compile_print_impl


class _CompilePrintBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bwd_leaf_fn, *tensors):
        ctx.bwd_leaf_fn = bwd_leaf_fn
        return tensors

    @staticmethod
    def backward(ctx, *grads):
        bwd_leaf_fn = ctx.bwd_leaf_fn
        processed_grads = bwd_leaf_fn(*grads)
        return (None,) + processed_grads


def compile_print(
    fwd_f: Callable[[torch.Tensor], None],
    bwd_f: Callable[[torch.Tensor], None],
    *args: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Run fwd_f(tensor) on all tensor args in forward, and bwd_f(grad) on all gradients in backward.

    NOTE: This function works in eager mode. For use inside torch.compile, use make_compile_print()
    to create the function ahead of time, then call it inside the compiled region.

    Both fwd_f and bwd_f have signature f(Tensor) -> None.

    Example (eager):
        >>> def print_fwd(t):
        ...     print(f"Forward: shape={t.shape}, mean={t.mean():.4f}")
        >>> def print_bwd(t):
        ...     print(f"Backward: shape={t.shape}, mean={t.mean():.4f}")
        >>> x = torch.randn(3, 3, requires_grad=True)
        >>> y = compile_print(print_fwd, print_bwd, x)
        >>> y[0].sum().backward()

    Example (compiled):
        >>> compile_print_fn = make_compile_print(print_fwd, print_bwd)
        >>> @torch.compile
        ... def fn(x):
        ...     out = compile_print_fn(x)
        ...     return out[0].sum()

    Args:
        fwd_f: Function to call on each tensor in forward pass. Signature: (Tensor) -> None.
        bwd_f: Function to call on each gradient in backward pass. Signature: (Tensor) -> None.
        *args: Tensors to pass through.

    Returns:
        Tuple of the input tensors, unchanged.
    """
    fn = make_compile_print(fwd_f, bwd_f)
    return fn(*args)
