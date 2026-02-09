from collections.abc import Callable

import torch
from torch._dynamo.decorators import leaf_function


def make_compile_print(
    fwd_f: Callable[[torch.Tensor], None],
    bwd_f: Callable[[torch.Tensor], None],
) -> Callable[..., None]:
    """
    Create a compile_print function with the given forward and backward callbacks.

    The returned function is side-effect-only: it calls fwd_f(tensor) on each tensor
    in the forward pass, and registers hooks so bwd_f(grad) is called on each gradient
    in the backward pass. It does not return anything.

    Example:
        >>> def print_fwd(t):
        ...     print(f"Forward: shape={t.shape}, mean={t.mean():.4f}")
        >>> def print_bwd(t):
        ...     print(f"Backward: shape={t.shape}, mean={t.mean():.4f}")
        >>> compile_print = make_compile_print(print_fwd, print_bwd)
        >>> @torch.compile
        ... def fn(x):
        ...     compile_print(x)
        ...     return x.sum()
        >>> x = torch.randn(3, 3, requires_grad=True)
        >>> fn(x).backward()

    Args:
        fwd_f: Function to call on each tensor in forward pass. Signature: (Tensor) -> None.
        bwd_f: Function to call on each gradient in backward pass. Signature: (Tensor) -> None.

    Returns:
        A side-effect-only function that takes tensors and returns None.
    """

    @leaf_function
    def fwd_leaf_fn(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                fwd_f(t)
        return (tensors[0].new_zeros(()),)

    @fwd_leaf_fn.register_fake
    def fwd_fake_fn(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (tensors[0].new_zeros(()),)

    def compile_print_impl(*args: torch.Tensor) -> None:
        fwd_leaf_fn(*args)
        # Backward hooks are registered on the original tensors. Since the tensors
        # are used downstream (e.g. x.sum()), gradients flow through them and the
        # hooks fire during backward.
        # Inside torch.compile, Dynamo traces through this function and handles
        # register_hook. Outside Dynamo, skip hooks during make_fx/aot_function
        # tracing where tensors are proxies or fakes.
        if torch.compiler.is_compiling():
            for t in args:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    t.register_hook(bwd_f)
        else:
            from torch._guards import detect_fake_mode
            from torch.fx.experimental.proxy_tensor import get_proxy_mode

            if get_proxy_mode() is None and detect_fake_mode(args) is None:
                for t in args:
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        t.register_hook(bwd_f)

    return compile_print_impl


def compile_print(
    fwd_f: Callable[[torch.Tensor], None],
    bwd_f: Callable[[torch.Tensor], None],
    *args: torch.Tensor,
) -> None:
    """
    Run fwd_f(tensor) on all tensor args in forward, and register bwd_f as a
    gradient hook on each tensor for backward.

    NOTE: This function works in eager mode. For use inside torch.compile, use
    make_compile_print() to create the function ahead of time, then call it inside
    the compiled region.

    Example (eager):
        >>> def print_fwd(t):
        ...     print(f"Forward: shape={t.shape}, mean={t.mean():.4f}")
        >>> def print_bwd(t):
        ...     print(f"Backward: shape={t.shape}, mean={t.mean():.4f}")
        >>> x = torch.randn(3, 3, requires_grad=True)
        >>> compile_print(print_fwd, print_bwd, x)
        >>> x.sum().backward()

    Example (compiled):
        >>> compile_print_fn = make_compile_print(print_fwd, print_bwd)
        >>> @torch.compile
        ... def fn(x):
        ...     compile_print_fn(x)
        ...     return x.sum()

    Args:
        fwd_f: Function to call on each tensor in forward pass. Signature: (Tensor) -> None.
        bwd_f: Function to call on each gradient in backward pass. Signature: (Tensor) -> None.
        *args: Tensors to observe.
    """
    fn = make_compile_print(fwd_f, bwd_f)
    fn(*args)
