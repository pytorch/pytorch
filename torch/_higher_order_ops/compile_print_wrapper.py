from collections.abc import Callable

import torch
from torch._dynamo.decorators import leaf_function


def _rank_enabled(ranks: int | set[int] | None) -> bool:
    """Check if the current rank should print. None means all ranks."""
    if ranks is None:
        return True
    if not torch.distributed.is_initialized():
        return True
    rank = torch.distributed.get_rank()
    if isinstance(ranks, int):
        return rank == ranks
    return rank in ranks


def _make_prefix(tag: str) -> str:
    """Build a prefix like '[rank 0][my_tag]' from the current rank and tag."""
    parts = []
    if torch.distributed.is_initialized():
        parts.append(f"[rank {torch.distributed.get_rank()}]")
    if tag:
        parts.append(f"[{tag}]")
    return "".join(parts)


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

    Note:
        Backward function bwd_f is called in both torch.compile and aot_function, but
        through different mechanisms. With aot_function, bwd_f appears as an
        explicit invoke_leaf_function graph node in the backward graph. With
        torch.compile, bwd_f runs as a runtime hook on the tensor.
    """

    @leaf_function
    def fwd_leaf_fn(
        *tensors: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> tuple[torch.Tensor, ...]:
        if _rank_enabled(ranks):
            prefix = _make_prefix(tag)
            if prefix:
                print(f"{prefix}[fwd]")
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    fwd_f(t)
        return (tensors[0].new_zeros(()),)

    @fwd_leaf_fn.register_fake  # pyrefly: ignore[missing-attribute]
    def fwd_fake_fn(
        *tensors: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> tuple[torch.Tensor, ...]:
        return (tensors[0].new_zeros(()),)

    # Backward callback is also a leaf function so it's opaque to tracing.
    # When register_hook fires during aot_autograd's backward trace, this
    # creates an invoke_leaf_function node in the backward graph.
    @leaf_function
    def bwd_leaf_fn(
        grad: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> tuple[torch.Tensor, ...]:
        if _rank_enabled(ranks):
            prefix = _make_prefix(tag)
            if prefix:
                print(f"{prefix}[bwd]")
            bwd_f(grad)
        return (grad.new_zeros(()),)

    @bwd_leaf_fn.register_fake  # pyrefly: ignore[missing-attribute]
    def bwd_fake_fn(
        grad: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> tuple[torch.Tensor, ...]:
        return (grad.new_zeros(()),)

    def compile_print_impl(
        *args: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> None:
        fwd_leaf_fn(*args, tag=tag, ranks=ranks)

        for t in args:
            if isinstance(t, torch.Tensor) and t.requires_grad:
                # The hook calls bwd_leaf_fn (a leaf function). During aot tracing,
                # register_hook fires in the backward trace, and bwd_leaf_fn creates
                # an invoke_leaf_function node in the backward graph. At runtime,
                # that node executes the real bwd_leaf_fn.
                def _make_hook(_tag, _ranks):
                    def hook(grad):
                        bwd_leaf_fn(grad, tag=_tag, ranks=_ranks)

                    return hook

                t.register_hook(_make_hook(tag, ranks))

    return compile_print_impl


def compile_print(
    fwd_f: Callable[[torch.Tensor], None],
    bwd_f: Callable[[torch.Tensor], None],
    *args: torch.Tensor,
    tag: str = "",
    ranks: int | set[int] | None = None,
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
        tag: Optional tag string. When set, prints [tag][fwd] and [tag][bwd] labels.
        ranks: Which distributed ranks should execute callbacks. None (default) means
            all ranks. An int means only that rank. A set of ints means those ranks.
    """
    fn = make_compile_print(fwd_f, bwd_f)
    fn(*args, tag=tag, ranks=ranks)
