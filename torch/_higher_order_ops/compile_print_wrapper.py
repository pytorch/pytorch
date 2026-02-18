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
    Create a callable that observes tensors during forward and backward passes,
    compatible with eager mode, ``torch.compile``, and ``aot_function``.

    The returned callable ``cp`` is side-effect-only: calling ``cp(t1, t2, ...)``
    invokes ``fwd_f`` on each tensor in the forward pass and registers hooks so
    that ``bwd_f`` is called on each gradient in the backward pass. ``cp``
    returns ``None`` and does not modify the tensors.

    ``cp`` must be created **outside** the compiled region and called **inside**
    it. Internally it uses :func:`@leaf_function <torch._dynamo.decorators.leaf_function>`
    so both the forward and backward callbacks are opaque to the compiler — the
    callbacks execute at runtime without being traced.

    Args:
        fwd_f: Called as ``fwd_f(tensor)`` on each positional tensor argument in the
            forward pass.
        bwd_f: Called as ``bwd_f(grad)`` on each gradient in the backward pass.

    Returns:
        A callable with signature ``cp(*tensors, tag="", ranks=None) -> None``.

        - **tag** (``str``): Optional label. When set, ``cp`` prints
          ``[tag][fwd]`` / ``[tag][bwd]`` to stdout each time the callbacks run.
          In distributed settings, the rank is prepended: ``[rank 0][tag][fwd]``.
        - **ranks** (``int | set[int] | None``): Which distributed ranks should
          execute callbacks. ``None`` (default) means all ranks. An ``int`` means
          only that rank. A ``set[int]`` means those ranks.

    .. note::

        ``bwd_f`` is called in both ``torch.compile`` and ``aot_function``, but
        through different mechanisms:

        - With ``torch.compile``, ``bwd_f`` runs as an autograd hook at runtime.
          If ``cp`` is called on a **leaf input**, the hook fires *after* the
          compiled backward graph (so ``invoke_leaf_function`` does not appear in
          the backward graph). If ``cp`` is called on an **intermediate tensor**
          (e.g., a module output), the hook fires *during* the backward pass and
          ``invoke_leaf_function`` appears in the backward graph.
        - With ``aot_function``, ``bwd_f`` always appears as an explicit
          ``invoke_leaf_function`` node in the backward graph regardless of
          whether it's called on a leaf or intermediate tensor.

    Example — basic usage::

        >>> import torch
        >>> from torch._higher_order_ops.compile_print_wrapper import make_compile_print
        >>> cp = make_compile_print(
        ...     fwd_f=lambda t: print(f"  fwd: shape={t.shape}, norm={t.norm():.4f}"),
        ...     bwd_f=lambda t: print(f"  bwd: shape={t.shape}, norm={t.norm():.4f}"),
        ... )

        Eager mode:

        >>> x = torch.randn(3, 3, requires_grad=True)
        >>> cp(x)                     # prints fwd info immediately
          fwd: shape=torch.Size([3, 3]), norm=...
        >>> x.sum().backward()        # prints bwd info when gradient flows
          bwd: shape=torch.Size([3, 3]), norm=...

        Inside torch.compile:

        >>> @torch.compile(backend="aot_eager")
        ... def fn(x):
        ...     cp(x)
        ...     return x.sum()
        >>> fn(torch.randn(3, 3, requires_grad=True)).backward()
          fwd: shape=torch.Size([3, 3]), norm=...
          bwd: shape=torch.Size([3, 3]), norm=...

    Example — tagging and multiple tensors::

        >>> cp = make_compile_print(
        ...     fwd_f=lambda t: print(f"    {t.shape}"),
        ...     bwd_f=lambda t: print(f"    {t.shape}"),
        ... )
        >>> x = torch.randn(2, 4, requires_grad=True)
        >>> y = torch.randn(2, 4, requires_grad=True)
        >>> cp(x, y, tag="inputs")
        [inputs][fwd]
            torch.Size([2, 4])
            torch.Size([2, 4])
        >>> (x + y).sum().backward()
        [inputs][bwd]
            torch.Size([2, 4])
        [inputs][bwd]
            torch.Size([2, 4])

    Example — instrument all modules in a model::

        >>> import torch.nn as nn
        >>> import torch.utils._pytree as pytree
        >>>
        >>> def install_debug_prints(model: nn.Module) -> None:
        ...     for name, module in model.named_modules():
        ...         tag = f"{module.__class__.__name__}:{name}"
        ...         cp = make_compile_print(
        ...             fwd_f=lambda t: print(f"  {t.shape} mean={t.mean():.4f}"),
        ...             bwd_f=lambda t: print(f"  {t.shape} mean={t.mean():.4f}"),
        ...         )
        ...         orig_forward = module.forward
        ...         def wrapped(*args, _orig=orig_forward, _cp=cp, _tag=tag, **kwargs):
        ...             out = _orig(*args, **kwargs)
        ...             pytree.tree_map_only(torch.Tensor, lambda t: _cp(t, tag=_tag), out)
        ...             return out
        ...         module.forward = wrapped
        >>>
        >>> model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        >>> install_debug_prints(model)
        >>> compiled_model = torch.compile(model, backend="aot_eager")
        >>> out = compiled_model(torch.randn(2, 8, requires_grad=True))
        [Linear:0][fwd]
          torch.Size([2, 16]) mean=...
        [ReLU:1][fwd]
          torch.Size([2, 16]) mean=...
        [Linear:2][fwd]
          torch.Size([2, 4]) mean=...
        >>> out.sum().backward()
        [Linear:2][bwd]
          torch.Size([2, 4]) mean=...
        [ReLU:1][bwd]
          torch.Size([2, 16]) mean=...
        [Linear:0][bwd]
          torch.Size([2, 16]) mean=...
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
