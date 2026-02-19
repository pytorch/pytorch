from collections.abc import Callable

import torch
from torch._dynamo.decorators import leaf_function


def _dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank_enabled(ranks: int | set[int] | None) -> bool:
    """Check if the current rank should print. None means all ranks."""
    if ranks is None:
        return True
    if not _dist_initialized():
        return True
    rank = torch.distributed.get_rank()
    if isinstance(ranks, int):
        return rank == ranks
    return rank in ranks


def _make_prefix(tag: str) -> str:
    """Build a prefix like '[rank 0][my_tag]' from the current rank and tag."""
    parts = []
    if _dist_initialized():
        parts.append(f"[rank {torch.distributed.get_rank()}]")
    if tag:
        parts.append(f"[{tag}]")
    return "".join(parts)


def make_opaque_inspect_tensor_fn(
    fwd_f: Callable[[torch.Tensor], None],
    bwd_f: Callable[[torch.Tensor], None],
) -> Callable[..., None]:
    """
    Create a callable that observes tensors during forward and backward passes,
    compatible with eager mode, ``torch.compile`` with ``aot_eager`` backend,
    and ``make_fx``.

    The returned callable ``opaque_inspect`` is side-effect-only: calling
    ``opaque_inspect(t1, t2, ...)`` invokes ``fwd_f`` on each tensor in the
    forward pass and registers hooks so that ``bwd_f`` is called on each
    gradient in the backward pass. ``opaque_inspect`` returns ``None`` and does
    not modify the tensors.

    ``opaque_inspect`` must be created **outside** the compiled region and
    called **inside** it. Internally it uses
    :func:`@leaf_function <torch._dynamo.decorators.leaf_function>`
    so both the forward and backward callbacks are opaque to the compiler — the
    callbacks execute at runtime without being traced.

    Args:
        fwd_f: Called as ``fwd_f(tensor)`` on each positional tensor argument in the
            forward pass.
        bwd_f: Called as ``bwd_f(grad)`` on each gradient in the backward pass.

    Returns:
        A callable with signature
        ``opaque_inspect(*tensors, tag="", ranks=None, phase=None) -> None``.

        - **tag** (``str``): Optional label. When set, ``opaque_inspect`` prints
          ``[tag][fwd]`` / ``[tag][bwd]`` to stdout each time the callbacks run.
          In distributed settings, the rank is prepended: ``[rank 0][tag][fwd]``.
        - **ranks** (``int | set[int] | None``): Which distributed ranks should
          execute callbacks. ``None`` (default) means all ranks. An ``int`` means
          only that rank. A ``set[int]`` means those ranks.
        - **phase** (``str | None``): Override which callback and label to use.
          ``None`` (default) auto-determines: calls ``fwd_f`` with label
          ``[fwd]`` and registers backward hooks for ``bwd_f``. ``"fwd"``
          calls ``fwd_f`` with label ``[fwd]`` but does not register backward
          hooks. ``"bwd"`` calls ``bwd_f`` directly with label ``[bwd]`` and
          does not register backward hooks. This is useful inside a custom
          ``torch.autograd.Function`` where you want to control which callback
          runs in forward vs backward.

    .. note::

        ``bwd_f`` is called in both ``torch.compile`` and ``make_fx``, but
        through slightly different mechanisms:

        - With ``torch.compile``, if ``opaque_inspect`` is called on an input tensor
          (i.e., a tensor that is a direct input to the compiled function, not one computed
          inside it), the hook fires *after* the compiled backward graph
          completes, so ``bwd_f`` still executes at runtime but
          ``invoke_leaf_function`` does not appear in the backward graph. If
          ``opaque_inspect`` is called on an **intermediate tensor** (one
          computed inside the compiled function), the
          hook fires *during* the backward pass and ``invoke_leaf_function``
          appears in the backward graph.
        - With ``make_fx``, ``bwd_f`` always appears as an explicit
          ``invoke_leaf_function`` node in the backward graph regardless of
          whether it's called on a leaf or intermediate tensor.

    Example — basic usage::

        >>> import torch
        >>> from torch._higher_order_ops.opaque_inspect_tensor import make_opaque_inspect_tensor_fn
        >>> opaque_inspect = make_opaque_inspect_tensor_fn(
        ...     fwd_f=lambda t: print(f"  fwd: shape={t.shape}, norm={t.norm():.4f}"),
        ...     bwd_f=lambda t: print(f"  bwd: shape={t.shape}, norm={t.norm():.4f}"),
        ... )

        Eager mode:

        >>> x = torch.randn(3, 3, requires_grad=True)
        >>> opaque_inspect(x)                     # prints fwd info immediately
          fwd: shape=torch.Size([3, 3]), norm=...
        >>> x.sum().backward()        # prints bwd info when gradient flows
          bwd: shape=torch.Size([3, 3]), norm=...

        Inside torch.compile:

        >>> @torch.compile(backend="aot_eager")
        ... def fn(x):
        ...     opaque_inspect(x)
        ...     return x.sum()
        >>> fn(torch.randn(3, 3, requires_grad=True)).backward()
          fwd: shape=torch.Size([3, 3]), norm=...
          bwd: shape=torch.Size([3, 3]), norm=...

    Example — tagging and multiple tensors::

        >>> opaque_inspect = make_opaque_inspect_tensor_fn(
        ...     fwd_f=lambda t: print(f"    {t.shape}"),
        ...     bwd_f=lambda t: print(f"    {t.shape}"),
        ... )
        >>> x = torch.randn(2, 4, requires_grad=True)
        >>> y = torch.randn(2, 4, requires_grad=True)
        >>> opaque_inspect(x, y, tag="inputs")
        [inputs][fwd]
            torch.Size([2, 4])
            torch.Size([2, 4])
        >>> (x + y).sum().backward()
        [inputs][bwd]
            torch.Size([2, 4])
        [inputs][bwd]
            torch.Size([2, 4])

    Example — inspect inside a custom autograd function::

        >>> from torch.autograd import Function
        >>> opaque_inspect = make_opaque_inspect_tensor_fn(
        ...     fwd_f=lambda t: print(f"  {t.shape} norm={t.norm():.4f}"),
        ...     bwd_f=lambda t: print(f"  {t.shape} norm={t.norm():.4f}"),
        ... )
        >>>
        >>> class MyReLU(Function):
        ...     @staticmethod
        ...     def forward(ctx, x):
        ...         ctx.save_for_backward(x)
        ...         opaque_inspect(x, tag="my_relu", phase="fwd")    # [my_relu][fwd]
        ...         return x.clamp(min=0)
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         (x,) = ctx.saved_tensors
        ...         opaque_inspect(grad_output, tag="my_relu", phase="bwd")  # [my_relu][bwd]
        ...         return grad_output * (x > 0).float()
    """

    @leaf_function
    def fwd_leaf_fn(
        *tensors: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> None:
        if _rank_enabled(ranks):
            prefix = _make_prefix(tag)
            if prefix:
                print(f"{prefix}[fwd]")
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    fwd_f(t)

    @fwd_leaf_fn.register_fake  # pyrefly: ignore[missing-attribute]
    def fwd_fake_fn(
        *tensors: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> None:
        pass

    # Backward callback is also a leaf function so it's opaque to tracing.
    # When register_hook fires during aot_autograd's backward trace, this
    # creates an invoke_leaf_function node in the backward graph.
    @leaf_function
    def bwd_leaf_fn(
        grad: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> None:
        if _rank_enabled(ranks):
            prefix = _make_prefix(tag)
            if prefix:
                print(f"{prefix}[bwd]")
            bwd_f(grad)

    @bwd_leaf_fn.register_fake  # pyrefly: ignore[missing-attribute]
    def bwd_fake_fn(
        grad: torch.Tensor, tag: str = "", ranks: int | set[int] | None = None
    ) -> None:
        pass

    def opaque_inspect_tensor_impl(
        *args: torch.Tensor,
        tag: str = "",
        ranks: int | set[int] | None = None,
        phase: str | None = None,
    ) -> None:
        if phase is not None and phase not in ("fwd", "bwd"):
            raise ValueError(f"Invalid phase: {phase}, phase should only be fwd or bwd")
        if phase == "bwd":
            # Explicitly in backward: call bwd_f on each tensor, no hooks.
            for t in args:
                if isinstance(t, torch.Tensor):
                    bwd_leaf_fn(t, tag=tag, ranks=ranks)
        else:
            # Forward (phase is None or "fwd"):
            # call fwd_f on all tensors.
            fwd_leaf_fn(*args, tag=tag, ranks=ranks)

            # Only register backward hooks in auto mode (phase=None).
            # When phase="fwd", the caller is explicitly marking this as a
            # forward-only call (e.g. inside autograd Function.forward where
            # backward is handled separately with phase="bwd").
            if phase is None:
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

    return opaque_inspect_tensor_impl
