# mypy: allow-untyped-defs
import functools
from collections.abc import Callable
from typing import Any

from torch._higher_order_ops.base_hop import (
    BaseHOP,
    BaseHOPFunction,
    FunctionWithNoFreeVars,
)


class FuseOrErr(BaseHOP):
    """
    HOP wrapping a user closure that, under torch.compile/Inductor, must compile
    into a single fused kernel. If the region does not collapse into one kernel,
    Inductor raises with the exact reason (a warning is emitted in fbcode
    instead). See ``fuse_or_err`` for the user-facing API.

    The ``_enforce_fusion`` kwarg gates the Inductor check: it is True for the
    forward region and equal to the user's ``fuse_backward`` for the backward
    region (see FuseOrErrFunction).
    """

    def __init__(self) -> None:
        super().__init__("fuse_or_err")

    def __call__(self, subgraph, *operands, fuse_backward=False, _enforce_fusion=True):  # type: ignore[override]
        if not isinstance(subgraph, FunctionWithNoFreeVars):
            subgraph = FunctionWithNoFreeVars(subgraph)
        return super().__call__(
            subgraph,
            *operands,
            fuse_backward=fuse_backward,
            _enforce_fusion=_enforce_fusion,
        )

    def _call_Autograd(self, subgraph, *operands, **kwargs):
        return FuseOrErrFunction.apply(self, subgraph, kwargs, *operands)


class FuseOrErrFunction(BaseHOPFunction):
    @staticmethod
    def backward(ctx, *grad_outputs):
        # The backward region is only required to fuse when the user asked for
        # it via fuse_backward; forward is always enforced.
        ctx.kwargs = {
            **ctx.kwargs,
            "_enforce_fusion": ctx.kwargs.get("fuse_backward", False),
        }
        return BaseHOPFunction.backward(ctx, *grad_outputs)


_fuse_or_err = FuseOrErr()


def fuse_or_err(fn: Callable | None = None, *, fuse_backward: bool = False) -> Callable:
    """
    Require that a closure compiles into a single fused Inductor kernel.

    Usage (wrapper or decorator)::

        out = fuse_or_err(lambda x: x.sin().cos())(inp)


        @fuse_or_err
        def region(x):
            return x.sin().cos()


        @fuse_or_err(fuse_backward=True)
        def region(x):
            return x.sin().cos()

    Under ``torch.compile`` the wrapped region is inlined into the graph and, if
    Inductor does not fuse all of its ops into one kernel, compilation raises
    with the reason the region split (a warning is emitted in fbcode). Outside
    ``torch.compile`` the closure runs eagerly with no fusion check.

    Args:
        fn: the closure to wrap. When omitted, ``fuse_or_err`` returns a
            decorator (so ``fuse_or_err(fuse_backward=...)`` can be used).
        fuse_backward: also require the backward region to fuse into one kernel.
            Defaults to False (backward is not checked).
    """

    def wrap(f: Callable) -> Callable:
        @functools.wraps(f)
        def inner(*operands: Any):
            return _fuse_or_err(f, *operands, fuse_backward=fuse_backward)

        return inner

    return wrap if fn is None else wrap(fn)
