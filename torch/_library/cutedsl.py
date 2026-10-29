import inspect
import sys
from collections.abc import Callable, Iterable
from typing import Any, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from torch._higher_order_ops.cutedsl_kernel_wrap import (
        TraceableCuteDSLKernelWrapper,
    )


def _jit_executor_name(fn: Callable[..., Any]) -> str | None:
    # @cute.jit and @cute.kernel wrappers close over the name of the DSL entry point
    # they call: "_func" for host functions, "_kernel_helper" for device kernels.
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        if name == "executor_name":
            try:
                return cell.cell_contents
            except ValueError:
                return None
    return None


def is_cutedsl_jit_function(obj: object) -> bool:
    """Whether ``obj`` is a ``@cute.jit`` host function."""
    if not inspect.isfunction(obj):
        return False
    dsl_cls = obj.__dict__.get("_dsl_cls")
    if not isinstance(dsl_cls, type):
        return False
    try:
        from cutlass.cutlass_dsl.cutlass import CuTeDSL
    except ImportError:
        return False
    return issubclass(dsl_cls, CuTeDSL) and _jit_executor_name(obj) != "_kernel_helper"


def is_traceable_cutedsl_wrapper(obj: object) -> bool:
    """Whether ``obj`` is what wrap_cutedsl returns, or an instance of a subclass that
    keeps its ``__call__`` (Dynamo traces one that overrides it as any object),
    without importing the HOP module (an instance implies it is imported)."""
    module = sys.modules.get("torch._higher_order_ops.cutedsl_kernel_wrap")
    if module is None:
        return False
    cls = module.TraceableCuteDSLKernelWrapper
    return isinstance(obj, cls) and type(obj).__call__ is cls.__call__


def _is_dsl_function(obj: object, module: str, qualname: str) -> bool:
    # The DSL's own function, not a replacement stored under its name: a mock, or a
    # wrapper made with functools.wraps (which copies __module__ and __qualname__).
    return (
        inspect.isfunction(obj)
        and obj.__module__ == module
        and obj.__qualname__ == qualname
        and not hasattr(obj, "__wrapped__")
    )


def is_cutedsl_from_dlpack(obj: object) -> bool:
    """Whether ``obj`` is ``cutlass.cute.runtime.from_dlpack``, which makes the cute
    tensors of a raw host-function call."""
    return _is_dsl_function(obj, "cutlass.cute.runtime", "from_dlpack")


def is_cutedsl_current_stream(obj: object) -> bool:
    """Whether ``obj`` is ``cutlass.torch.current_stream``, which makes the stream
    argument of a raw host-function call."""
    return _is_dsl_function(obj, "cutlass.torch", "current_stream")


def wrap_cutedsl(
    host_fn: Callable[..., Any],
    /,
    *,
    mutates_args: Iterable[str] | Literal["unknown"],
) -> "TraceableCuteDSLKernelWrapper":
    """Allows capture of a CuTeDSL kernel launch into a graph via make_fx or
    non-strict ``torch.export``: the counterpart of :func:`torch.library.wrap_triton`
    for ``@cute.jit`` host functions. These trace through ``__torch_dispatch__`` and
    cannot see a raw call of a host function. Dynamo (``torch.compile``, strict
    ``torch.export``) captures a call of the wrapper as well as a raw call written as
    for eager, with ``from_dlpack(t)`` tensors and ``cutlass.torch.current_stream()``.

    ``host_fn`` must have exactly one parameter annotated ``cuda.CUstream``; callers
    of the returned wrapper do not pass it, because the kernel is launched on the
    current CUDA stream. Tensor arguments are ``torch.Tensor`` (they are converted
    with ``cutlass.cute.runtime.from_dlpack``). ``mutates_args`` names the tensor
    parameters the kernel writes, or is ``"unknown"`` to treat every tensor argument
    as written; PT2 uses it to functionalize the launch.

    Examples:

        >>> # xdoctest: +SKIP
        >>> import cuda.bindings.driver as cuda
        >>> import cutlass
        >>> import cutlass.cute as cute
        >>> from torch._library.cutedsl import wrap_cutedsl
        >>>
        >>> @cute.kernel
        >>> def add_device(gX: cute.Tensor, gY: cute.Tensor, gOut: cute.Tensor,
        >>>                n: cutlass.Int32):
        >>>     tidx, _, _ = cute.arch.thread_idx()
        >>>     bidx, _, _ = cute.arch.block_idx()
        >>>     i = bidx * 128 + tidx
        >>>     if i < n:
        >>>         gOut[i] = gX[i] + gY[i]
        >>>
        >>> @cute.jit
        >>> def add_kernel(mX: cute.Tensor, mY: cute.Tensor, mOut: cute.Tensor,
        >>>                n: cutlass.Int32, stream: cuda.CUstream):
        >>>     add_device(mX, mY, mOut, n).launch(
        >>>         grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)
        >>>
        >>> add = wrap_cutedsl(add_kernel, mutates_args=["mOut"])
        >>>
        >>> class M(torch.nn.Module):
        >>>     def forward(self, x, y):
        >>>         out = torch.empty_like(x)
        >>>         add(x, y, out, x.numel())
        >>>         return out.relu()
        >>>
        >>> x = torch.randn(1000, device="cuda")
        >>> ep = torch.export.export(M(), (x, torch.randn_like(x)), strict=False)
    """
    from torch._higher_order_ops.cutedsl_kernel_wrap import (
        TraceableCuteDSLKernelWrapper,
    )

    if not is_cutedsl_jit_function(host_fn):
        raise RuntimeError(
            "wrap_cutedsl only works on host functions decorated with @cute.jit"
        )
    return TraceableCuteDSLKernelWrapper(host_fn, mutates_args)
