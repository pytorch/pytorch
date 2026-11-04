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
    # they call: "_func" for host functions, "_kernel_helper" for device kernels. A
    # functools.wraps wrapper of one (a decorator, torch.compiler.disable) copies its
    # _dsl_cls but closes over no such name.
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
    # Without cutlass imported, no function has a CuTeDSL _dsl_cls.
    if not isinstance(dsl_cls, type) or sys.modules.get("cutlass") is None:
        return False
    from cutlass.cutlass_dsl import CuTeDSL

    return issubclass(dsl_cls, CuTeDSL) and _jit_executor_name(obj) == "_func"


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
    current CUDA stream, so positional arguments bind to the parameters before it
    and a parameter after it is passed by keyword. Tensor arguments are
    ``torch.Tensor`` (they are converted with ``cutlass.cute.runtime.from_dlpack``),
    and a parameter whose default is a tensor must be passed one. ``mutates_args``
    names the tensor parameters the kernel writes, or is ``"unknown"`` to treat every
    tensor argument as written; PT2 uses it to functionalize the launch.

    A call of the wrapper is a call of the ``cutedsl_kernel_wrapper_mutation`` HOP,
    which make_fx and export record. As for ``wrap_triton`` wrappers,
    :func:`torch._library.triton.set_wrap_triton_enabled` can disable that (a
    ``triton_op``'s eager backend does): the wrapper then checks the call and
    launches the kernel directly.

    The kernel is compiled for its constant arguments (Constexpr ones, and others
    that are neither tensors nor scalars), so the compile cache keys them, as does
    the side table that holds them for make_fx and export: a list, tuple, dict,
    dataclass instance or plain object by its type and contents, which must not hold
    a tensor or a DSL tensor, pointer or stream, and must be hashable values or such
    containers. A bound ``@cute.jit`` method or a kernel object (one whose class's
    ``__call__`` is ``@cute.jit``) cannot be wrapped: wrap a ``@cute.jit`` function
    that takes the object as a ``cutlass.Constexpr`` parameter and calls the method
    on it, and pass the object to that wrapper. A kernel object is keyed by its
    attributes like a plain object: equal objects, such as one built for each call,
    share one compile, one mutated between calls is compiled again, and one with an
    unhashable attribute (a set, a numpy array, or a cute layout that a direct
    ``cute.compile`` of it stored on it) is refused.

    Dynamo captures a raw call as the call of ``wrap_cutedsl(host_fn,
    mutates_args="unknown")``: every tensor argument counts as written, read-only
    ones included. Each is cloned in the functionalized graph; a graph input passed
    to the call becomes a mutated input, copied back after the graph runs and listed
    in ``graph_signature.user_inputs_to_mutate`` by strict ``torch.export``; and the
    same tensor passed for two parameters, or views that may overlap, graph-break
    (two written arguments that share memory cannot be functionalized), although
    eager runs the call. To give the precise write set, call the kernel through
    ``wrap_cutedsl(host_fn, mutates_args=[...])`` instead.

    Examples:

        >>> # xdoctest: +SKIP
        >>> import cuda.bindings.driver as cuda
        >>> import cutlass
        >>> import cutlass.cute as cute
        >>> import torch
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
        cutedsl_kernel_name,
        TraceableCuteDSLKernelWrapper,
    )

    # A kernel object stands for its bound __call__, as in cute.compile.
    bound = host_fn
    if callable(host_fn) and not inspect.ismethod(host_fn):
        bound = host_fn.__call__
    if inspect.ismethod(bound) and is_cutedsl_jit_function(bound.__func__):
        raise TypeError(
            f"CuTeDSL kernel {cutedsl_kernel_name(bound)}: wrap_cutedsl does not "
            f"support bound methods or kernel objects (got {bound.__qualname__}): "
            "instead, wrap a @cute.jit function that takes the object as a "
            "cutlass.Constexpr parameter and calls the method on it, and pass the "
            "object to that wrapper, which keys a kernel object by its attributes: "
            "equal objects share one compile, and one mutated between calls is "
            "compiled again"
        )
    if not is_cutedsl_jit_function(host_fn):
        raise TypeError(
            f"CuTeDSL kernel {cutedsl_kernel_name(host_fn)}: wrap_cutedsl only works "
            "on host functions decorated with @cute.jit"
        )
    return TraceableCuteDSLKernelWrapper(host_fn, mutates_args)
