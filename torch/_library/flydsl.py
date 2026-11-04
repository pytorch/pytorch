import functools
import sys
from collections.abc import Callable, Iterable
from typing import Any, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from torch._higher_order_ops.flydsl_kernel_wrap import TraceableFlyDSLKernelWrapper


_MIN_FLYDSL_RELEASE = (0, 4)


@functools.cache
def _flydsl_version() -> str | None:
    try:
        import flydsl
    except ImportError:
        return None
    return str(getattr(flydsl, "__version__", "0"))


def flydsl_version_ok() -> bool:
    """Whether flydsl is importable and at least 0.4.0, the first release with what
    PT2 relies on (CompiledFunction.export_to_c and its call convention)."""
    from torch._vendor.packaging.version import InvalidVersion, Version

    version = _flydsl_version()
    if version is None:
        return False
    try:
        return Version(version).release[:2] >= _MIN_FLYDSL_RELEASE
    except InvalidVersion:
        return False


def require_flydsl() -> None:
    """Raise a RuntimeError unless flydsl_version_ok()."""
    if not flydsl_version_ok():
        version = _flydsl_version()
        found = "no importable flydsl" if version is None else f"flydsl {version}"
        raise RuntimeError(
            f"FlyDSL kernels in PT2 require flydsl >= 0.4.0, found {found}"
        )


def is_flydsl_jit_function(obj: object) -> bool:
    """Whether ``obj`` is a ``@flyc.jit`` host function: a FlyDSL JitFunction (a
    ``@flyc.kernel`` device kernel is a KernelFunction). Cheap enough for Dynamo's
    VariableBuilder: it imports nothing (an instance implies FlyDSL is imported)."""
    cls = type(obj)
    if cls.__name__ != "JitFunction" or not cls.__module__.startswith("flydsl."):
        return False
    module = sys.modules.get("flydsl.compiler.jit_function")
    return module is not None and isinstance(obj, module.JitFunction)


def is_traceable_flydsl_wrapper(obj: object) -> bool:
    """Whether ``obj`` is what wrap_flydsl returns, without importing the HOP
    module (an instance implies it is imported)."""
    module = sys.modules.get("torch._higher_order_ops.flydsl_kernel_wrap")
    return module is not None and isinstance(obj, module.TraceableFlyDSLKernelWrapper)


def wrap_flydsl(
    host_fn: Callable[..., Any],
    /,
    *,
    mutates_args: Iterable[str] | Literal["unknown"],
) -> "TraceableFlyDSLKernelWrapper":
    """Allows capture of a FlyDSL kernel launch into a graph via ``torch.compile``,
    make_fx or non-strict ``torch.export``: the counterpart of
    :func:`torch.library.wrap_triton` for ``@flyc.jit`` host functions (requires
    flydsl >= 0.4).

    ``host_fn`` must have exactly one parameter annotated ``fx.Stream``; callers of
    the returned wrapper do not pass it, because the kernel is launched on the
    current HIP stream. Tensor arguments are ``torch.Tensor``. ``mutates_args`` names
    the tensor parameters the kernel writes, or is ``"unknown"`` to treat every
    tensor argument as written; PT2 uses it to functionalize the launch. Two written
    arguments that may share memory (the same tensor, or views that may overlap)
    cannot be functionalized: PT2 refuses such a call, which eager runs.

    Examples:

        >>> # xdoctest: +SKIP
        >>> import flydsl.compiler as flyc
        >>> import flydsl.expr as fx
        >>> from torch._library.flydsl import wrap_flydsl
        >>>
        >>> @flyc.kernel
        >>> def add_device(X, Y, Out, n: fx.Int32):
        >>>     i = fx.block_idx.x * 1024 + fx.thread_idx.x
        >>>     if i < n:
        >>>         Out[i] = X[i] + Y[i]
        >>>
        >>> @flyc.jit
        >>> def add_kernel(X, Y, Out, n: fx.Int32, stream: fx.Stream):
        >>>     add_device(X, Y, Out, n).launch(
        >>>         grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream)
        >>>
        >>> add = wrap_flydsl(add_kernel, mutates_args=["Out"])
        >>>
        >>> @torch.compile(fullgraph=True)
        >>> def f(x, y):
        >>>     out = torch.empty_like(x)
        >>>     add(x, y, out, x.numel())
        >>>     return out.relu()
    """
    require_flydsl()
    from torch._higher_order_ops.flydsl_kernel_wrap import (
        TraceableFlyDSLKernelWrapper,
        tracing_flydsl_hopifier_singleton,
    )

    # A @flyc.jit method of an object: JitFunction.__get__ binds the object with
    # functools.partial(jit_function.__call__, obj).
    if isinstance(host_fn, functools.partial) and is_flydsl_jit_function(
        getattr(host_fn.func, "__self__", None)
    ):
        raise RuntimeError(
            "wrap_flydsl: @flyc.jit methods are not supported (FlyDSL binds the first "
            "parameter, 'self', to the object); wrap a @flyc.jit function without "
            "'self'"
        )
    if not is_flydsl_jit_function(host_fn):
        raise RuntimeError(
            "wrap_flydsl only works on host functions decorated with @flyc.jit"
        )
    # Checked here, where None is a user error: the wrapper takes None for a host
    # function that was not wrapped (Dynamo meets one raw).
    return TraceableFlyDSLKernelWrapper(
        host_fn, tracing_flydsl_hopifier_singleton.normalized_mutates_args(mutates_args)
    )
