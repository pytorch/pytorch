"""``torch.compiler.export_python``: a decorator whose compile cache is a file you own."""

import functools
import os
import threading
from collections.abc import Callable
from typing import cast, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._precompile import (
    _DynamoCapture,
    capture,
    DynamoTracer,
    load,
    PrecompiledRunnable,
    PrecompileError,
)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def export_python(
    path: str | os.PathLike[str],
    *,
    backend: str = "inductor",
    dynamic: bool | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Compile a function once into a Python file, and run that file from then on.

    .. warning::

        This is a prototype API. Its signature, error types and artifact format may
        change between releases without a deprecation cycle.

    ``path`` is the whole cache key. If it does not exist, the first call is captured
    with :func:`torch.compiler.precompile.capture` under the default
    :class:`~torch.compiler.precompile.DynamoTracer` (so graph breaks, keyword
    arguments and Python control flow work as under :func:`torch.compile`), returns
    its real result, and writes the artifact to ``path`` and its acceleration cache to
    ``path + ".cache"``. Every later call, in this process or any other, serves from
    ``path`` through :func:`torch.compiler.precompile.load`: it dispatches to the
    captured variant whose guards hold, and does not trace or compile. A call no
    captured variant covers raises :class:`~torch.compiler.PrecompileError` rather than
    recompiling. Nothing hashes the function, its config or the machine, so editing
    the function does not recapture it: delete ``path`` to recapture. Concurrent first
    calls in one process capture once; processes that share ``path`` (the ranks of one
    job, say) should let one of them make the first call::

        @torch.compiler.export_python(path="artifacts/step.py")
        def step(model, x):
            return model(x).relu()


        y = step(model, x)  # first run: captures and writes artifacts/step.py
        y = step(model, x)  # served from artifacts/step.py, here or in a new process

    Capture in a fresh process: a frame that something else in the process already
    compiles or serves when the capture reaches it (a function another loaded artifact
    serves, or one compiled with :func:`torch.compile`) is not compiled into the
    artifact, which then runs that frame uncompiled.

    The artifact is trusted executable Python, locked to the PyTorch build and Python
    version that produced it. Decorate a function defined in an importable module:
    loading the artifact imports the module the function was captured from.

    Args:
        path: Where the artifact lives. Its parent directories are created on capture.
        backend: ``"inductor"`` (default) or ``"eager"``.
        dynamic: Forwarded to :class:`~torch.compiler.precompile.DynamoTracer`, as
            ``torch.compile(dynamic=)``.
    """

    torch._C._log_api_usage_once("torch.compiler.export_python")

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        exported = _ExportedPython(fn, path, backend, dynamic)

        @functools.wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return exported(*args, **kwargs)

        return wrapped

    return decorator


class _ExportedPython:
    def __init__(
        self,
        fn: Callable[..., object],
        path: str | os.PathLike[str],
        backend: str,
        dynamic: bool | None,
    ) -> None:
        if getattr(fn, "__module__", None) == "__main__":
            raise PrecompileError(
                f"export_python: {getattr(fn, '__qualname__', fn)!r} is defined in the "
                "__main__ script, which a later process cannot import to load its "
                "artifact. Define it in an importable module."
            )
        self._fn = fn
        self._path = os.fspath(path)
        self._cache_path = self._path + ".cache"
        self._backend = backend
        self._tracer = DynamoTracer(dynamic=dynamic)
        self._runnable: PrecompiledRunnable | None = None
        self._lock = threading.RLock()

    def __call__(self, *args: object, **kwargs: object) -> object:
        if self._runnable is None:
            with self._lock:
                if self._runnable is None:
                    if not os.path.exists(self._path):
                        return self._capture(*args, **kwargs)
                    self._runnable = load(self._path, self._cache_path)
        return self._runnable(*args, **kwargs)

    def _capture(self, *args: object, **kwargs: object) -> object:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        cap = capture(
            self._fn,
            artifact_path=self._path,
            cache_path=self._cache_path,
            tracer=self._tracer,
            backend=self._backend,
        )
        try:
            with cap:
                result = cap(*args, **kwargs)
        finally:
            # The capture's live compile holds the names its artifact binds on load,
            # and one that raised must not stay live under the next call's capture.
            cast(_DynamoCapture, cap)._release_live_compile()
        self._runnable = load(self._path, self._cache_path)
        return result


# Public as torch.compiler.export_python; rehomed like torch.compiler.PrecompileError.
export_python.__module__ = "torch.compiler"
