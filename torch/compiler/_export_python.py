"""``torch.compiler.export_python``: a decorator whose compile cache is a file you own."""

import functools
import logging
import os
from collections.abc import Callable
from typing import cast, TypeVar
from typing_extensions import ParamSpec

from torch._precompile import (
    _cache_matches_artifact,
    _DynamoCapture,
    capture,
    DynamoTracer,
    load,
    PrecompiledRunnable,
    PrecompileError,
)


log = logging.getLogger(__name__)

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
    the function does not recapture it: delete ``path`` to recapture.

    ``path`` is also the artifact to edit: each compiled graph's Inductor code is in
    its ``GRAPH_SOURCES`` section, and that text is what runs. An edited file is
    served without its cache (built for the captured text), and its edited graphs
    compile the kernels their text defines. A process serves one version of a
    capture: rerun in a new process to pick up an edit, since loading the edited
    file where an earlier version of it is loaded raises::

        @torch.compiler.export_python(path="artifacts/step.py")
        def step(model, x):
            return model(x).relu()


        y = step(model, x)  # first run: captures and writes artifacts/step.py
        y = step(model, x)  # served from artifacts/step.py, here or in a new process

    The artifact is trusted executable Python, locked to the PyTorch build and Python
    version that produced it. Decorate a function defined in an importable module:
    loading the artifact imports the module the function was captured from.

    Args:
        path: Where the artifact lives. Its parent directories are created on capture.
        backend: ``"inductor"`` (default) or ``"eager"``.
        dynamic: Forwarded to :class:`~torch.compiler.precompile.DynamoTracer`, as
            ``torch.compile(dynamic=)``.
    """

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

    def __call__(self, *args: object, **kwargs: object) -> object:
        if self._runnable is None:
            if not os.path.exists(self._path):
                return self._capture(*args, **kwargs)
            self._runnable = self._load()
        return self._runnable(*args, **kwargs)

    def _load(self) -> PrecompiledRunnable:
        cache: str | None = self._cache_path
        if not _cache_matches_artifact(self._path, self._cache_path):
            # An edit (or a missing cache) leaves nothing to accelerate: the edited
            # graphs compile the kernels their text defines.
            log.info(
                "export_python: %s does not match its cache %s (edited after capture, "
                "or the cache is missing); serving it without the cache",
                self._path,
                self._cache_path,
            )
            cache = None
        return load(self._path, cache)

    def _capture(self, *args: object, **kwargs: object) -> object:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with capture(
            self._fn,
            artifact_path=self._path,
            cache_path=self._cache_path,
            tracer=self._tracer,
            backend=self._backend,
        ) as cap:
            result = cap(*args, **kwargs)
        # The capture's live compile holds the names its artifact binds on load.
        cast(_DynamoCapture, cap)._release_live_compile()
        self._runnable = self._load()
        return result


# Public as torch.compiler.export_python; rehomed like torch.compiler.PrecompileError.
export_python.__module__ = "torch.compiler"
