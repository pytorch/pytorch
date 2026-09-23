"""Path-cached ``torch.compiler.export_python`` decorator.

``torch.compiler.precompile`` captures a function ahead of time and lowers it to a
self-contained, human-readable Python source artifact (see
``torch/_precompile.py``). ``torch.compiler.export_python`` wraps that in a
decorator keyed off a file on disk: the first run writes the emitted
``python_code`` to ``path``; every later run reads the ``.py`` back and executes
it directly instead of recompiling.

Because the artifact is self-contained, re-executable Python, ``path`` is meant to be
committed and shipped -- and, when a kernel starts to matter, hand-edited in place by
an engineer or an agent. This is ejectable compilation: the emitted source is the
source of truth and is always exec'd, so an edit is simply what runs from then on, in
production as much as in development. There is no acceleration cache and no
``precompile.load`` round-trip, so keeping the edited source correct is the caller's
responsibility.
"""

import copy
import functools
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any, cast, TypeVar
from typing_extensions import ParamSpec

import torch


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


class ExportedPythonArtifact:
    """Materializes and disk-caches a ``torch.compiler.precompile`` artifact.

    Materialization is lazy and happens on the first call: if ``path`` exists the
    emitted Python is read from disk, otherwise the wrapped ``fn`` is precompiled
    against the example inputs and the emitted source is written to disk. Either
    way the source is exec'd directly to build the runnable. The loaded callable is
    reused for all subsequent calls in the process; a later process re-reads
    whatever is on disk.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        path: str,
        backend: str,
        tracer: str,
        decompositions: dict | None,
        example_inputs: Sequence[object] | None,
    ) -> None:
        self._fn = fn
        self._path = path
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._example_inputs = None if example_inputs is None else tuple(example_inputs)
        self._loaded: Callable[..., Any] | None = None

    def _precompile_and_save(self, args: tuple[Any, ...]) -> str:
        example = self._example_inputs
        if example is None:
            # Capture runs fn once on the example inputs (real-mode make_fx), which
            # mutates them; deep-copy the live call args so capture side effects (in-
            # place input mutation, module buffer updates) do not leak onto the
            # caller before the artifact itself runs on the real args exactly once.
            try:
                example = copy.deepcopy(args)
            except Exception as e:
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    "torch.compiler.export_python could not deep-copy the "
                    "first-call arguments to capture without mutating them (e.g. a "
                    "non-leaf tensor or a weight_norm module). Pass explicit "
                    "example_inputs=... to precompile against dedicated inputs."
                ) from e
        # precompile returns (python_code, cache); the cache is an acceleration
        # artifact that export_python does not use -- the emitted source is
        # self-contained and always exec'd -- so only the code is written to disk.
        code, _cache = torch.compiler.precompile(
            self._fn,
            *example,
            backend=self._backend,
            tracer=self._tracer,
            decompositions=self._decompositions,
        )
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            f.write(code)
        return code

    def _load_from_disk(self) -> str:
        with open(self._path, encoding="utf-8") as f:
            return f.read()

    def _load(self, code: str, *, from_disk: bool) -> Callable[..., Any]:
        # The emitted source is self-contained: exec it directly (no cache, no
        # precompile.load round-trip).
        from torch._precompile import _make_inlined_forward

        if from_disk:
            log.warning(
                "torch.compiler.export_python is about to EXEC the artifact at %s; "
                "the file is trusted executable Python and may have been edited or "
                "replaced since export. Only load paths whose contents you trust.",
                self._path,
            )
        return _make_inlined_forward(code, warn=False, filename=self._path)

    def _materialize(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        from_disk = os.path.exists(self._path)
        code = self._load_from_disk() if from_disk else self._precompile_and_save(args)
        entry = self._load(code, from_disk=from_disk)
        self._example_inputs = None
        self._decompositions = None
        return entry

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(
                "torch.compiler.export_python: the precompile calling convention is "
                f"positional; pass {sorted(kwargs)} positionally."
            )
        loaded = self._loaded
        if loaded is None:
            loaded = self._loaded = self._materialize(args)
        return loaded(*args)


def export_python(
    *,
    path: str,
    backend: str = "inductor",
    tracer: str = "make_fx",
    decompositions: dict | None = None,
    example_inputs: Sequence[object] | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """See :func:`torch.compiler.export_python`."""

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        artifact = ExportedPythonArtifact(
            fn,
            path=path,
            backend=backend,
            tracer=tracer,
            decompositions=decompositions,
            example_inputs=example_inputs,
        )

        @functools.wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return cast("_R", artifact(*args, **kwargs))

        return wrapped

    return decorator
