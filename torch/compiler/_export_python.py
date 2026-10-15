"""Path-cached ``torch.compiler.export_python`` decorator.

``torch.compiler.precompile`` captures a function ahead of time and lowers it to a
self-contained, human-readable Python source artifact (see
``torch/_precompile.py``). ``torch.compiler.export_python`` wraps that in a
decorator keyed off a file on disk: the first run writes the emitted
``python_code`` to ``path``; every later run reads the ``.py`` back and executes
it directly instead of recompiling.
"""

import copy
import functools
import inspect
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any, cast, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _precompile_error(msg: str) -> Exception:
    from torch._precompile import PrecompileError

    return PrecompileError(msg)


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
        self._call_signature = inspect.signature(fn)
        kw_only = [
            p.name
            for p in self._call_signature.parameters.values()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        if kw_only:
            raise TypeError(
                "torch.compiler.export_python does not support functions that "
                f"declare keyword-only parameters ({kw_only}); the precompile "
                "calling convention is positional."
            )
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
        else:
            example = self._bind_positional(example, {}, "example_inputs")
            self._check_supported_args(example, "example_inputs")
        # Only the python_code is written: the emitted source is self-contained and
        # always exec'd, so export_python never builds precompile's acceleration cache.
        from torch._precompile import PrecompiledModule

        compiled = PrecompiledModule(
            self._fn,
            backend=self._backend,
            tracer=self._tracer,
            decompositions=self._decompositions,
        )
        compiled._compile(example)
        code = compiled.to_python_code()
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            f.write(code)
        return code

    def _load_from_disk(self) -> str | None:
        # None means "not there after all" -- the presence gate raced a peer deleting
        # the artifact to force a regenerate, which should fall through to capture
        # rather than surface a bare FileNotFoundError.
        try:
            with open(self._path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as e:
            hint = " rather than a directory" if os.path.isdir(self._path) else ""
            raise _precompile_error(
                f"torch.compiler.export_python: could not read the artifact at "
                f"{self._path} ({e}). Check that the path names a readable UTF-8 "
                f"file{hint}."
            ) from e

    def _load(self, code: str, *, from_disk: bool) -> Callable[..., Any]:
        # The emitted source is self-contained: exec it directly (no cache, no
        # precompile.load round-trip). A broken hand-edit and an environment or version
        # mismatch (an import that fails under the current torch) surface as distinct,
        # actionable PrecompileErrors rather than one catch-all "delete to regenerate".
        from torch._precompile import _make_inlined_forward, PrecompileError

        if not from_disk:
            # Source this call just captured is precompile's own output, so a failure
            # running it is a bug to surface as-is, not a file to fix or delete.
            return _make_inlined_forward(code, warn=False, filename=self._path)
        log.warning(
            "torch.compiler.export_python is about to EXEC the artifact at %s; "
            "the file is trusted executable Python and may have been edited or "
            "replaced since export. Only load paths whose contents you trust.",
            self._path,
        )
        try:
            return _make_inlined_forward(code, warn=False, filename=self._path)
        except PrecompileError:
            raise
        except SyntaxError as e:
            if e.filename != self._path:
                # Raised by code the artifact runs (an import, a nested exec); the
                # line belongs to that file, so report it like any other failure.
                raise PrecompileError(
                    "torch.compiler.export_python: an unexpected error occurred running "
                    f"the artifact at {self._path} ({type(e).__name__}: {e}). Fix it "
                    "there, or delete it to regenerate."
                ) from e
            # Kernels are hoisted to module level, so Python reports a typo in one
            # against this file at the right line. Say so: telling someone to delete an
            # artifact they are midway through tuning is the wrong advice.
            where = f" at line {e.lineno}" if e.lineno else ""
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} does not "
                f"parse{where}: {e.msg}. Fix it there, or delete the file to regenerate "
                "from the original function."
            ) from e
        except ImportError as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} failed "
                f"to import a dependency ({e}); it was edited, or produced by a "
                "different torch version or environment. Fix it there, or delete it "
                "to regenerate against the current torch."
            ) from e
        except Exception as e:
            raise PrecompileError(
                "torch.compiler.export_python: an unexpected error occurred running "
                f"the artifact at {self._path} ({type(e).__name__}: {e}). Fix it there, "
                "or delete it to regenerate."
            ) from e

    def _materialize(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        code = self._load_from_disk() if os.path.exists(self._path) else None
        from_disk = code is not None
        if code is None:
            code = self._precompile_and_save(args)
        entry = self._load(code, from_disk=from_disk)
        self._example_inputs = None
        self._decompositions = None
        return entry

    def _bind_positional(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        source: str = "the call arguments",
    ) -> tuple[Any, ...]:
        # The artifact's forward is positional (the precompile calling convention),
        # so map any keyword call args onto fn's positional parameters -- this lets
        # callers invoke the decorated fn naturally (e.g. rope(q=..., k=...)).
        # Anything that cannot be laid out positionally is rejected below.
        sig = self._call_signature
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"torch.compiler.export_python: could not bind {source} to "
                f"{getattr(self._fn, '__name__', 'fn')}'s signature: {e}"
            ) from e
        bound.apply_defaults()
        # After apply_defaults every positional-or-keyword parameter is placed and
        # __init__ refused keyword-only ones, so bound.kwargs holds only **kwargs.
        if bound.kwargs:
            raise TypeError(
                "torch.compiler.export_python does not support **kwargs parameters "
                f"(got {sorted(bound.kwargs)}); the precompile calling convention is "
                "positional."
            )
        return bound.args

    def _check_supported_args(
        self, args: tuple[Any, ...], source: str = "the call arguments"
    ) -> None:
        # args is the bound positional layout: the named positional parameters in
        # order, then any *args values.
        P = inspect.Parameter
        params = self._call_signature.parameters.values()
        positional = (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)
        names = [p.name for p in params if p.kind in positional]
        var = next((p.name for p in params if p.kind == P.VAR_POSITIONAL), None)
        for pos, arg in enumerate(args):
            if isinstance(arg, torch.nn.Module):
                continue
            unsupported = [
                leaf
                for leaf in pytree.tree_leaves(arg)
                if not isinstance(leaf, torch.Tensor)
            ]
            if not unsupported:
                continue
            name = names[pos] if pos < len(names) else f"{var}[{pos - len(names)}]"
            where = f"parameter {name!r} of {source}"
            # These two land often enough that the generic "close the constant over"
            # advice is actively wrong for them: a module must stay an argument, and an
            # optional parameter has no constant to close over in the first place.
            if any(isinstance(leaf, torch.nn.Module) for leaf in unsupported):
                raise TypeError(
                    "torch.compiler.export_python: nn.Module arguments must be passed "
                    f"directly, not nested inside a container ({where}). "
                    "Pass the module itself as its own positional argument."
                )
            if all(leaf is None for leaf in unsupported):
                raise TypeError(
                    "torch.compiler.export_python does not support None arguments "
                    f"({where}); make_fx specializes the None branch without "
                    "a runtime guard. Split the function, or pass a tensor."
                )
            raise TypeError(
                "torch.compiler.export_python supports only Tensor pytrees and "
                "nn.Module positional arguments; Python scalar/config values are "
                "specialized by make_fx without runtime guards. Close constants "
                f"over in the function instead of passing {unsupported[0]!r} as "
                f"{where}."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        args = self._bind_positional(args, kwargs)
        self._check_supported_args(args)
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
