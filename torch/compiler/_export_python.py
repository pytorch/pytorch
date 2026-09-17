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
import errno
import functools
import inspect
import logging
import os
import secrets
import threading
from collections.abc import Callable, Sequence
from typing import Any, cast, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


# os.link failures that mean the filesystem cannot do hard links at all, as opposed to
# a real I/O problem (a full disk, a bad permission) that must not be swallowed.
_NO_HARDLINK_ERRNOS = frozenset(
    getattr(errno, name)
    for name in ("EPERM", "EOPNOTSUPP", "ENOTSUP", "EXDEV", "EMLINK", "ENOSYS")
    if hasattr(errno, name)
)


def _atomic_publish(path: str, data: bytes) -> bool:
    # Publish a fully-written file, never a partial one, and report whether this call
    # is the writer that published it. A hard link is the no-replace publish: exactly
    # one concurrent writer wins and every loser loads that winner rather than exec'ing
    # its own divergent source. Only errnos that mean "this filesystem has no hard
    # links" fall back to replace (last-writer-wins, still never partial); a full disk
    # or a permissions problem must surface rather than silently weaken the guarantee.
    dir_name = os.path.dirname(path) or "."
    base = os.path.basename(path)
    tmp = os.path.join(
        dir_name,
        f".{base}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(8)}.tmp",
    )
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        except OSError as e:
            if e.errno not in _NO_HARDLINK_ERRNOS:
                raise
            log.warning(
                "torch.compiler.export_python: %s has no hard links (%s), so %s is "
                "published last-writer-wins; concurrent first writers may each run "
                "their own generated source.",
                dir_name,
                e.strerror,
                path,
            )
            os.replace(tmp, path)
        return True
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


# Elements below this fraction of the reference's peak are too small for a relative diff
# to say anything; they are covered by the absolute term instead.
_REL_REPORT_FLOOR = 1e-6


def _allclose(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> bool:
    """torch.allclose, but tolerant of dtypes that have no comparison kernel.

    float8 reaches allclose and dies inside it on a missing mul (as NotImplementedError,
    which is a RuntimeError); promote and compare there rather than failing an artifact
    that is perfectly honest.
    """
    try:
        return bool(torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    except RuntimeError:
        return bool(
            torch.allclose(a.double(), b.double(), rtol=rtol, atol=atol, equal_nan=True)
        )


def _finite_mask(t: torch.Tensor) -> torch.Tensor | None:
    """Where ``t`` is finite, or None for a dtype that cannot answer.

    float8 is is_floating_point() but has no isfinite kernel, so asking crashes the
    check on an artifact that is perfectly honest.
    """
    if not t.is_floating_point():
        return None
    try:
        return torch.isfinite(t)
    except RuntimeError:
        return None


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
        self._signature = inspect.signature(fn)
        self._call_signature = self._signature
        self._path = path
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._example_inputs = None if example_inputs is None else tuple(example_inputs)
        self._loaded: Callable[..., Any] | None = None
        # (pid, tid) currently inside _materialize. There is deliberately no
        # per-artifact lock: capture is already serialized process-wide, so a second
        # lock would add nothing but a second acquisition order to deadlock against.
        # This is the re-entrancy guard the (reentrant) capture lock cannot provide.
        # The pid makes it fork-safe without a registry -- a child never matches a
        # marker left by a thread it did not inherit.
        self._materializing: tuple[int, int] | None = None

    def _precompile_and_save(self, args: tuple[Any, ...]) -> tuple[str, bool]:
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
            example = self._bind_positional(example, {}, "example_inputs=")
            self._check_supported_args(example)
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
        if _atomic_publish(self._path, code.encode("utf-8")):
            return code, False
        # Lost the publish race; the winner's file is complete and already linked.
        winner = self._load_from_disk()
        if winner is None:
            raise _precompile_error(
                f"torch.compiler.export_python: another writer published {self._path} "
                "and it was deleted before this call could load it. Retry."
            )
        return winner, True

    def _load_from_disk(self) -> str | None:
        # None means "not there after all" -- the presence gate raced a peer deleting
        # the artifact to force a regenerate, which should fall through to capture
        # rather than surface a bare FileNotFoundError.
        try:
            with open(self._path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise _precompile_error(
                f"torch.compiler.export_python: could not read the artifact at "
                f"{self._path} ({e.strerror}). Check that the path names a readable "
                "file rather than a directory."
            ) from e

    def _load(self, code: str, *, from_disk: bool) -> Callable[..., Any]:
        # The emitted source is self-contained: exec it directly (no cache, no
        # precompile.load round-trip). A clobbered hand-edit (dropped forward / syntax
        # error) and an environment or version mismatch (an import that fails under the
        # current torch) surface as distinct, actionable PrecompileErrors rather than
        # one catch-all "delete to regenerate".
        from torch._precompile import _make_inlined_forward, PrecompileError

        try:
            if from_disk:
                log.warning(
                    "torch.compiler.export_python is about to EXEC the artifact at %s; "
                    "the file is trusted executable Python and may have been edited or "
                    "replaced since export. Only load paths whose contents you trust.",
                    self._path,
                )
            return _make_inlined_forward(code, warn=False, filename=self._path)
        except SyntaxError as e:
            # Kernels are hoisted to module level, so Python reports a typo in one
            # against this file at the right line. Say so: telling someone to delete an
            # artifact they are midway through tuning is the wrong advice.
            where = f" at line {e.lineno}" if e.lineno else ""
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} does not "
                f"parse{where}: {e.msg}. Fix it there, or delete the file to regenerate "
                "from the original function."
            ) from e
        except KeyError as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} could "
                "not be run as precompile source; it is not a valid "
                "torch.compiler.precompile artifact (a hand-edit may have clobbered "
                "it, e.g. dropping forward()). Delete it to regenerate."
            ) from e
        except ImportError as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} failed "
                "to import a dependency; it was likely produced by a different torch "
                f"version or environment. Delete {self._path} to regenerate against "
                "the current torch."
            ) from e
        except Exception as e:
            raise PrecompileError(
                "torch.compiler.export_python: an unexpected error occurred running "
                f"the artifact at {self._path}. Delete it to regenerate."
            ) from e

    def _materialize(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        code = self._load_from_disk() if os.path.exists(self._path) else None
        from_disk = code is not None
        if code is None:
            code, from_disk = self._precompile_and_save(args)
        entry = self._load(code, from_disk=from_disk)
        self._example_inputs = None
        self._decompositions = None
        return entry

    def _materialize_once(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        # Materialization runs under the one process-wide capture lock. Capture runs
        # fn, which may call another decorated function, so any second lock taken
        # around this would give two threads two orders to acquire them in and deadlock
        # -- which is why the artifact holds no lock of its own.
        import torch._precompile as precompile_impl

        ident = (os.getpid(), threading.get_ident())
        if self._materializing == ident:
            raise _precompile_error(
                "torch.compiler.export_python: re-entrant call into "
                f"{getattr(self._fn, '__name__', 'fn')} while it is being precompiled. "
                "A decorated function cannot call itself: capture would have to run "
                "inside its own capture. Move the recursion into an undecorated helper."
            )
        with precompile_impl._CAPTURE_LOCK:
            if self._loaded is None:
                self._materializing = ident
                try:
                    self._loaded = self._materialize(args)
                finally:
                    self._materializing = None
            return self._loaded

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
        # bound.kwargs holds every argument bind() could not place positionally. That
        # is a keyword-only / **kwargs param (never positional), or a plain
        # positional-or-keyword param passed by keyword while an earlier one was left
        # to its default -- distinguish them so the error names the real cause.
        if bound.kwargs:
            params = sig.parameters
            kw_only = sorted(
                n
                for n in bound.kwargs
                if n in params and params[n].kind == inspect.Parameter.KEYWORD_ONLY
            )
            if kw_only:
                raise TypeError(
                    "torch.compiler.export_python does not support keyword-only "
                    f"parameters (got {kw_only}); the precompile calling convention "
                    "is positional."
                )
            # Names not declared as parameters were absorbed by a **kwargs param;
            # they are never positional, so name **kwargs as the cause rather than
            # misreporting them as a positional-or-keyword arg left to its default.
            var_kw = sorted(n for n in bound.kwargs if n not in params)
            if var_kw:
                raise TypeError(
                    "torch.compiler.export_python does not support **kwargs "
                    f"parameters (got {var_kw}); the precompile calling convention "
                    "is positional."
                )
            raise TypeError(
                "torch.compiler.export_python could not place keyword arguments "
                f"{sorted(bound.kwargs)} positionally because an earlier positional "
                "parameter was left to its default; pass those arguments positionally "
                "or provide example_inputs."
            )
        return bound.args

    def _check_supported_args(self, args: tuple[Any, ...]) -> None:
        params = list(self._call_signature.parameters)
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
            name = params[pos] if pos < len(params) else f"argument {pos}"
            # These two land often enough that the generic "close the constant over"
            # advice is actively wrong for them: a module must stay an argument, and an
            # optional parameter has no constant to close over in the first place.
            if any(isinstance(leaf, torch.nn.Module) for leaf in unsupported):
                raise TypeError(
                    "torch.compiler.export_python: nn.Module arguments must be passed "
                    f"directly, not nested inside a container (parameter {name!r}). "
                    "Pass the module itself as its own positional argument."
                )
            if all(leaf is None for leaf in unsupported):
                raise TypeError(
                    "torch.compiler.export_python does not support None arguments "
                    f"(parameter {name!r}); make_fx specializes the None branch without "
                    "a runtime guard. Split the function, or pass a tensor."
                )
            raise TypeError(
                "torch.compiler.export_python supports only Tensor pytrees and "
                "nn.Module positional arguments; Python scalar/config values are "
                "specialized by make_fx without runtime guards. Close constants "
                f"over in the function instead of passing parameter {name!r} "
                f"({unsupported[0]!r})."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        args = self._bind_positional(args, kwargs)
        self._check_supported_args(args)
        loaded = self._loaded
        if loaded is None:
            loaded = self._materialize_once(args)
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
