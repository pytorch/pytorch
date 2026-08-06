"""
Ahead-of-time precompilation of a callable into MANY graphs.

``torch.compile(fn, fullgraph=True).aot_compile(...)`` produces exactly one
graph and rejects graph breaks. This module is the multi-graph counterpart: it
captures every frame Dynamo produces -- the entry frame, each
``torch_dynamo_resume_in_*`` continuation created by a graph break, and every
recompiled variant of each -- into a single serializable artifact.

Usage::

    session = precompile_capture(model, backend="inductor")
    with session as compiled:
        for variant in variants:  # every path you want covered
            with variant:
                compiled(*args)
    session.save(path)

    # later, in a fresh process
    compiled = precompile_load(model, path, backend="inductor")
    with serving():  # no compilation permitted
        compiled(*args)

Capture is by execution: a resume function only exists once the frame ahead of
it has actually run, so every variant must be exercised. Whatever you do not
run is not in the artifact.

Know these before relying on an artifact in production:

* Capture inference artifacts under ``torch.no_grad()`` or
  ``torch.inference_mode()``. AOTAutograd only records a bundled backend once
  the BACKWARD compiles, so a forward-only capture with grad enabled -- the
  default, and what ``model.eval()`` still leaves you in -- records no backends
  and cannot be saved. Capturing a training step that calls ``.backward()``
  works too.
* A value that crosses a graph break is guarded by equality, so a model whose
  breaks come from ``.item()`` or other data-dependent control flow yields an
  artifact that only serves inputs reproducing those exact values.
  ``summary().wont_generalize`` lists them; expect poor coverage on new data.
  ``dynamic=True`` helps with shapes but not with pinned values.
* Identity guards cannot be serialized, so precompiling gives up on noticing
  that a guarded object was rebound. ``summary().dropped_guards`` is the
  authoritative list and ``risky_dropped_guards`` is a lint over it, not a
  proof -- see ``_is_risky_drop`` for what it does and does not catch.
* The model must live in an importable module. Source is checksummed, so a
  class defined in ``__main__`` or a REPL cannot be loaded elsewhere.
* ``install()`` patches code objects process-globally, so an artifact is not
  scoped to the object it was loaded onto: other instances of the same class
  are served from it too, and ``torch._dynamo.reset()`` unloads it.

This wraps CompilePackage, which is the low-level component and is not meant to
be used directly.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import logging
import types
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING
from typing_extensions import Self

import torch
import torch._functorch.config as functorch_config

from .exc import PackageError
from .guards import CheckFunctionManager
from .package import _DynamoCacheEntry, CompilePackage, DiskDynamoStore


if TYPE_CHECKING:
    from .types import GuardFilterEntry


log = logging.getLogger(__name__)

__all__ = [
    "PrecompileSession",
    "PrecompileSummary",
    "PrecompiledCallable",
    "precompile_capture",
    "precompile_load",
    "serving",
]


def default_guard_filter_fn(
    guard_entries: Sequence[GuardFilterEntry],
) -> Sequence[bool]:
    """
    Drop the guard types that cannot be serialized, and keep everything else.

    Read this before trusting an artifact. The unserializable set is exactly the
    IDENTITY guards -- ID_MATCH, FUNCTION_MATCH, CLOSURE_MATCH, MODULE_MATCH,
    NN_MODULE, CLASS_MATCH, DICT_VERSION, WEAKREF_ALIVE -- so precompiling
    inherently gives up on noticing that a guarded object was REBOUND to a
    different object of the same shape. Most such guards are on modules and
    builtins and are stable in practice, but one on a global holding a function
    is not: rebind it between capture and load and the artifact serves the graph
    traced against the old one, with no error.

    There is no safe alternative default -- keeping these makes serialization
    raise for essentially every function -- so instead every drop is recorded
    with its source name in ``PrecompileSummary.dropped_guards``, and ``save()``
    refuses by default when a drop looks load-bearing. See ``risky_dropped_guards``.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]


@dataclasses.dataclass(frozen=True)
class PrecompileSummary:
    """What a capture actually produced. Assert on this in a build step."""

    frames: int
    resume_functions: int
    guarded_codes: int
    backend_graphs: int
    bypassed: tuple[str, ...]
    truncated: tuple[str, ...] = ()
    # Frames Dynamo produced but compiled nothing for. The entry frame landing
    # here means the model runs eager despite the artifact existing.
    uncovered_frames: tuple[str, ...] = ()
    # Guards pinning a value that crossed a graph break. These make the artifact
    # serve only the inputs it was captured with.
    wont_generalize: tuple[str, ...] = ()
    # (guard_type, source_name) for every guard the filter discarded / retained.
    dropped_guards: tuple[tuple[str, str], ...] = ()
    kept_guards: tuple[tuple[str, str], ...] = ()
    # Subset of dropped_guards whose loss can plausibly change results.
    risky_dropped_guards: tuple[tuple[str, str], ...] = ()

    @property
    def complete(self) -> bool:
        return not self.bypassed and not self.truncated and self.guarded_codes > 0

    def dropped_guard_types(self) -> dict[str, int]:
        return _count_types(self.dropped_guards)

    def kept_guard_types(self) -> dict[str, int]:
        return _count_types(self.kept_guards)

    def __str__(self) -> str:
        base = (
            f"{self.frames} frames ({self.resume_functions} from graph breaks), "
            f"{self.guarded_codes} guarded codes, "
            f"{self.backend_graphs} backend graphs"
        )
        if self.dropped_guards:
            base += f", dropped guards {self.dropped_guard_types()}"
        if self.risky_dropped_guards:
            base += f", RISKY drops {[n for _, n in self.risky_dropped_guards]}"
        if self.uncovered_frames:
            base += f", {len(self.uncovered_frames)} UNCOVERED: {list(self.uncovered_frames)}"
        if self.wont_generalize:
            base += f", {len(self.wont_generalize)} value-pinned guards"
        if self.truncated:
            base += f", {len(self.truncated)} TRUNCATED: {list(self.truncated)}"
        if self.bypassed:
            base += f", {len(self.bypassed)} BYPASSED: {list(self.bypassed)}"
        return base


def _owning_module(value: object) -> str | None:
    if isinstance(value, types.ModuleType):
        return value.__name__
    owner = getattr(value, "__module__", None)
    return owner if isinstance(owner, str) else None


def _is_risky_drop(source: str, value: object) -> bool:
    """
    Whether losing this identity guard can plausibly change results.

    Classify by what the guarded object IS, not by how the source is spelled.
    Source spelling cannot work: guard names have their local scope stripped, so
    a guard on ``self.act`` arrives as ``'self.act'`` and is indistinguishable
    from a global by pattern. Provenance survives that -- an object owned by
    torch or by builtins is library machinery whose binding no serving setup
    rebinds, while anything owned by user code is a candidate for the
    ``self.act = ACT2FN[cfg.act]`` dispatch shape, where a rebind silently serves
    the graph traced against the old callable.

    For a capture-here / serve-there deployment the concern is not in-process
    rebinding but DIVERGENCE: the serving machine runs the same source but picks
    a different object because config, a flag, or an env var differs.

    KNOWN GAP: only the capture-time value is visible. Capturing with a
    torch-owned callable and diverging to a different torch-owned one (``relu``
    -> ``sigmoid``) is classified benign and is NOT caught. ``dropped_guards`` is
    the authoritative list; this predicate is a lint over it, not a proof of
    safety.
    """
    owner = _owning_module(value)
    if owner is None:
        # Unknown provenance: assume it matters rather than quietly allowing it.
        return True
    return not (owner == "builtins" or owner == "torch" or owner.startswith("torch."))


def _count_types(pairs: Sequence[tuple[str, str]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for guard_type, _ in pairs:
        counts[guard_type] += 1
    return dict(counts)


def _summarize(
    entry: _DynamoCacheEntry,
    dropped: set[tuple[str, str]],
    kept: set[tuple[str, str]],
    risky: set[tuple[str, str]],
    truncated: frozenset[str],
) -> PrecompileSummary:
    # An entry with no guarded codes is skip_code()'d at install time, so that
    # frame runs eager forever. Resume frames legitimately have none when the
    # continuation was folded into a parent, so only flag the entry frame.
    uncovered = tuple(
        c.python_code.co_name
        for c in entry.codes[:1]
        if not c.guarded_codes and not c.bypassed
    )
    # Dynamo names the value that crossed a graph break ___stackN. Any guard
    # retained on one pins the artifact to the exact value seen at capture, and
    # the guard type varies (CONSTANT_MATCH, EQUALS_MATCH, ...), so key on the
    # source rather than the type.
    wont_generalize = tuple(sorted({n for _, n in kept if "___stack" in n}))
    return PrecompileSummary(
        frames=len(entry.codes),
        resume_functions=sum(1 for c in entry.codes if c.install_to_global),
        guarded_codes=sum(len(c.guarded_codes) for c in entry.codes),
        backend_graphs=len(entry.backend_ids),
        bypassed=tuple(c.python_code.co_name for c in entry.codes if c.bypassed),
        truncated=tuple(sorted(truncated)),
        uncovered_frames=uncovered,
        wont_generalize=wont_generalize,
        dropped_guards=tuple(sorted(dropped)),
        kept_guards=tuple(sorted(kept)),
        risky_dropped_guards=tuple(sorted(risky)),
    )


class PrecompileSession:
    """
    A capture in progress. Use as a context manager to get the callable to
    exercise, then ``save()``.
    """

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        backend: str = "inductor",
        guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
        | None = None,
        recompile_limit: int = 256,
        dynamic: bool | None = None,
    ) -> None:
        self._fn = fn
        self._backend = backend
        self._dropped_guards: set[tuple[str, str]] = set()
        self._kept_guards: set[tuple[str, str]] = set()
        self._risky_dropped_guards: set[tuple[str, str]] = set()
        self._guard_filter_fn = self._recording_filter(
            guard_filter_fn or default_guard_filter_fn
        )
        self._recompile_limit = recompile_limit
        self._dynamic = dynamic
        self._entry_fn = _entry_fn_of(fn)
        self._package = CompilePackage(self._entry_fn)
        self._stack: contextlib.ExitStack | None = None
        self._compiled: Callable[..., object] | None = None

    def __enter__(self) -> Callable[..., object]:
        if self._stack is not None:
            raise RuntimeError("PrecompileSession is already active")
        stack = contextlib.ExitStack()
        # Backends must serialize into the artifact rather than into the
        # process-local inductor cache.
        stack.enter_context(functorch_config.patch("bundled_autograd_cache", True))
        self._stack = stack
        self._compiled = torch._dynamo.optimize(
            self._backend,
            package=self._package,
            guard_filter_fn=self._guard_filter_fn,
            recompile_limit=self._recompile_limit,
            dynamic=self._dynamic,
        )(self._fn)
        return self._compiled

    def __exit__(self, *exc: object) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def _recording_filter(
        self,
        inner: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]],
    ) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
        """
        Remember which guard types were discarded. A dropped guard does not
        fail at serving time, it silently widens what a graph is reused for, so
        the set has to be inspectable rather than invisible.
        """

        def filter_fn(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
            decisions = inner(entries)
            for keep, entry in zip(decisions, entries):
                target = self._kept_guards if keep else self._dropped_guards
                target.add((entry.guard_type, entry.name))
                if not keep and _is_risky_drop(entry.name, entry.value):
                    self._risky_dropped_guards.add((entry.guard_type, entry.name))
            return decisions

        return filter_fn

    def summary(self) -> PrecompileSummary:
        return _summarize(
            self._package.cache_entry(),
            self._dropped_guards,
            self._kept_guards,
            self._risky_dropped_guards,
            self._package.truncated_frames,
        )

    def save(
        self,
        path: str,
        *,
        require_complete: bool = True,
        require_no_risky_drops: bool = True,
        require_no_dropped_guards: bool = False,
    ) -> PrecompileSummary:
        """
        Write the artifact, refusing by default to write one that cannot serve
        what it claims. See ``PrecompileSummary.complete``.
        """
        if self._stack is not None:
            raise RuntimeError("save() must be called after the capture block exits")
        summary = self.summary()
        if require_no_risky_drops and summary.risky_dropped_guards:
            raise PackageError(
                f"Precompilation dropped identity guard(s) on "
                f"{[n for _, n in summary.risky_dropped_guards]}. These guard objects "
                f"owned by your code rather than by torch, and identity guards cannot "
                f"be serialized, so nothing checks them at load time. Identical source "
                f"is not enough: if config, a feature flag, or an environment variable "
                f"selects a different object on the serving machine, the artifact "
                f"serves the graph traced against the capture-time object and returns "
                f"a wrong answer with no error. Make the value reachable as data the "
                f"graph can guard, pin it so both machines agree, or pass "
                f"require_no_risky_drops=False to accept the risk."
            )
        if require_no_dropped_guards and summary.dropped_guards:
            raise PackageError(
                f"Precompilation dropped {len(summary.dropped_guards)} guard(s) that "
                f"cannot be serialized: {list(summary.dropped_guards)}. Rebinding any "
                f"of those sources between capture and load would silently serve a "
                f"graph traced against the old value."
            )
        if require_complete:
            if summary.guarded_codes == 0:
                raise PackageError(
                    "Precompilation captured no compiled code. Capture happens by "
                    "execution, so the callable must actually be run inside the "
                    "capture block; a callable Dynamo traced to an empty graph also "
                    "lands here and cannot be precompiled."
                )
            if summary.truncated:
                raise PackageError(
                    f"Precompilation is incomplete: {len(summary.truncated)} frame(s) "
                    f"exceeded recompile_limit (currently {self._recompile_limit}) and "
                    f"are missing variants: {list(summary.truncated)}. A frame needs "
                    f"one slot per variant, and frames shared across module instances "
                    f"accumulate them. Raise recompile_limit, or pass "
                    f"require_complete=False to accept a partial artifact."
                )
            if summary.bypassed:
                raise PackageError(
                    f"Precompilation is incomplete: {len(summary.bypassed)} frame(s) "
                    f"were bypassed and will serve nothing: {list(summary.bypassed)}. "
                    f"This usually means their guards could not be serialized. Pass "
                    f"require_complete=False to accept a partial artifact."
                )
        if summary.uncovered_frames:
            # Not fatal: an entry frame that only dispatches to submodules has no
            # graph of its own while the submodules are still served. It IS how a
            # frame Dynamo gave up on looks though (gb0124), in which case the
            # model runs eager despite the artifact existing.
            log.warning(
                "precompile: no compiled code for entry frame(s) %s; install() will "
                "skip them, so they run eager. Expected if the frame only "
                "dispatches to submodules, otherwise check TORCH_LOGS=graph_breaks "
                "for a frame Dynamo could not handle.",
                list(summary.uncovered_frames),
            )
        if summary.wont_generalize:
            log.warning(
                "precompile: %d guard(s) pin a value that crossed a graph break "
                "(%s...). The artifact will only serve inputs producing those exact "
                "values; anything else misses every graph.",
                len(summary.wont_generalize),
                summary.wont_generalize[0],
            )
        store = DiskDynamoStore()
        if self._backend == "eager":
            # Eager "backends" are fx graphs with no compiled artifact of their
            # own, so they have to be handed to the store explicitly.
            for backend_id, backend in self._package.cached_backends.items():
                store.record_eager_backend(backend_id, backend)
        try:
            store.save_package(self._package, path)
        except RuntimeError as e:
            if "is not found in the given backends" not in str(e):
                raise
            raise PackageError(
                "Precompilation captured graphs but their compiled backends were "
                "never recorded, so there is nothing to serialize. AOTAutograd only "
                "records the bundled artifact once the BACKWARD compiles, so a "
                "forward-only capture with grad enabled -- the default, and what "
                "model.eval() still leaves you in -- records nothing. Capture under "
                "torch.no_grad() or torch.inference_mode() for an inference "
                "artifact, or run .backward() inside the capture block for a "
                "training one."
            ) from e
        log.info("precompile: saved %s to %s", summary, path)
        return summary


def precompile_capture(
    fn: Callable[..., object],
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
) -> PrecompileSession:
    """
    Begin capturing ``fn`` into a multi-graph artifact.

    ``recompile_limit`` defaults well above Dynamo's usual 8 because a
    precompile deliberately wants one compiled variant per condition, whereas
    the normal limit exists to catch runaway recompilation.
    """
    return PrecompileSession(
        fn,
        backend=backend,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
    )


def precompile_load(
    fn: Callable[..., object],
    path: str,
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
) -> PrecompiledCallable:
    """
    Load an artifact and return a callable ready to serve it.

    The wiring is order-sensitive -- the package has to be attached to the
    optimize context before its globals and guarded codes are installed -- so it
    is done here rather than left to callers.

    Installing mutates global state on the underlying code objects, which
    ``torch._dynamo.reset()`` does not undo, so the result is also a context
    manager that unloads on exit.
    """
    entry_fn = _entry_fn_of(fn)
    store = DiskDynamoStore()
    cache_entry = store.load_cache_entry(path)
    _check_artifact_matches(cache_entry.dynamo, entry_fn, path)
    package, backends = (
        CompilePackage(entry_fn, cache_entry.dynamo),
        cache_entry.backends,
    )
    compiled = torch._dynamo.optimize(
        backend,
        package=package,
        guard_filter_fn=guard_filter_fn or default_guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
    )(fn)
    package.install(backends)
    return PrecompiledCallable(compiled, package)


class PrecompiledCallable:
    """A loaded artifact. Call it, or use it as a context manager to scope it."""

    def __init__(
        self, compiled: Callable[..., object], package: CompilePackage
    ) -> None:
        self._compiled = compiled
        self._package = package

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._compiled(*args, **kwargs)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        """Remove installed globals and precompile entries from the code objects."""
        self._package.uninstall()


@contextlib.contextmanager
def serving() -> Iterator[None]:
    """
    Forbid compilation, so a call the artifact does not cover raises instead of
    quietly recompiling. This is process-wide, not a property of the artifact.
    """
    with torch.compiler.set_stance("fail_on_recompile"):
        yield


def _check_artifact_matches(
    dynamo: _DynamoCacheEntry, entry_fn: Callable[..., object], path: str
) -> None:
    """
    Refuse an artifact captured from a different callable.

    CompilePackage.initialize discards the serialized entry code object and
    rebinds the stored guards and bytecode onto whatever function it is given,
    and the source checksum only covers the ORIGINAL function's source, so a
    mismatch is not otherwise detected: the wrong callable simply returns the
    captured one's results.
    """
    code = getattr(entry_fn, "__code__", None)
    if code is None:
        return
    expected = dynamo.fn_name
    actual = getattr(entry_fn, "__qualname__", None)
    if expected is not None and actual is not None and expected != actual:
        raise PackageError(
            f"Artifact at {path} was captured from {expected!r} but is being "
            f"loaded onto {actual!r}. Loading it would serve the captured "
            f"function's graphs for this one."
        )
    stored = dynamo.codes[0].python_code if dynamo.codes else None
    if stored is not None and stored.co_name != code.co_name:
        raise PackageError(
            f"Artifact at {path} was captured from code object "
            f"{stored.co_name!r} but is being loaded onto {code.co_name!r}."
        )


@functools.singledispatch
def _entry_fn_of(fn: object) -> Callable[..., object]:
    if not callable(fn):
        raise TypeError(f"expected a callable or nn.Module, got {type(fn).__name__}")
    return fn  # type: ignore[return-value]


@_entry_fn_of.register
def _(fn: torch.nn.Module) -> Callable[..., object]:
    return fn.forward
