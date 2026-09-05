"""
Ahead-of-time precompilation of a callable into MANY graphs: the multi-graph
counterpart of ``torch.compile(fn, fullgraph=True).aot_compile(...)``. Every
frame Dynamo produces while the caller's calls run -- the entry
frame, each ``torch_dynamo_resume_in_*`` continuation created by a graph break,
and every recompiled variant of each -- is captured into one serializable
artifact on top of CompilePackage.

Everything here is internal; the session that drives it and the
``torch.compiler.precompile.capture(..., tracer=DynamoTracer())`` entry point build on
these in later commits. This is distinct from
``torch._dynamo.config.caching_precompile``, which caches ``torch.compile``
artifacts transparently without an explicit capture.

Capture is by execution, and the caller drives it: the session hands back a
callable, the caller invokes it with real inputs inside their own loop, and
every frame Dynamo produces is recorded. Runtime guards stay intact during
capture; ``guard_filter_fn`` applies only to the serialized copy, and every
dropped guard is reported in ``PrecompileSummary.dropped_guards``.

    with torch.compiler.precompile.capture(
        step, artifact_path="m.py", cache_path="m.cache", backend="inductor"
    ) as cap:
        y1 = cap(model, x1)  # runs step(model, x1), returns its result
        y2 = cap(model, x2)  # exercises another variant

    # later, in a fresh process
    compiled = torch.compiler.precompile.load("m.py", "m.cache")
    with compiled, torch.no_grad():
        compiled(model, x1)

The caller's calls ARE the capture: each ``cap(...)`` runs the callable for
real, returns its result, and records every frame, break continuation and
guarded variant it exercises. ``precompile.accumulate`` is the same model, rewriting the artifact on every
call instead of once at block exit.

Calls run with the grad mode the caller sets -- capture does not force
``no_grad()`` or ``enable_grad()``. ``training=True`` lowers the backward
eagerly so the artifact carries one and a served output can be backpropagated.
No loss is needed for that: the joint trace synthesizes tangents from the
forward outputs' own metadata.

Live capture retains every runtime guard, so later examples trigger the same
recompilations as ordinary ``torch.compile``. ``guard_filter_fn`` applies only
to the serialized copy. If serialization drops a configuration-dependent guard,
the artifact is refused by default rather than written with variants whose
dispatch would be ambiguous after load. ``invariants`` writes a readable report
that separates, per frame, the guards holding in EVERY variant from the ones
that differed: the first are preconditions the artifact is only valid under,
the second are what tell its graphs apart. Guards from different frames are not
comparable -- an entry frame guards its arguments, a resume frame guards
whatever crossed the break -- so the intersection is per frame.

Capture is by execution: a resume function only exists once the frame ahead of
it has actually run, so every variant must be exercised. Whatever you do not
run is not in the artifact, and ``summary().complete`` means complete only for
the observed capture, not for every possible input to the callable. A captured
call that raises marks the session incomplete even if caller code catches it.

Know these before relying on an artifact in production:

* An inference artifact is the default: the caller runs the calls under
  ``torch.no_grad()``. For a training artifact pass ``training=True``, which
  traces with grad on and lowers the backward eagerly -- without it, AOTAutograd
  defers the backward to the first ``.backward()`` call, so a grad-enabled
  capture that never makes one records no backends and cannot be written.
* A non-tensor argument, and any value that crosses a graph break, is guarded
  by equality, so an int/bool/str argument or a break coming from ``.item()``
  yields an artifact that only serves calls reproducing those exact values.
  ``summary().wont_generalize`` lists them; exercise every value you need to
  serve with a ``cap(...)`` call, or expect poor coverage on new data.
  ``dynamic=True`` helps with shapes but not with pinned values.
* Identity guards cannot be serialized, so precompiling gives up on noticing
  that a guarded object was rebound. ``summary().dropped_guards`` is the
  authoritative list. ``risky_dropped_guards`` includes every drop observed to
  distinguish captured variants plus a lint for configuration-like sources; it
  is still not a proof for unobserved deployments. See ``_is_risky_drop``. The
  public ``torch.compiler.precompile`` facade rejects the RISKY subset by
  default. Refusing every drop is opt-in: every model drops the identity guards
  precompile cannot serialize, so ``require_no_dropped_guards=True`` refuses
  essentially every real artifact. Some models trip the lint on
  library internals: measured on stock models, torchvision resnet18 and
  mobilenet_v3 report none, timm's ViT reports one (a re-exported
  ``torch._assert``) and transformers' Qwen2 reports 33 built from a two-layer
  config, 55 for the pretrained 24-layer, of which only the
  attention-implementation registry looks genuinely config-selected. Report
  counts are per model, not per library: torchvision's efficientnet_b0 reports
  2 and timm's swin reports 5, one of which is a real config slot. Audit the
  list before relying on the relaxed dropped-guard default, and before
  relaxing the risky-drop rail on top of it.
* Some models do not capture yet. For example, T5 raises ``PackageError: Cannot
  find module for code <code object __init__`` from ``_get_code_source``, which
  is byte-identical to base and which plain ``caching_precompile`` also raises.
* The model must live in an importable module. Source is checksummed, so a
  class defined in ``__main__`` or a REPL cannot be loaded elsewhere.
* ``install()`` writes compiled and resume functions into module globals, but
  guarded dispatch is scoped to the isolated compile region owned by the
  returned callable. Call the returned object rather than another instance of
  the same class. Multiple loaded artifacts can share entry, inner, and resume
  code objects without taking each other's entries; ``unload()`` removes only
  its own region and the globals it still owns.

This wraps CompilePackage, which is the low-level component and is not meant to
be used directly.

The public surface is ``torch.compiler.precompile.capture(...)``, a caller-driven
capture used as a context manager: the caller's own calls inside the block drive
the capture, and the ``(python_code, cache)`` artifact is written to the given files
when the block exits (its default ``tracer=DynamoTracer()`` records many calls;
``tracer=MakeFxTracer()`` produces a self-contained Python source artifact from one
call); ``torch.compiler.precompile.accumulate(...)``, the counterpart that rewrites
the files after every call; and ``torch.compiler.precompile.load``.
The helpers in this module, including the capture session, implement that surface
and remain internal. All of it is distinct from ``torch._dynamo.config.caching_precompile``,
which caches ``torch.compile`` artifacts transparently without an explicit
capture block.
"""

from __future__ import annotations

import collections
import contextlib
import contextvars
import functools
import hashlib
import importlib.machinery
import logging
import os
import re
import site
import sys
import sysconfig
import types
from typing import Any, TYPE_CHECKING

import torch
import torch._functorch.config as functorch_config
from torch._guards import ChainedSource, Source
from torch.compiler._precompile_types import (
    FrameInvariants,
    GuardFact as _GuardFact,
    PrecompileSummary,
)

from .convert_frame import CatchErrorsWrapper
from .exc import PackageError
from .guards import CheckFunctionManager
from .source import AttrSource, DictGetItemSource, GlobalSource


if TYPE_CHECKING:
    import traceback
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from .convert_frame import ConvertFrameReturn
    from .eval_frame import OptimizeContext
    from .package import _DynamoCacheEntry, CompilePackage
    from .types import CacheEntry, DynamoFrameType, GuardFilterEntry
    from .variables.builder import FrameStateSizeEntry


log = logging.getLogger(__name__)

# Built once: config.patch() allocates a class and a ContextVar each time it is
# called, and this runs on every frame Dynamo compiles for a package.
_ALLOW_EMPTY_GRAPHS = torch._dynamo.config._make_closure_patcher(
    allow_empty_graphs=True
)

# Not a public surface -- see the module docstring. This exists so `from ...
# import *` in a debugging session pulls the entry points rather than every
# private helper, and so linters do not flag them as unused.
__all__ = [
    "FrameInvariants",
    "PrecompileSummary",
]


# Depth per context so overlapping sessions on one thread patch once and
# restore once; a worker thread starts at zero and patches for itself.
_CAPTURE_CONFIG_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_CAPTURE_CONFIG_DEPTH", default=0
)
_CAPTURE_CONFIG_STACK: contextvars.ContextVar[contextlib.ExitStack | None] = (
    contextvars.ContextVar("_CAPTURE_CONFIG_STACK", default=None)
)


@contextlib.contextmanager
def _capture_config(training: bool) -> Iterator[None]:
    # Backends serialize into the artifact rather than the process-local
    # inductor cache. AOTAutograd lowers the backward lazily on the first
    # .backward(), so a training capture that never makes one forces it eager.
    depth = _CAPTURE_CONFIG_DEPTH.get()
    if depth == 0:
        functorch_patch: dict[str, Any] = {
            "bundled_autograd_cache": True,
            # AOTAutogradCache refuses to KEY a graph it cannot address soundly
            # -- a graph calling anything outside its allowlist -- and a refusal
            # means it never saves, so the bundled artifact precompile needs is
            # never recorded and the capture ends with nothing to serialize.
            # That gate asks whether the key tells this graph's behaviour apart
            # from another's, which a precompile artifact does not depend on: it
            # is addressed by backend id and pinned to one torch build, so fall
            # back to a nonce key rather than declining, as
            # torch._dynamo.aot_compile and aot_compile_joint_with_descriptors
            # already do.
            "bypass_autograd_cache_key": True,
        }
        if training:
            functorch_patch["force_non_lazy_backward_lowering"] = True
        stack = contextlib.ExitStack()
        stack.enter_context(functorch_config.patch(functorch_patch))
        # allow_empty_graphs keeps an empty graph as a compiled frame so its
        # guards reach the artifact. It also extends the lifetime of objects the
        # frame holds: with it on, a weakref callback on a value the frame
        # captured does not fire when the caller drops its reference
        # (test/dynamo/test_repros.py ReproTests.test_weakref_callback).
        try:
            stack.enter_context(torch._dynamo.config.patch(allow_empty_graphs=True))
        except BaseException:
            stack.close()
            raise
        _CAPTURE_CONFIG_STACK.set(stack)
    _CAPTURE_CONFIG_DEPTH.set(depth + 1)
    try:
        yield
    finally:
        remaining = _CAPTURE_CONFIG_DEPTH.get() - 1
        _CAPTURE_CONFIG_DEPTH.set(remaining)
        if remaining == 0:
            stack = _CAPTURE_CONFIG_STACK.get()
            _CAPTURE_CONFIG_STACK.set(None)
            if stack is not None:
                stack.close()


def _clear_package_region(
    codes: Sequence[types.CodeType], isolate_recompiles_id: int
) -> None:
    from .eval_frame import _clear_cache_entries_for_region

    for code in codes:
        _clear_cache_entries_for_region(code, isolate_recompiles_id)


class _AllowEmptyGraphsCallback(CatchErrorsWrapper):
    """The package's Dynamo callback, compiling its frames with allow_empty_graphs.

    An uncovered no-op branch must become a guarded variant rather than Dynamo's
    ordinary eager-only SkipFrame, or one fallback call permanently skips that
    frame and serving() can no longer detect it. Patched here as well as in
    _capture_config so the package's own frames get it even when the callback
    runs outside a capture-config scope.
    """

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        frame_state: dict[str, int | FrameStateSizeEntry],
    ) -> ConvertFrameReturn:
        revert = _ALLOW_EMPTY_GRAPHS()
        try:
            return super().__call__(frame, cache_entry, frame_state)
        finally:
            revert()


def _optimize_isolated(
    backend: _PrecompileBackend,
    package: CompilePackage,
    *,
    recompile_limit: int,
    dynamic: bool | None,
) -> OptimizeContext:
    from .eval_frame import OptimizeContext

    optimize_ctx = torch._dynamo.optimize(
        backend,
        package=package,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        isolate_recompiles=True,
    )
    if not isinstance(optimize_ctx, OptimizeContext):
        raise PackageError("torch.compiler.precompile requires Dynamo to be enabled")
    callback = optimize_ctx.callback
    if not isinstance(callback, CatchErrorsWrapper):
        raise AssertionError(f"expected a CatchErrorsWrapper, got {type(callback)}")
    optimize_ctx.callback = _AllowEmptyGraphsCallback(
        callback._torchdynamo_orig_backend, callback.hooks
    )
    return optimize_ctx


def _compose_with_default(
    user: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]],
) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
    """AND a caller's filter with the default rather than replacing it.

    ``default_guard_filter_fn`` is not a default in the "sensible starting point"
    sense -- it is what drops the identity guards that CANNOT be serialized at
    all. Replacing it means a caller who wanted to drop three of their own guards
    silently re-admits every unserializable one, and the failure surfaces as
    "ID_MATCH guard cannot be serialized" in frames that have nothing to do with
    their filter. A custom filter can only ever want to drop MORE, so composing
    is the only reading that makes sense.
    """

    def composed(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
        base = default_guard_filter_fn(entries)
        chosen = user(entries)
        if len(chosen) != len(entries):
            raise ValueError(
                f"guard_filter_fn returned {len(chosen)} decisions for "
                f"{len(entries)} guards; it must return one per entry."
            )
        return [bool(a) and bool(b) for a, b in zip(base, chosen)]

    return composed


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

    Keeping these makes serialization raise for essentially every function, so
    every drop is recorded with its source name in
    ``PrecompileSummary.dropped_guards``. A caller is not refused for having
    them, because requiring none would refuse essentially every model. The rail
    that is on is the risky-drop
    lint, and a lint is not a proof. See ``risky_dropped_guards``.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]


def _owning_module(value: object) -> str | None:
    if isinstance(value, types.ModuleType):
        return value.__name__
    owner = getattr(value, "__module__", None)
    return owner if isinstance(owner, str) else None


def _source_root(source: Source) -> Source:
    while isinstance(source, ChainedSource):
        source = source.base
    return source


# Locals Dynamo synthesizes when a resume function is itself nested, passed
# positionally into the continuation. They name generated code, not a slot any
# config chooses, so an identity guard lost on one cannot diverge.
_DYNAMO_SYNTHESIZED = ("__nested_resume_fns", "__nested_frame_values")


def _is_dynamo_synthesized(source_name: str) -> bool:
    return any(
        source_name == n or source_name.startswith(n + "[") for n in _DYNAMO_SYNTHESIZED
    )


# A pip target nested inside the stdlib dir that sysconfig does not name in
# this layout -- Debian's /usr/lib/python3.X/dist-packages -- still ends in one
# of these.
_INSTALL_DIR_NAMES = frozenset({"site-packages", "dist-packages"})


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


@functools.cache
def _stdlib_roots() -> tuple[str, ...]:
    """
    Where this interpreter's own library lives. os is unquestionably stdlib, so
    its directory is the direct evidence and the only one that stays right when
    the stdlib is a zip; sysconfig and sys._stdlib_dir cover a build where os is
    frozen with no __file__.
    """
    roots = []
    os_file = getattr(os, "__file__", None)
    if os_file:
        roots.append(os.path.dirname(os_file))
    frozen_dir = getattr(sys, "_stdlib_dir", None)  # 3.11+
    if frozen_dir:
        roots.append(frozen_dir)
    paths = sysconfig.get_paths()
    roots += [p for p in (paths.get("stdlib"), paths.get("platstdlib")) if p]
    return tuple(sorted({_norm(p) for p in roots}))


@functools.cache
def _install_roots() -> tuple[str, ...]:
    """
    Where a third party lands. This is the load-bearing exclusion: purelib is
    NESTED inside stdlib in a conda layout and inside platstdlib in a venv, so
    without it every pip-installed package is under a stdlib root.
    """
    paths = sysconfig.get_paths()
    roots = [p for p in (paths.get("purelib"), paths.get("platlib")) if p]
    for name in ("getsitepackages", "getusersitepackages"):
        # getattr, not a direct reference: the "old virtualenv site.py" this
        # guards against does not DEFINE these, so naming them here would raise
        # the very AttributeError the except is for -- out of a lint, aborting
        # the capture it was asked to check.
        get = getattr(site, name, None)
        try:
            got = get() if get is not None else None
        except Exception:
            continue  # -S, or a site.py that defines it but cannot answer
        if got is None:
            continue
        roots += [got] if isinstance(got, str) else list(got)
    return tuple(sorted({_norm(p) for p in roots}))


@functools.cache
def _torch_roots() -> tuple[str, ...]:
    """
    Every directory torch's own submodules come from. An editable build splits
    them -- torch/__init__.py out of the source tree, _C.so and version.py out
    of site-packages -- and torch.__path__ is exactly that set. It is only
    trusted if the directory this file is running from is in it, so a
    sys.modules['torch'] that is not us cannot nominate its own roots.
    """
    own_file = globals().get("__file__")
    if not own_file:
        return ()  # frozen torch: nothing to anchor to, fall back to the name
    own = _norm(os.path.dirname(os.path.dirname(own_file)))
    roots = {own}
    search = getattr(sys.modules.get("torch"), "__path__", None) or ()
    listed = {_norm(p) for p in search if isinstance(p, str)}
    if own in listed:
        roots |= listed
    return tuple(sorted(roots))


def _within(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == r or path.startswith(r + os.sep) for r in roots)


@functools.cache
def _classify_file(file: str, stdlib: bool) -> bool | None:
    """
    Cached on the __file__ string rather than on the module name: the roots are
    fixed for the process, so the answer for a path never changes, while the
    module a name resolves to can.
    """
    if not os.path.isabs(file):
        # Resolving it would be against a cwd that is not the one it was
        # recorded under, so it is evidence in neither direction. torch.ops
        # really does carry __file__ == "_ops.py".
        return None
    path = _norm(file)
    if not stdlib:
        return _within(path, _torch_roots())
    if _INSTALL_DIR_NAMES.intersection(path.split(os.sep)):
        return False
    return _within(path, _stdlib_roots()) and not _within(path, _install_roots())


def _located(module: types.ModuleType, name: str, stdlib: bool) -> bool | None:
    """Shipped here (True), shipped elsewhere (False), or no evidence (None)."""
    # The module dict rather than getattr: a PEP 562 module __getattr__ is user
    # code, and a module that raises on an unknown attribute would take save()
    # down from inside a lint.
    attrs = getattr(module, "__dict__", None) or {}
    file = attrs.get("__file__")
    if isinstance(file, str) and file:
        verdict = _classify_file(file, stdlib)
        if verdict is not None:
            return verdict
    spec = attrs.get("__spec__")
    origin = getattr(spec, "origin", None)
    loader = attrs.get("__loader__") or getattr(spec, "loader", None)
    if origin == "built-in" or loader is importlib.machinery.BuiltinImporter:
        # Statically linked, and BuiltinImporter precedes PathFinder on
        # sys.meta_path, so no file on sys.path is reachable under this name.
        return name.partition(".")[0] in sys.builtin_module_names
    if origin == "frozen" or loader is importlib.machinery.FrozenImporter:
        return True  # frozen also precedes the path finder
    return None  # namespace package, exec'd in memory, REPL __main__


def _is_library_module(module_name: str | None) -> bool:
    """
    Owned by torch or the stdlib, so config on the serving machine does not
    choose between implementations. NB this trusts the OWNER, not the binding:
    a third party that monkeypatches ``F.gelu`` at import time still diverges,
    and that is called out in ``_is_risky_drop``'s KNOWN GAP.

    sys.stdlib_module_names is a list of NAMES, and a waiver keyed on a name is
    a collision away from being wrong: graphlib, queue, code and distutils are
    all stdlib names a third party can and does supply. Worse, the name can be
    right and the code still not be the stdlib's -- in a default setuptools
    install ``import distutils`` gets site-packages/setuptools/_distutils, and
    SETUPTOOLS_USE_DISTUTILS picks which one, which is exactly the
    config-chooses-the-implementation shape this lint exists to catch. So the
    module has to RESOLVE to code shipped with the interpreter: located under a
    stdlib root and not under an install root (purelib nests inside stdlib in
    conda and inside platstdlib in a venv, so the exclusion is what does the
    work), or with no file at all because it is built in or frozen, which the
    path finder cannot shadow. A name that is not imported, a namespace
    package, and a module with no location evidence are all untrusted.
    """
    if module_name is None:
        return False
    top = module_name.partition(".")[0]
    if top == "torch":
        if not _torch_roots():
            return True
        stdlib = False
    elif top in sys.stdlib_module_names:
        stdlib = True
    else:
        return False
    root = sys.modules.get(top)
    if root is None or _located(root, top, stdlib) is not True:
        return False
    parts = module_name.split(".")
    for i in range(2, len(parts) + 1):
        name = ".".join(parts[:i])
        module = sys.modules.get(name)
        # An unimported inner name has nothing to check, and the package it
        # would have to be found in has already been located.
        if module is not None and _located(module, name, stdlib) is False:
            return False
    return True


def _defined_where_read(
    value: object, user_stack: traceback.StackSummary | None
) -> bool:
    """
    Whether the def lives in the file of the frame that read it.

    A def bound to its own name in its OWN module takes an edit there to
    repoint. ``from impl_a import op`` takes only a conditional import in the
    reader, which is not an edit at all and which no checksum covers.
    """
    home = sys.modules.get(getattr(value, "__module__", None) or "")
    file = getattr(home, "__file__", None)
    return bool(user_stack) and file is not None and file == user_stack[-1].filename


def _dynamo_alias_module(global_name: str) -> types.ModuleType | None:
    """The module behind an ``__import_a_dot_b`` alias, mirroring import_source."""
    prefix = "__import_"
    if not global_name.startswith(prefix):
        return None
    return sys.modules.get(global_name[len(prefix) :].replace("_dot_", "."))


def _module_namespaces(
    entries: Sequence[GuardFilterEntry],
) -> dict[str, types.ModuleType]:
    """
    Sources holding a module whose binding config cannot repoint, mapped to the
    module itself. Dynamo guards every module it walks through, so the path
    down to ``F.gelu`` is guarded module by module, which is what lets an
    attribute read be recognised as coming off a namespace rather than off an
    object a config could have swapped. The module comes back with the name
    because whether a read off a namespace is safe depends on which module it
    is -- see ``_is_risky_drop``.

    TRUSTED is the load-bearing half and is deliberately narrow. A module is
    that if torch or the stdlib owns it, if it is bound under its own name --
    ``import mypkg.layers``, and the ``__import_x`` alias Dynamo installs to
    reach an inlined function's own globals -- or if it is an attribute of a
    trusted module under a name that module already owns: its own ``__name__``
    (a plain ``import own_sub`` inside the parent) or the parent's plus the
    attribute (``from . import sub``). An ALIASED user module is none of those:
    ``if flag: import impl_b as impl`` picks what ``impl.op`` resolves to per
    machine, and so does the same alias spelled ``from . import impl_b as
    impl`` in a package __init__. Inheriting the parent's trust without
    checking the name is what let ``mypkg.impl.op`` through before.
    """
    modules = {
        e.orig_guard.originating_source.name: (e.orig_guard.originating_source, e.value)
        for e in entries
        if isinstance(e.value, types.ModuleType)
        and isinstance(_source_root(e.orig_guard.originating_source), GlobalSource)
    }
    # Dynamo guards the attributes it reads off an import alias but never the
    # bare alias, so a real model produces G['__import_torch'].Tensor with no
    # module-valued entry for G['__import_torch'] to anchor it. The alias name
    # encodes its module, so recover it rather than treating torch.Tensor as a
    # config-swappable slot.
    for e in entries:
        root = _source_root(e.orig_guard.originating_source)
        if isinstance(root, GlobalSource) and root.name not in modules:
            aliased = _dynamo_alias_module(root.global_name)
            if aliased is not None:
                modules[root.name] = (root, aliased)
    trusted: dict[str, bool] = {}

    def is_trusted(name: str) -> bool:
        if name not in trusted:
            trusted[name] = False  # also breaks cycles while recursing
            found = modules.get(name)
            if found is not None:
                source, module = found
                # Mirrors InstructionTranslator.import_source's alias.
                dynamo_alias = "__import_" + module.__name__.replace(".", "_dot_")
                if _is_library_module(module.__name__):
                    trusted[name] = True
                elif isinstance(source, GlobalSource):
                    trusted[name] = source.global_name in (
                        module.__name__,
                        dynamo_alias,
                    )
                elif isinstance(source, AttrSource):
                    outer = modules.get(source.base.name)
                    trusted[name] = (
                        outer is not None
                        and is_trusted(source.base.name)
                        and module.__name__
                        in (source.member, f"{outer[1].__name__}.{source.member}")
                    )
        return trusted[name]

    return {name: module for name, (_, module) in modules.items() if is_trusted(name)}


# Dynamo's own handle on the builtins dict, minted by
# OutputGraph.install_builtins_dict_in_fglobals.
_BUILTINS_DICT_PREFIX = "__builtins_dict__"


def _reads_a_builtin(source: Source, value: object) -> bool:
    """
    ``len`` or ``sorted`` reached the ordinary way, through the builtins dict
    Dynamo installs to resolve them. No binding sits in front of those, so
    nothing can repoint them.

    A builtin parked in a slot -- ``self.act = abs``, straight out of an
    ACT2FN-style table -- is a slot like any other, so this deliberately keys
    on where the read comes FROM rather than on who owns the value.
    """
    return (
        isinstance(source, DictGetItemSource)
        and isinstance(source.base, GlobalSource)
        and source.base.global_name.startswith(_BUILTINS_DICT_PREFIX)
        and _owning_module(value) == "builtins"
    )


def _is_risky_drop(
    entry: GuardFilterEntry, namespaces: dict[str, types.ModuleType]
) -> bool:
    """
    Whether losing this identity guard can plausibly change results.

    Intersect the binding SITE with who owns the value; either test alone is
    wrong. Site alone: ``self.act = getattr(F, cfg.activation)`` and
    ``self.act = cfg.act_fn`` are the same swappable slot, so calling the first
    benign because ``F.gelu`` is torch-owned waves through the exact divergence
    this check exists for -- capture gelu, serve silu, get the gelu graph and no
    error. Ownership alone: ``if flag: import impl_b as impl`` then ``impl.op``
    is a read off a module namespace exactly like ``F.gelu``, and the module is
    user code an env var chose; so is ``self.act = abs``, where the value is a
    builtin nothing can repoint but the attribute holding it is a slot. The site
    survives the name stripping that makes source spelling useless -- a guard on
    ``self.act`` arrives as ``'self.act'`` -- because the structured source is
    still on the guard.

    For a capture-here / serve-there deployment the concern is not in-process
    rebinding but DIVERGENCE: the serving machine runs the same source but picks
    a different object because config, a flag, or an env var differs. Three
    bindings are waived -- a builtin read the ordinary way (see
    ``_reads_a_builtin``), a read off a TRUSTED namespace (see
    ``_module_namespaces``) that torch or the stdlib owns or that owns the
    value itself, and a global bound to a def of that same name when torch or
    the stdlib owns the def or it lives in the file doing the reading. The rest
    are slots whose occupant config chooses: instance attributes, closure
    cells, aliased imports, cross-module ``from x import op``, registry
    lookups.

    Trusting a namespace is not trusting everything read off it. ``F.gelu`` is
    waived because torch owns torch.nn.functional and there is only one of it;
    ``own_helpers.call`` is waived because own_helpers owns the def, subject to
    the gap below. ``mypkg.op`` re-exported from ``mypkg.impl_b``,
    ``dispatch.op`` and ``mypkg.impl.op`` are not waived: the import that chose
    the implementation lives in a file the inlined-source checksum never sees,
    so capture and serve can disagree with every other rail passing. Waiving
    those is how this predicate failed open in an earlier round;
    ``_RISKY_DROP_CORPUS`` in test_precompile_package.py is the regression net that keeps
    them, and every other shape found so far, flagged.

    KNOWN GAP, and it is a wrong-answer one. EVERY waiver above judges the
    object capture happened to bind, not the statement that bound it, so any
    name bound CONDITIONALLY is waived whenever the branch taken on the capture
    machine is one of the waived shapes. This is not specific to the def-name
    arm and it is not limited to the file being read:

    - def-name arm. ``if HAVE_FAST: from fastops import gelu`` / ``else: from
      torch.nn.functional import gelu``, captured without the flag, drops
      ``G['gelu']`` and reports nothing; a serving machine that has fastops
      runs torch's gelu instead, with no error. A def the reading file itself
      redefines under an ``if`` is the same shape.
    - namespace-owns-the-value arm. A module that binds a name under an ``if``
      and is read as a namespace by an ordinary ``import mod`` elsewhere is
      also waived, because at capture time the module really does own whichever
      def the branch produced. The reading file binds nothing conditionally,
      so it looks covered and is not.
    - library-namespace arm, and this one needs no conditional at all. The
      waiver trusts the owner, not the binding, so a third party that rebinds a
      torch or stdlib attribute -- ``F.gelu = _fast_gelu`` executed at import
      by a package that happens to be installed on the serving host -- is
      waived even though the model itself reads ``F.gelu`` unconditionally.
      ``functools.wraps(F.gelu)(user_fn)`` reaches the same waiver by a
      different route, since it copies ``__name__`` and ``__module__`` off the
      torch function it wraps.

    An ``allow_in_graph`` function passes too, and Dynamo traces it opaquely so
    the inlined-source checksum never covers it either. Nothing at capture time
    distinguishes a conditional bind from an unconditional one -- only the
    resulting object is visible -- so this is a limit of the approach rather
    than a missing check. ``dropped_guards`` is the authoritative list; this
    predicate is a lint over it, not a proof of safety.
    """
    source = entry.orig_guard.originating_source
    value = entry.value
    # entry.name, not source.name: guard names arrive with local scope stripped
    # ("L['x'].y" -> "x.y"), and these are always locals of a resume frame.
    if _is_dynamo_synthesized(entry.name):
        return False
    if source.name in namespaces:
        return False
    if _reads_a_builtin(source, value):
        return False
    if isinstance(source, ChainedSource):
        namespace = namespaces.get(source.base.name)
        if namespace is not None:
            return not (
                _is_library_module(namespace.__name__)
                or _owning_module(value) == namespace.__name__
            )
    if (
        isinstance(source, GlobalSource)
        and getattr(value, "__name__", None) == source.global_name
    ):
        return not (
            _is_library_module(_owning_module(value))
            or _defined_where_read(value, entry.orig_guard.user_stack)
        )
    return True


# CONSTANT_MATCH covers bool/None/int, EQUALS_MATCH everything else comparable.
_VALUE_EQUALITY_GUARD_TYPES = frozenset({"CONSTANT_MATCH", "EQUALS_MATCH"})


def _pins_a_value(guard_type: str, name: str) -> bool:
    """
    Whether this kept guard makes the artifact serve only the value it saw.

    Two things have to line up, and keying on either one alone is wrong.

    The guard has to be a value-equality one. TENSOR_MATCH, SHAPE_ENV and the
    global-state guards are what every capture has and they generalize fine --
    a TENSOR that crosses a graph break gets TENSOR_MATCH on a ``___stackN``
    source and is emphatically not a pin.

    And the source has to be a BARE name -- a plain local of some traced frame,
    or the ``___stackN`` Dynamo gives a value crossing a graph break. Anything
    dotted or subscripted (``self.eps``, ``model._modules['ln'].eps``,
    ``G['CFG'].width``) is reached THROUGH an argument rather than being one,
    which is where model config lives: every LayerNorm and Dropout contributes
    a CONSTANT_MATCH there, so counting those would flag every model and make
    the field noise.

    KNOWN GAP: a constant inside a container argument is guarded on a
    subscripted source (``dims[0]`` for ``x.sum(dim=[0])``) and is not counted.
    ``kept_guards`` is the authoritative list; this is a lint over it.
    """
    return guard_type in _VALUE_EQUALITY_GUARD_TYPES and not any(
        c in name for c in ".["
    )


# Object addresses differ every run, so they are scrubbed from rendered guard
# facts. Keep these anchored to the call shapes that carry addresses: a bare
# \b\d{9,}\b also eats a user constant (a dict key, a slice bound), so two
# variants guarding different values render the same fact and invent an
# invariant neither holds.
_OBJ_ID = re.compile(r"(?<=, )\d+(?=\), type=)")
_SAVED_HOOK_IDS = re.compile(r"(?<=top_saved_tensors_hooks ids == )\(\d+(?:, \d+)*\)")
_DYNAMO_INDICES = re.compile(r"_dynamo_\w*indices")
# Dynamo appends a per-process counter to the globals it installs, so the same
# guard reads __builtins_dict___6 in one compilation and ___8 in the next.
# Leaving that in makes identical guards look like they differ.
_DYNAMO_COUNTER = re.compile(
    r"(__builtins_dict__|__compiled_fn|__resume_at)_*\d+(_\d+)?"
)
# OutputGraph.install_global_by_id names a global "<prefix>_<id(value)>_c<n>",
# so a guard reading one carries BOTH an address and a compile counter inside
# an identifier, where neither pattern above can see it. Real models reach this
# -- transformers' Qwen2 installs three -- and the report then differs run to
# run, which is exactly what the "commit and diff" contract rules out.
_DYNAMO_GLOBAL_BY_ID = re.compile(r"_\d{9,}_c\d+\b")


def _normalize(text: str) -> str:
    text = _SAVED_HOOK_IDS.sub("(<ids>)", text)  # see _saved_hooks_fingerprint
    text = _DYNAMO_GLOBAL_BY_ID.sub("_<id>_c<n>", _OBJ_ID.sub("<id>", text))
    return _DYNAMO_COUNTER.sub(r"\1_<n>", text)


def _render_code(code_list: Sequence[str] | None) -> tuple[str, ...]:
    # Keep the _dynamo_*_indices parts: they carry TENSOR_MATCH's dimension
    # marking, so mark_static on one variant and not the next shows up only here.
    return tuple(_normalize(part) for part in (code_list or ()))


class _PrecompileBackend:
    """Give one explicit session its own Dynamo cache identity."""

    def __init__(
        self, backend: str, keep_graphs: bool = False, serving: bool = False
    ) -> None:
        inner = torch._dynamo.lookup_backend(backend)
        self._torchdynamo_orig_backend = inner
        self._torchdynamo_cache_key = object()
        # Process-wide and never cleared: get_backend only looks for the key
        # once told one exists, since the miss costs every torch.compile user.
        torch._C._dynamo.eval_frame._enable_precompile_cache_keys()
        self.backend_ctx_ctor = getattr(
            inner, "backend_ctx_ctor", contextlib.nullcontext
        )
        # Rendering a subgraph as source needs the graph, which only exists
        # here. Kept only where something will render it (see the caller):
        # retaining deepcopies every compiled graph for the session.
        self._keep_graphs = keep_graphs
        self.graphs: dict[str, tuple[torch.fx.GraphModule, list[Any]]] = {}
        # Serving an INSTALLED artifact answers a guard miss by compiling,
        # because a frame reachable only through the frame evaluator has no
        # other way to run. Counted, and said out loud once per graph: an
        # artifact that quietly compiles more of itself on every batch looks
        # exactly like one that is serving.
        self.serving = serving
        self.serve_time_compiles = 0

    def get_compiler_config(self) -> Any:
        getter = getattr(self._torchdynamo_orig_backend, "get_compiler_config", None)
        return None if getter is None else getter()


# Guards whose check IS object identity, directly or through a derived guard,
# which is the same test default_guard_filter_fn drops on.
_IDENTITY_GUARD_TYPES = frozenset(
    CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
)


_STABLE_CONST_TYPES = (str, int, float, complex, bytes, bool, type(None))


def _stable_consts(consts: tuple[object, ...]) -> tuple[object, ...]:
    """
    co_consts reduced to the part that reprs the same in every process.

    A nested code object reprs with its ADDRESS, so it cannot go into a digest
    that ends up in a file meant to be committed and diffed. Containers are
    filtered recursively rather than dropped whole: two lambdas differing only
    in a tuple or frozenset constant -- ``x * (1, 2)`` against ``x * (1, 3)`` --
    are genuinely different variants, and dropping the container is what let
    them collide.
    """
    out: list[object] = []
    for c in consts:
        if isinstance(c, _STABLE_CONST_TYPES):
            out.append(c)
        elif isinstance(c, types.CodeType):
            # A nested code object reprs with its ADDRESS, so it cannot go in
            # verbatim -- but dropping it merges two lambdas that differ only in
            # a comprehension or an inner lambda, which is this same bug one
            # level down. Recurse into its own fingerprint instead.
            out.append(_code_fingerprint(c))
        elif isinstance(c, tuple):
            out.append(_stable_consts(c))
        elif isinstance(c, frozenset):
            # Sorted by repr so the digest does not inherit set iteration order.
            out.append(tuple(sorted(_stable_consts(tuple(c)), key=repr)))
    return tuple(out)


def _code_fingerprint(code: types.CodeType) -> str:
    """
    Name a code object by its body, for callables a definition site cannot tell
    apart -- an ACT2FN table written on one source line makes every lambda in it
    agree on file AND lineno.

    Everything hashed is derived from the source, so the digest is identical in
    another process. It is NOT stable across Python versions, since co_code is
    version-specific bytecode: a committed invariants file churns wholesale on
    an interpreter upgrade even with unchanged source.
    """
    return _hash_text(
        repr(
            (
                code.co_code,
                code.co_names,
                code.co_varnames,
                # LOAD_DEREF addresses a cell by INDEX, so two closures that
                # capture different variables have identical co_code and are
                # told apart only by the names they close over.
                code.co_freevars,
                code.co_cellvars,
                _stable_consts(code.co_consts),
            )
        )
    )


def _object_identity(value: object) -> str:
    """
    A stable stand-in for the id ``_normalize`` stripped.

    A name rather than an address, so the file still diffs clean across runs.
    Two objects of one type still collapse -- two Linear instances are
    indistinguishable here -- but the case that matters must separate: one slot
    holding a different callable in different variants. A qualname alone does
    NOT achieve that, because the shape this exists for is an ACT2FN-style
    table, whose entries are all ``<lambda>`` in one module. Two of those then
    render identically, the CLOSURE_MATCH that split the compilations lands in
    the intersection, and the report calls the one thing that varies an
    invariant of both. So a callable is also named by where it is DEFINED and by
    a digest of its body, both source-derived and so still stable across
    processes; the file is reduced to its basename so the report does not carry
    a checkout path.

    The discriminating part goes FIRST. Truncation is what bounds this string,
    and a qualname alone can exceed the limit on real models -- a transformers
    lambda nested in a long module path -- so a digest appended at the end is
    cut off exactly on the names that need it most, re-colliding what it was
    added to separate.
    """
    if isinstance(value, types.ModuleType):
        return f"is module {value.__name__}"
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(name, str):
        code = getattr(value, "__code__", None)
        where = ""
        if isinstance(code, types.CodeType):
            site = os.path.basename(code.co_filename or "?")
            where = f"@{site}:{code.co_firstlineno}#{_code_fingerprint(code)} "
        return _normalize(f"is {where}{_owning_module(value) or '?'}.{name}")[:160]
    return f"is a {type(value).__module__}.{type(value).__qualname__}"[:160]


# Guards that pin an input's SHAPE or VALUE, and are never policy-dropped even
# when they held identically across every captured variant. Dropping a guard is
# licensed by "it discriminated nothing", but with a single example nothing CAN
# discriminate, and what silently disappears is the check that the runtime tensor
# looks like the captured one at all -- so an out-of-domain shape reaches a kernel
# specialized for a different one, which crashes on inductor and can quietly
# miscompute on eager. Shape is the axis a caller is most likely to vary and least
# likely to expect to be unchecked, so it is always serialized.
_SHAPE_BEARING_GUARD_TYPES = frozenset(
    {
        "TENSOR_MATCH",
        "SEQUENCE_LENGTH",
        # Value-equality guards belong here for the same reason and are the
        # half that bites hardest: they pin a Python value the graph
        # specialized on -- an int or bool argument, `module.training`, an
        # `.item()` result, `mask=None`. Dropped, the artifact serves the
        # captured branch for every other value, with correct-looking numerics
        # and nothing in the header to say so. Shapes at least crash inside a
        # kernel; these do not.
        "CONSTANT_MATCH",
        "EQUALS_MATCH",
        "DUPLICATE_INPUT",
        # And the one that pins whether an attribute is THERE. hasattr is a
        # branch like any other, so dropping it serves the captured side to a
        # caller on the other one -- the same silent wrong answer as a dropped
        # CONSTANT_MATCH. Reachable on the DEFAULT gates, because a
        # single-variant capture makes every slot look invariant and the drop
        # is not classed risky.
        "HASATTR",
        # And the guard that pins an input's KIND. Dropped, a graph traced for
        # one class is served to another and returns the first one's answer,
        # silently -- there is no shape to crash on. Upstream depends on this
        # specifically: an AsyncCollectiveTensor's tensor-class guards are
        # deliberately removed so an ACT-traced graph can be reused for the
        # resolved tensor, and the observation sites reinstall exactly this
        # guard to keep that sound. FAKE_SCRIPT_TYPE_MATCH is the same pin for
        # a reference-type opaque object (type(unwrapped) is T).
        "TYPE_MATCH",
        "FAKE_SCRIPT_TYPE_MATCH",
        # And the default-device pin: the graph specialized on
        # utils_device.CURRENT_DEVICE, so a capture under the default None
        # served under torch.set_default_device("cuda") returns CPU tensors
        # with no refusal.
        "DEFAULT_DEVICE",
        # And every guard that pins a Python fact about a value or a container's
        # contents: whether a key is in a dict or a set, which keys a dict has,
        # whether an attribute is absent from an instance __dict__, how long a
        # tuple iterator is, where a range/count iterator stands, whether a
        # value is None or a given bool. Each one is a branch the graph
        # specialized on, so a drop serves the captured branch to the other
        # side, silently -- and a module-owned dict (self.opts = {}) is
        # environment-rooted, which is exactly where the policy used to drop it.
        "BOOL_MATCH",
        "CONSTANT_SUBCLASS_MATCH",
        "COUNT_ITERATOR_MATCH",
        "DICT_CONTAINS",
        "DICT_KEYS_MATCH",
        "DICT_NOT_CONTAINS",
        "MAPPING_KEYS_CHECK",
        "NONE_MATCH",
        "NOT_NONE_MATCH",
        "NOT_PRESENT_IN_GENERIC_DICT",
        "RANGE_ITERATOR_MATCH",
        "SET_CONTAINS",
        "SET_NOT_CONTAINS",
        "TUPLE_ITERATOR_LEN",
    }
)


# Guards whose C++ leaf compares something no fingerprint here models: subclass
# metadata, a DTensor placement, an opaque object's guard values, a raw
# DispatchKeySet, or process-wide state carried entirely in the leaf. Calling
# two of these equal is how the report ends up asserting a precondition that
# does not hold, so they are never compared -- they are reported separately as
# undetermined, which is the honest answer and cannot mislead.
_UNMODELLED_GUARD_TYPES = frozenset(
    {
        "DISPATCH_KEY_SET_MATCH",
        "DTENSOR_SPEC_MATCH",
        # Its builder is a no-op like GRAD_MODE's, but GlobalStateGuard does not
        # snapshot FSDP training state and the state is per param group, so
        # nothing here can model or vouch for it.
        "FSDP_TRAINING_STATE",
        "GLOBAL_STATE",
        "OPAQUE_OBJ_GUARD_FN_MATCH",
        "SHAPE_ENV",
        "TENSOR_SUBCLASS_METADATA_MATCH",
        "TORCH_FUNCTION_STATE",
    }
)


# Guard types whose GuardBuilder method is `pass`: the guard is a marker, and
# the check it names is made by GLOBAL_STATE's leaf. Nothing about them is
# serialized or dropped, so they never appear in a dropped-guard report --
# listing GRAD_MODE as "a precondition nothing checks" would be false, since
# GlobalStateGuard checks it on every call. Their facts ARE compared, from the
# same process state GlobalStateGuard snapshots (see _value_fingerprint).
_NOOP_GUARD_TYPES = frozenset({"DETERMINISTIC_ALGORITHMS", "GRAD_MODE"})


def _is_noop_guard_type(guard_type: str) -> bool:
    # EMPTY_NN_MODULE_HOOKS_DICT is a no-op by config: under
    # skip_nnmodule_hook_guards, the default, GuardBuilder emits nothing for it.
    return guard_type in _NOOP_GUARD_TYPES or (
        guard_type == "EMPTY_NN_MODULE_HOOKS_DICT"
        and torch._dynamo.config.skip_nnmodule_hook_guards
    )


# The ONLY guard types the invariance policy may drop, and only when proven
# invariant across every captured variant: identity guards (which the default
# filter drops anyway, as unserializable) and process-wide compiler state. The
# four sets form a total, disjoint classification of GuardBuilder's
# guard-producing methods, pinned by
# test_precompile_package.test_guard_policy_classification_is_total: a guard type in
# none of them -- i.e. any type added to GuardBuilder after this list -- is
# KEPT unconditionally until someone classifies it here, so a new value-pinning
# guard can never become silently droppable by default.
_INVARIANT_DROPPABLE_GUARD_TYPES = frozenset(
    {
        "AUTOGRAD_SAVED_TENSORS_HOOKS",
        "BUILTIN_MATCH",
        "CLASS_MATCH",
        "CLOSURE_MATCH",
        # COW_TENSOR_MATCH pins a folded torch._C._is_cow_tensor branch, so by
        # value it reads like the shape-bearing membership guards -- but it
        # cannot be rebuilt from the pickle: the tensor comes back fake and its
        # builder rejects that, so the default filter keeps it and then the
        # policy's re-serialization, or a load, fails on it rather than losing
        # it silently. It lives here rather than in the shape-bearing set
        # because keeping it never yields a loadable artifact.
        "COW_TENSOR_MATCH",
        "DICT_VERSION",
        "DUAL_LEVEL",
        "EMPTY_NN_MODULE_HOOKS_DICT",
        "FUNCTION_MATCH",
        "FUNCTORCH_STACK_MATCH",
        "ID_MATCH",
        "MODULE_MATCH",
        "NN_MODULE",
        "WEAKREF_ALIVE",
    }
)


def _saved_hooks_fingerprint() -> str:
    """Name the installed saved-tensors hooks by content, never by address."""
    try:
        from torch._functorch._aot_autograd.utils import top_saved_tensors_hooks

        hooks = top_saved_tensors_hooks()
    except Exception:
        return ""
    if not hooks:
        return "hooks=None"
    names = []
    for hook in hooks:
        code = getattr(hook, "code", None)  # fx GraphModule renders its graph
        if isinstance(code, str):
            names.append(_hash_text(code))
        else:
            names.append(_object_identity(hook))
    return "hooks=(" + ", ".join(names) + ")"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _value_fingerprint(entry: GuardFilterEntry) -> str:
    """
    What the guard checks, when the rendered code does not say.

    TENSOR_MATCH is the case that matters: its code_list carries only the
    _dynamo_*_indices hasattr checks, while everything it really compares lives
    in the C++ leaf. Without those two specializations of one frame look
    identical and wrongly land in the intersection, so this mirrors TensorCheck
    -- python type and the full dispatch key set included, since a Parameter
    against a Tensor, a conjugated view against a plain one, or an
    inference-mode tensor against a no_grad one, splits a compilation exactly
    as dtype does. KNOWN GAP: that leaf checks
    nothing for a dim the compile made dynamic, so under ``dynamic=True`` the
    concrete shape here is narrower than the guard and a shape-generic
    TENSOR_MATCH is reported as varying rather than invariant.

    An identity guard needs one too, because ``_normalize`` strips the id its
    code renders: without a name for the object, two variants holding different
    callables at one source collapse into one fact and are reported as an
    invariant neither of them holds.

    Every other guard takes its value from its own rendered code, which names
    it, so fingerprinting it again SPLITS identical guards: TYPE_MATCH on an
    unspecialized int checks only that the int is an int, and stamping 1 on one
    variant and 2 on the next demotes a real invariant into two identical
    'varies' lines.
    """
    if entry.guard_type == "AUTOGRAD_SAVED_TENSORS_HOOKS":
        # Its code renders tuple(map(id, hooks)), which _normalize has to erase
        # or the file churns -- but erasing it alone would merge two variants
        # that differ ONLY in their hooks and report the guard that split them
        # as an invariant. Put back a discriminator derived from what the hooks
        # ARE rather than where they live, which is both stable across
        # processes and still telling.
        return _saved_hooks_fingerprint()
    if entry.guard_type == "GRAD_MODE":
        # Global-state guards carry no name, code or value, so two variants of
        # one frame that differ only in grad mode render identically and land
        # in the intersection. The filter runs in the traced frame's own state.
        return f"grad_enabled={torch.is_grad_enabled()}"
    if entry.guard_type == "DETERMINISTIC_ALGORITHMS":
        # Same as GRAD_MODE: the two fields GlobalStateGuard snapshots for it.
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        return f"deterministic={torch.are_deterministic_algorithms_enabled()}, warn_only={warn_only}"
    if not entry.has_value:
        return ""
    value = entry.value
    if isinstance(value, torch.Tensor):
        # Render exactly what TensorCheck stores (notably the TLS-adjusted
        # dispatch key set, not the tensor's own) rather than reconstructing it.
        from .guards import convert_to_concrete_values, get_tensor_guard_code_part

        try:
            return get_tensor_guard_code_part(
                value,
                "",
                convert_to_concrete_values(value.size()),
                convert_to_concrete_values(value.stride()),
                type(value),
                torch._C._dispatch_keys(value),
            )
        except Exception:
            return f"type={type(value).__name__}, dtype={value.dtype}, <unrenderable>"
    if entry.guard_type in _IDENTITY_GUARD_TYPES or any(
        d in _IDENTITY_GUARD_TYPES for d in entry.derived_guard_types
    ):
        return _object_identity(value)
    return ""


def _fact_order(fact: _GuardFact) -> tuple[str, str, str, str]:
    # value is part of the key: once the boilerplate code parts are filtered a
    # TENSOR_MATCH renders no code, so two shape specializations would otherwise
    # tie and sort unstably, making the file differ run to run.
    return (fact.source, fact.guard_type, " ".join(fact.code), fact.value)


def _wont_generalize(
    kept: set[tuple[str, str]],
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
) -> tuple[str, ...]:
    """Sources no captured variant will serve a new value for.

    A source pinned in ONE variant is not pinned for the artifact: the union of
    kept guards says "some graph equality-matched this", while what the warning
    claims is "no graph will take anything else". A frame whose other variant
    guards the same source generically -- the ordinary shape once two examples
    are captured -- serves the new value fine, and warning about it tells the
    caller to enumerate values that already work.
    """
    pinned = {n for t, n in kept if _pins_a_value(t, n)}
    if not pinned:
        return ()
    for variants in guard_sets.values():
        for facts in variants:
            mentioned = {f.source for f in facts}
            pins_here = {
                f.source for f in facts if _pins_a_value(f.guard_type, f.source)
            }
            # This variant reached the source without pinning it, so it is the
            # graph that serves other values.
            pinned -= mentioned - pins_here
    return tuple(sorted(pinned))


# Healing re-serializes, which can in principle prune something new. Bounded
# rather than open: in practice one pass is always enough.
_VALIDATION_PASSES = 4


def varying_guard_slots(
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
) -> frozenset[tuple[str, str]]:
    """The guard slots that actually discriminate between captured variants.

    A slot is ``(guard_type, normalized source)``. It varies when two variants
    of one frame recorded DIFFERENT facts for it, and also when it is present in
    some variants and absent in others -- a guard only one variant carries is
    what tells that variant apart, and comparing values alone would call it
    invariant and drop it. That present-in-some case is the majority of what is
    kept, not an edge.

    Everything else held identically in every variant, which is what licenses a
    caller to leave it out of the serialized copy.
    """
    varying: set[tuple[str, str]] = set()
    for variants in guard_sets.values():
        seen: dict[tuple[str, str], set[tuple[tuple[str, ...], str]]] = {}
        present: collections.Counter[tuple[str, str]] = collections.Counter()
        for facts in variants:
            for slot in {(f.guard_type, f.source) for f in facts}:
                present[slot] += 1
            for f in facts:
                seen.setdefault((f.guard_type, f.source), set()).add((f.code, f.value))
        for slot, rendered in seen.items():
            if len(rendered) > 1 or present[slot] != len(variants):
                varying.add(slot)
    return frozenset(varying)


def _summarize(
    entry: _DynamoCacheEntry,
    dropped: set[tuple[str, str]],
    kept: set[tuple[str, str]],
    policy_dropped: set[tuple[str, str]],
    risky: set[tuple[str, str]],
    truncated: frozenset[str],
    uncovered: frozenset[str],
    capture_errors: Sequence[str],
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
    dropped_code: Mapping[tuple[str, str], str],
) -> PrecompileSummary:
    wont_generalize = _wont_generalize(kept, guard_sets)
    return PrecompileSummary(
        frames=len(entry.codes),
        resume_functions=sum(1 for c in entry.codes if c.install_to_global),
        guarded_codes=sum(len(c.guarded_codes) for c in entry.codes),
        backend_graphs=len(entry.backend_ids),
        bypassed=tuple(c.python_code.co_name for c in entry.codes if c.bypassed),
        truncated=tuple(sorted(truncated)),
        uncovered_frames=tuple(sorted(uncovered)),
        wont_generalize=wont_generalize,
        dropped_guards=tuple(sorted(dropped)),
        dropped_guard_code=tuple(
            (gtype, name, dropped_code[(gtype, name)])
            for gtype, name in sorted(dropped | policy_dropped | risky)
            if (gtype, name) in dropped_code
        ),
        kept_guards=tuple(sorted(kept)),
        risky_dropped_guards=tuple(sorted(risky)),
        policy_dropped_guards=tuple(sorted(policy_dropped)),
        capture_errors=tuple(capture_errors),
    )
