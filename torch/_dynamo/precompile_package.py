"""
Ahead-of-time precompilation of a callable into MANY graphs.

``torch.compile(fn, fullgraph=True).aot_compile(...)`` produces exactly one
graph and rejects graph breaks. This module is the multi-graph counterpart: it
captures every frame Dynamo produces while exercising the supplied calls -- the entry frame, each
``torch_dynamo_resume_in_*`` continuation created by a graph break, and every
recompiled variant of each -- into a single serializable artifact.

Usage::

    python_code, cache = torch.compiler.precompile(
        step,  # e.g. lambda model, x: model(x)
        backend="inductor",
        example_inputs=[(model, x1), (model, x2)],
    )

    # later, in a fresh process
    compiled = torch.compiler.precompile.load(python_code, cache)
    with compiled, torch.no_grad():
        compiled(model, x1)

The examples ARE the capture: each one is run for you and every frame, break
continuation and guarded variant it exercises is recorded. There is no manual
capture block -- driving the calls yourself is not part of the public surface.

Calls run under ``torch.no_grad()``. ``training=True`` runs them with grad
enabled and lowers the backward eagerly instead, so the artifact carries one
and a served output can be backpropagated. No loss is needed for that: the
joint trace synthesizes tangents from the forward outputs' own metadata.

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

* An inference artifact is the default: examples run under ``torch.no_grad()``.
  For a training artifact pass ``training=True``, which traces with grad on and
  lowers the backward eagerly -- without it, AOTAutograd defers the backward to
  the first ``.backward()`` call, so a grad-enabled capture that never makes one
  records no backends and cannot be written.
* A non-tensor argument, and any value that crosses a graph break, is guarded
  by equality, so an int/bool/str argument or a break coming from ``.item()``
  yields an artifact that only serves calls reproducing those exact values.
  ``summary().wont_generalize`` lists them; exercise every value you need to
  serve through ``example_inputs``, or expect poor coverage on new data.
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

The public surface is ``torch.compiler.precompile`` -- with ``example_inputs``
for this multi-graph form -- and ``torch.compiler.precompile.load``. The
helpers in this module, including the capture session, implement that surface
and remain internal. The positional ``torch.compiler.precompile`` form
produces a self-contained Python source artifact from one example call. Both are distinct from
``torch._dynamo.config.caching_precompile``, which caches ``torch.compile``
artifacts transparently without an explicit capture block.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import hashlib
import importlib.machinery
import logging
import os
import pickle
import re
import site
import sys
import sysconfig
import threading
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch._functorch.config as functorch_config
from torch._guards import ChainedSource, Source
from torch.compiler._precompile_types import (
    ExampleInput,
    FrameInvariants,
    GuardFact as _GuardFact,
    PrecompileSummary,
)
from torch.utils._pytree import tree_leaves

from .exc import PackageError
from .guards import CheckFunctionManager
from .package import (
    _BackendId,
    _defining_module_name,
    _DynamoCacheEntry,
    CompilePackage,
    DynamoStore,
    PrecompileCacheEntry,
)
from .pgo import _use_code_state
from .source import AttrSource, DictGetItemSource, GlobalSource


if TYPE_CHECKING:
    import traceback

    from .types import GuardFilterEntry


log = logging.getLogger(__name__)

# Built once: the served call path enters this on every call, and
# config.patch() allocates a class and a ContextVar each time it is called.
_ALLOW_EMPTY_GRAPHS = torch._dynamo.config._make_closure_patcher(
    allow_empty_graphs=True
)


_CAPTURE_CONFIG_STATE: ContextVar[tuple[int, tuple[bool, bool, bool] | None]] = (
    ContextVar("precompile_capture_config_state", default=(0, None))
)

# Not a public surface -- see the module docstring. This exists so `from ...
# import *` in a debugging session pulls the entry points rather than every
# private helper, and so linters do not flag them as unused.
__all__ = [
    "precompile_load",
    "ExampleInput",
    "FrameInvariants",
    "PrecompileSession",
    "PrecompileSummary",
    "PrecompiledCallable",
    "precompile_capture",
    "serving",
]


@contextlib.contextmanager
def _capture_config(training: bool = False) -> Iterator[None]:
    depth, prior = _CAPTURE_CONFIG_STATE.get()
    dynamo_config: Any = torch._dynamo.config
    if depth == 0:
        prior = (
            functorch_config.bundled_autograd_cache,
            dynamo_config.allow_empty_graphs,
            functorch_config.force_non_lazy_backward_lowering,
        )
        functorch_config.bundled_autograd_cache = True
        dynamo_config.allow_empty_graphs = True
        if training:
            # AOTAutograd lowers the backward on the first .backward() call, so
            # a capture that never calls one records a forward and nothing else.
            # Lower it eagerly instead: the joint trace already synthesizes
            # tangents from the forward outputs' metadata, so the backward graph
            # exists without the caller having to produce a loss.
            functorch_config.force_non_lazy_backward_lowering = True
    _CAPTURE_CONFIG_STATE.set((depth + 1, prior))
    try:
        yield
    finally:
        depth, prior = _CAPTURE_CONFIG_STATE.get()
        if depth <= 0 or prior is None:
            raise AssertionError("precompile capture config is not active")
        depth -= 1
        if depth == 0:
            functorch_config.bundled_autograd_cache = prior[0]
            dynamo_config.allow_empty_graphs = prior[1]
            functorch_config.force_non_lazy_backward_lowering = prior[2]
            _CAPTURE_CONFIG_STATE.set((0, None))
        else:
            _CAPTURE_CONFIG_STATE.set((depth, prior))


def _clear_package_region(
    codes: Sequence[types.CodeType], isolate_recompiles_id: int
) -> None:
    from .eval_frame import _clear_cache_entries_for_region

    for code in codes:
        _clear_cache_entries_for_region(code, isolate_recompiles_id)


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
    ``PrecompileSummary.dropped_guards``. ``save()`` does NOT refuse them by
    default -- ``require_no_dropped_guards`` is False, because requiring none
    would refuse essentially every model. The rail that is on is the risky-drop
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
    ``_RISKY_DROP_CORPUS`` in test_package.py is the regression net that keeps
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


# Object ids and the per-tensor _dynamo_*_indices probes are noise in a file
# meant to be read and diffed: the first differs every run, the second is
# identical on every tensor guard. The id pattern matches the second argument
# of the ___check_obj_id / ___check_type_id call that emits it, NOT every long
# integer: a bare \b\d{9,}\b also eats a user constant, and two variants
# guarding different big values -- dict keys, a slice bound -- then render the
# same fact and land in the intersection as an invariant neither of them holds.
# An id inside ___check_obj_id(x, N), type=... and the raw tuple of them that
# AUTOGRAD_SAVED_TENSORS_HOOKS renders. Both are addresses; both must go, and
# neither may be widened into "any long integer", which would also erase a
# large constant that legitimately differs between variants and so invent an
# invariant. Keep these anchored to the shapes that actually carry addresses.
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
    # The _dynamo_*_indices parts are NOT boilerplate, whatever an earlier
    # comment here claimed: they are the whole of TENSOR_MATCH's export info and
    # they encode the dimension-marking guards, so mark_static on one variant
    # and not the next shows up only here. Dropping them reported the guard that
    # split two compilations as an invariant of both.
    return tuple(_normalize(part) for part in (code_list or ()))


class _PrecompileBackend:
    """Give one explicit session its own Dynamo cache identity."""

    def __init__(self, backend: str) -> None:
        inner = torch._dynamo.lookup_backend(backend)
        self._torchdynamo_orig_backend = inner
        self._torchdynamo_cache_key = object()
        # get_backend skips looking for that key until it knows one exists,
        # because the lookup is a raising miss at every level of the callback
        # chain for every torch.compile user who never precompiles.
        torch._C._dynamo.eval_frame._enable_precompile_cache_keys()
        self.backend_ctx_ctor = getattr(
            inner, "backend_ctx_ctor", contextlib.nullcontext
        )

    def __call__(self, gm: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Any:
        return self._torchdynamo_orig_backend(gm, inputs)

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


# Guards whose C++ leaf compares something no fingerprint here models: subclass
# metadata, a DTensor placement, an opaque object's guard values, a raw
# DispatchKeySet, or process-wide state carried entirely in the leaf. Calling
# two of these equal is how the report ends up asserting a precondition that
# does not hold, so they are never compared -- they are reported separately as
# undetermined, which is the honest answer and cannot mislead.
_UNMODELLED_GUARD_TYPES = frozenset(
    {
        "DETERMINISTIC_ALGORITHMS",
        "DISPATCH_KEY_SET_MATCH",
        "DTENSOR_SPEC_MATCH",
        "FSDP_TRAINING_STATE",
        "GLOBAL_STATE",
        "OPAQUE_OBJ_GUARD_FN_MATCH",
        "SHAPE_ENV",
        "TENSOR_SUBCLASS_METADATA_MATCH",
        "TORCH_FUNCTION_STATE",
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
    if not entry.has_value:
        return ""
    value = entry.value
    if isinstance(value, torch.Tensor):
        # Ask the guard system what it stores rather than reconstructing it.
        # Hand-rolling this failed four times running -- the concrete shape,
        # then python type, then the conj/neg bits, then the rest of the
        # dispatch key set -- and the last of those was still wrong, because
        # TensorCheck stores the TLS-ADJUSTED key set, (ks | tls_included) -
        # tls_excluded, not the tensor's own. get_tensor_guard_code_part is the
        # function guards.py itself uses to render that, so it cannot drift.
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


def _example_call(
    example: ExampleInput | tuple[object, ...],
) -> tuple[tuple[object, ...], dict[str, object]]:
    if isinstance(example, ExampleInput):
        return example.args, example.kwargs
    if isinstance(example, tuple):
        return example, {}
    raise TypeError(
        f"example_inputs takes tuples of positional args or ExampleInput, got "
        f"{type(example).__name__}. Wrap keyword arguments in "
        f"torch.compiler.precompile.ExampleInput."
    )


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


def varying_guard_slots(
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
) -> frozenset[tuple[str, str]]:
    """The guard slots that actually discriminate between captured variants.

    A slot is ``(guard_type, normalized source)``. It varies when two variants
    of one frame recorded DIFFERENT facts for it, and also when it is present in
    some variants and absent in others -- a guard only one variant carries is
    what tells that variant apart, and comparing values alone would call it
    invariant and drop it. On a 62-frame ranking model that second case is 225
    of the 274 slots kept, so it is the majority of the answer, not an edge.

    Everything else held identically in every variant, so it is not serialized:
    see the rationale at the call site in ``torch._precompile``.
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
    risky: set[tuple[str, str]],
    truncated: frozenset[str],
    uncovered: frozenset[str],
    capture_errors: Sequence[str],
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
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
        kept_guards=tuple(sorted(kept)),
        risky_dropped_guards=tuple(sorted(risky)),
        capture_errors=tuple(capture_errors),
    )


class PrecompileSession:
    """
    A capture in progress. Use as a context manager to get the callable to
    exercise, then ``save()``. A session is one-shot and cannot be re-entered.
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
        example_inputs: Sequence[ExampleInput | tuple[object, ...]] | None = None,
        invariants: str | None = None,
        keep_only: frozenset[tuple[str, str]] | None = None,
        training: bool = False,
    ) -> None:
        # Not `example_inputs or ()`: truth-testing the caller's object turns the
        # likeliest mistake -- passing the tensors themselves instead of a
        # sequence of call tuples -- into either a raw "Boolean value of Tensor
        # is ambiguous" from inside torch, or, for a falsy tensor, a silently
        # empty capture that only fails much later at save().
        bad_container = isinstance(
            example_inputs, (torch.Tensor, torch.nn.Module, str)
        ) or not isinstance(example_inputs, Sequence)
        if example_inputs is None:
            self._example_inputs: tuple[ExampleInput | tuple[object, ...], ...] = ()
        elif bad_container:
            raise TypeError(
                f"example_inputs takes a sequence of calls -- each a tuple of "
                f"positional args, or a "
                f"torch.compiler.precompile.ExampleInput -- got "
                f"{type(example_inputs).__name__}. A single call is "
                f"example_inputs=[(x,)], not example_inputs=x."
            )
        else:
            self._example_inputs = tuple(example_inputs)
        self._invariants_path = invariants
        self._fn = fn
        self._backend = backend
        self._custom_guard_filter = guard_filter_fn is not None
        # The guard slots that DID vary across the capture; every other slot is
        # dropped. None means serialize everything serializable, which only the
        # internal capture entry point uses -- precompile() always passes a set.
        self._keep_only: frozenset[tuple[str, str]] | None = keep_only
        # A training capture traces with grad on and lowers the backward
        # eagerly, so the artifact carries AOTAutograd's CompiledFunction and
        # calling .backward() on a served output runs precompiled code.
        self._training = training
        self._policy_dropped_guards: set[tuple[str, str]] = set()
        self._dropped_guards: set[tuple[str, str]] = set()
        self._kept_guards: set[tuple[str, str]] = set()
        self._risky_dropped_guards: set[tuple[str, str]] = set()
        self._capture_errors: list[str] = []
        self._recorded_exception_keys: set[tuple[type[BaseException], str]] = set()
        # (co_name, co_filename, co_firstlineno) -> one fact set per compilation
        self._guard_sets: dict[tuple[str, str, int], list[frozenset[_GuardFact]]] = {}
        self._undetermined: dict[tuple[str, str, int], set[_GuardFact]] = {}
        self._guard_filter_fn = self._recording_filter(
            default_guard_filter_fn if guard_filter_fn is None else guard_filter_fn
        )
        self._recompile_limit = recompile_limit
        self._dynamic = dynamic
        self._entry_fn = _entry_fn_of(fn)
        self._package = CompilePackage(
            self._entry_fn,
            serialization_guard_filter_fn=self._guard_filter_fn,
            requires_native_backend_compatibility=backend != "eager",
        )
        self._backend_artifacts: dict[_BackendId, Any] = {}
        self._stack: contextlib.ExitStack | None = None
        self._optimized: Callable[..., object] | None = None
        self._compiled: Callable[..., object] | None = None
        self._isolate_recompiles_id: int | None = None
        from .pgo import _new_code_state

        self._pgo_state = _new_code_state()
        self._state = threading.Condition()
        self._active_calls = 0
        self._closing = False
        self._finished = False

    def _take_backend_artifacts(self) -> None:
        from torch._dynamo.precompile_context import PrecompileContext

        for backend_id in self._package.cache_entry().backend_ids:
            artifact = PrecompileContext.take_artifact(backend_id)
            if artifact is not None:
                self._backend_artifacts[backend_id] = artifact

    def _record_capture_error(self, error: BaseException) -> None:
        message = str(error)
        key = (type(error), message)
        if key in self._recorded_exception_keys:
            return
        self._recorded_exception_keys.add(key)
        self._capture_errors.append(f"{type(error).__name__}: {message}")

    @staticmethod
    def _check_module_state(module: torch.nn.Module) -> None:
        state = (
            ("parameter", module.named_parameters()),
            ("buffer", module.named_buffers()),
        )
        for kind, tensors in state:
            for name, tensor in tensors:
                if torch.is_inference(tensor):
                    raise PackageError(
                        f"automatic example capture found inference tensor {kind} "
                        f"{name!r} on {type(module).__name__}. Automatic examples "
                        "capture ordinary torch.no_grad() semantics, but leaving "
                        "inference_mode does not turn existing tensors into ordinary "
                        "tensors. Create the module outside torch.inference_mode(), or "
                        "use capture() manually and serve under the matching inference "
                        "mode."
                    )

    def _clear_runtime_cache(self) -> None:
        if self._isolate_recompiles_id is None:
            return
        from .eval_frame import _unregister_explicit_compile_region

        isolate_recompiles_id = self._isolate_recompiles_id
        try:
            _clear_package_region(self._package.region_codes(), isolate_recompiles_id)
        finally:
            _unregister_explicit_compile_region(isolate_recompiles_id)
            self._isolate_recompiles_id = None
            self._optimized = None
            if self._backend != "eager":
                self._package.cached_backends.clear()
            self._pgo_state.clear()

    def _call(self, *args: object, **kwargs: object) -> object:
        with self._state:
            if self._compiled is None or self._closing:
                raise RuntimeError("PrecompileSession is not active")
            compiled = self._compiled
            self._active_calls += 1
        from .pgo import _use_code_state

        try:
            with _capture_config(), _use_code_state(self._pgo_state):
                return compiled(*args, **kwargs)
        except BaseException as e:
            self._record_capture_error(e)
            raise
        finally:
            with self._state:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._state.notify_all()

    def __enter__(self) -> Callable[..., object]:
        if self._finished:
            raise RuntimeError("PrecompileSession cannot be re-entered")
        if self._stack is not None:
            raise RuntimeError("PrecompileSession is already active")
        stack = contextlib.ExitStack()
        # Backends must serialize into the artifact rather than into the
        # process-local inductor cache.
        stack.enter_context(_capture_config(self._training))
        self._stack = stack
        try:
            if self._example_inputs:
                module = (
                    self._fn
                    if isinstance(self._fn, torch.nn.Module)
                    else getattr(self._fn, "__self__", None)
                )
                if isinstance(module, torch.nn.Module):
                    self._check_module_state(module)
            if self._optimized is None:
                optimize_ctx = torch._dynamo.optimize(
                    _PrecompileBackend(self._backend),
                    package=self._package,
                    recompile_limit=self._recompile_limit,
                    dynamic=self._dynamic,
                    isolate_recompiles=True,
                )
                isolate_recompiles_id = getattr(
                    optimize_ctx, "_isolate_recompiles_id", None
                )
                if not isinstance(isolate_recompiles_id, int):
                    raise AssertionError("missing isolate_recompiles_id")
                self._isolate_recompiles_id = isolate_recompiles_id
                from .eval_frame import _register_explicit_compile_region

                _register_explicit_compile_region(isolate_recompiles_id, self)
                self._optimized = optimize_ctx(self._fn)
            self._compiled = self._optimized
            # Automatic examples are the ordinary no_grad inference path.
            # inference_mode is a distinct guarded state and is disabled here
            # even when the caller entered it before starting capture.
            for example in self._example_inputs:
                args, kwargs = _example_call(example)
                if any(
                    isinstance(value, torch.Tensor) and torch.is_inference(value)
                    for value in tree_leaves((args, kwargs))
                ):
                    raise PackageError(
                        "example_inputs contains an inference tensor. Automatic "
                        "examples capture ordinary torch.no_grad() semantics, but "
                        "leaving inference_mode does not turn an existing inference "
                        "tensor into an ordinary tensor. Create automatic inputs "
                        "outside torch.inference_mode(), or use capture() manually "
                        "and serve under the matching inference mode."
                    )
                for value in tree_leaves((args, kwargs)):
                    if isinstance(value, torch.nn.Module):
                        self._check_module_state(value)
                if self._training:
                    # Grad must be ON: the point of a training capture is that
                    # AOTAutograd builds its CompiledFunction and the joint
                    # graph, which it only does when the outputs require grad.
                    with torch.inference_mode(False), torch.enable_grad():
                        self._call(*args, **kwargs)
                else:
                    with torch.inference_mode(False), torch.no_grad():
                        self._call(*args, **kwargs)
        except BaseException as e:
            self._record_capture_error(e)
            # A __enter__ that raises never gets its __exit__, so without this
            # the config patch above stays on for the life of the process and
            # the session is wedged: save() reports the block as still open.
            self._stack = None
            self._compiled = None
            try:
                stack.close()
            finally:
                try:
                    self._take_backend_artifacts()
                finally:
                    self._clear_runtime_cache()
                    self._finished = True
            raise
        finally:
            # Guard/backend state is already in the package. Do not retain the
            # caller's potentially large CPU/GPU example tensors with the session.
            self._example_inputs = ()
        return self._call

    def __exit__(self, *exc: object) -> None:
        if isinstance(exc[1], BaseException):
            self._record_capture_error(exc[1])
        with self._state:
            self._closing = True
            try:
                while self._active_calls:
                    self._state.wait()
            except BaseException:
                self._closing = False
                self._state.notify_all()
                raise
            stack = self._stack
            self._stack = None
            self._compiled = None
        if stack is not None:
            try:
                stack.close()
            except BaseException as e:
                self._record_capture_error(e)
                raise
            finally:
                try:
                    self._take_backend_artifacts()
                finally:
                    self._clear_runtime_cache()
                    self._finished = True
                    with self._state:
                        self._state.notify_all()
        self._recorded_exception_keys.clear()
        if self._invariants_path is None:
            return
        if exc[0] is None:
            self.write_invariants(self._invariants_path)
        else:
            # A partial capture's report reads exactly like a complete one, so
            # it is not written; say why rather than leaving the user looking
            # for a file that never appeared.
            log.warning(
                "precompile: the capture block raised %s, so no invariants "
                "report was written to %s. Call write_invariants() for the "
                "partial one.",
                getattr(exc[0], "__name__", exc[0]),
                self._invariants_path,
            )

    def _recording_filter(
        self,
        inner: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]],
    ) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
        """
        Remember which guard types were discarded. A dropped guard does not
        fail at serving time, it silently widens what a graph is reused for, so
        the set has to be inspectable rather than invisible.
        """

        # One object per distinct fact, shared by every compilation that
        # produced it. A recompiled frame repeats nearly all of its guards, so
        # storing a copy per compilation makes the session grow with variants
        # rather than with facts: 200 variants of resnet18 held 83MB for 1281.
        pool: dict[_GuardFact, _GuardFact] = {}

        def filter_fn(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
            decisions = inner(entries)
            namespaces = _module_namespaces(entries)
            # A slot whose fact was identical in every variant of every frame
            # discriminates nothing, so it is not serialized. Recorded apart
            # from the ordinary drops, which are "could not be serialized";
            # these could, and were not wanted.
            policy_dropped: set[int] = set()
            if self._keep_only is not None:
                for i, entry in enumerate(entries):
                    # An unmodelled guard never enters _guard_sets, so it was
                    # never shown to be constant -- only never analyzed. This
                    # policy drops what it PROVED invariant, so these stay.
                    # SHAPE_ENV is the one that matters: it carries symbolic
                    # shape constraints that no TENSOR_MATCH repeats.
                    if entry.guard_type in _UNMODELLED_GUARD_TYPES:
                        continue
                    if (
                        decisions[i]
                        and (
                            entry.guard_type,
                            _normalize(entry.name),
                        )
                        not in self._keep_only
                    ):
                        policy_dropped.add(i)
                decisions = [
                    keep and i not in policy_dropped for i, keep in enumerate(decisions)
                ]
            facts: set[_GuardFact] = set()
            undetermined: set[_GuardFact] = set()
            for i, (keep, entry) in enumerate(zip(decisions, entries)):
                slot = (entry.guard_type, entry.name)
                if i in policy_dropped:
                    # Not "risky": the policy is the caller stating that the
                    # environment is fixed and every variation is in the inputs,
                    # so a slot that never varied is out of the artifact's
                    # declared domain rather than an unchecked hazard.
                    self._policy_dropped_guards.add(slot)
                    continue
                target = self._kept_guards if keep else self._dropped_guards
                target.add(slot)
                if not keep and (
                    self._custom_guard_filter or _is_risky_drop(entry, namespaces)
                ):
                    self._risky_dropped_guards.add(slot)
                unmodelled = entry.guard_type in _UNMODELLED_GUARD_TYPES
                fact = _GuardFact(
                    guard_type=entry.guard_type,
                    source=_normalize(entry.name),
                    code=_render_code(entry.code),
                    value="" if unmodelled else _value_fingerprint(entry),
                    enforced=keep,
                )
                fact = pool.setdefault(fact, fact)
                # Never compared, so never claimed to hold: see
                # _UNMODELLED_GUARD_TYPES.
                (undetermined if unmodelled else facts).add(fact)
            # One filter call is one compilation, and the package knows which
            # frame is being compiled, which is the only place that mapping is
            # available to us.
            current = self._package._current_entry
            code = current.python_code if current is not None else None
            key = (
                (code.co_name, code.co_filename, code.co_firstlineno)
                if code is not None
                else ("<unknown>", "<unknown>", 0)
            )
            self._guard_sets.setdefault(key, []).append(frozenset(facts))
            self._undetermined.setdefault(key, set()).update(undetermined)
            return decisions

        return filter_fn

    def invariants(self) -> tuple[FrameInvariants, ...]:
        """
        Per frame, the guards that held in EVERY compiled variant of it.

        Intersection is per frame rather than global because guards from
        different frames are not comparable: the entry frame guards its
        arguments, a resume frame guards whatever crossed the graph break, so a
        global intersection would be empty for any model that breaks.

        A frame compiled once reports everything as invariant, which is true but
        uninformative -- exercise more than one variant for the diff to mean
        anything.

        GLOBAL_STATE, TORCH_FUNCTION_STATE and DETERMINISTIC_ALGORITHMS carry
        no value of their own, so nothing here can say whether two variants
        agreed on, say, autocast. They are listed as UNDETERMINED rather than
        compared -- see _UNMODELLED_GUARD_TYPES, and note that calling them
        equal is precisely how the report would assert a precondition that does
        not hold. GRAD_MODE is fingerprinted instead, because the capture path
        itself produces that split and the fingerprint can model it.
        """
        out = []
        for (name, filename, lineno), sets in sorted(self._guard_sets.items()):
            shared = frozenset.intersection(*sets) if sets else frozenset()
            everything: set[_GuardFact] = set()
            for one in sets:
                everything |= one
            out.append(
                FrameInvariants(
                    frame=name,
                    filename=filename,
                    lineno=lineno,
                    variants=len(sets),
                    invariant=tuple(sorted(shared, key=_fact_order)),
                    varying=tuple(sorted(everything - shared, key=_fact_order)),
                    undetermined=tuple(
                        sorted(
                            self._undetermined.get((name, filename, lineno), set()),
                            key=_fact_order,
                        )
                    ),
                )
            )
        return tuple(out)

    def write_invariants(self, path: str) -> None:
        """
        Write :meth:`invariants` to ``path`` in human-readable form.

        ``path`` is a FILE, written exactly as given, with parent directories
        created -- ``snapshots/invariants.txt`` is a text file. Same contract as
        :meth:`save`.

        Output is stable across runs of the same capture: object ids and
        Dynamo's per-process counters are normalized away, so the file can be
        committed and diffed to see what a model change did to its guards.
        """
        frames = self.invariants()
        target = getattr(self._fn, "__qualname__", None) or type(self._fn).__qualname__
        lines = [
            f"# precompile invariants for {target}",
            "#",
            "# Conditions that held in EVERY compiled variant of a frame. A call",
            "# violating one cannot be served by any graph in this artifact, so",
            "# these are the preconditions the artifact is only valid under.",
            "# 'varies' lists what differed between variants -- those are what",
            "# distinguish one compiled graph from another, not preconditions.",
            "# 'unknown' lists guards whose check this report cannot model, so it",
            "# cannot say whether they held across variants. Treat them as",
            "# neither: they may or may not be preconditions.",
            "#",
            "# enforced = the guard is serialized and rechecked when the artifact",
            "#            is loaded.",
            "# dropped  = it was not serialized, so it is a precondition",
            "#            NOTHING checks at serving time. See",
            "#            PrecompileSummary.dropped_guards.",
            "#",
            f"# {len(frames)} frame(s), "
            f"{sum(f.variants for f in frames)} compilation(s)",
        ]
        if any(f.variants < 2 for f in frames):
            lines.append(
                "# NOTE: some frames were compiled once, so their invariants are"
                " just every guard. Exercise more variants for a real diff."
            )
        for f in frames:
            where = f"{os.path.basename(f.filename)}:{f.lineno}"
            lines.append("")
            lines.append(
                f"frame {f.frame} ({where})  {f.variants} variant(s), "
                f"{len(f.invariant)} invariant, {len(f.varying)} varying, "
                f"{len(f.undetermined)} undetermined"
            )
            if not f.invariant:
                lines.append("  invariant: (none)")
            for fact in f.invariant:
                lines.append(f"  invariant {fact.render()}")
            for fact in f.varying:
                lines.append(f"  varies    {fact.render()}")
            for fact in f.undetermined:
                lines.append(f"  unknown   {fact.render()}")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # UTF-8 explicitly: the report renders user identifiers (module, class and
        # parameter names) and the ambient locale can be ASCII in a container, where
        # a non-ASCII name would raise UnicodeEncodeError and leave a truncated file
        # behind -- from the one call that exists to explain the artifact.
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        log.info(
            "precompile: wrote invariants for %d frame(s) to %s", len(frames), path
        )

    def summary(self) -> PrecompileSummary:
        # A dropped fact counts as risky through this arm only when the SAME
        # source held DIFFERENT values across the captured variants -- i.e. the
        # drop demonstrably distinguishes them. Merely being absent from one
        # variant does not: a MODULE_MATCH on torch.nn.functional that one
        # branch happens not to touch was flagging every ordinary multi-branch
        # capture, which is precisely the shape example_inputs= exists for, and
        # left the operator no escape but disabling the rail wholesale. Grouped
        # by source rather than by (guard_type, source) because a rebind can
        # change the guard type too, and that is the case worth keeping.
        values_by_source: dict[str, set[str]] = {}
        for frame in self.invariants():
            for fact in frame.varying:
                if not fact.enforced:
                    values_by_source.setdefault(fact.source, set()).add(fact.value)
        varying_dropped = {
            (fact.guard_type, fact.source)
            for frame in self.invariants()
            for fact in frame.varying
            if not fact.enforced and len(values_by_source.get(fact.source, ())) > 1
        }
        # Normalized: a Dynamo per-process counter (__builtins_dict___14) makes
        # one logical drop appear once per compilation, under names that change
        # every run, which inflates the count save() truncates at 8 and can push
        # the genuinely config-selected drop out of the operator's view.
        risky = {
            (guard_type, _normalize(name))
            for guard_type, name in self._risky_dropped_guards
        } | {
            (guard_type, _normalize(name))
            for guard_type, name in self._dropped_guards
            if (guard_type, _normalize(name)) in varying_dropped
        }
        return _summarize(
            self._package.cache_entry(),
            self._dropped_guards,
            self._kept_guards,
            risky,
            self._package.truncated_frames,
            self._package.uncovered_frames,
            self._capture_errors,
            self._guard_sets,
        )

    def _gated_summary(
        self,
        *,
        require_complete: bool,
        require_no_risky_drops: bool,
        require_no_dropped_guards: bool,
        caller: str,
    ) -> PrecompileSummary:
        """Run the coverage and guard gates, or raise saying which one failed."""
        if self._stack is not None:
            raise RuntimeError(f"{caller} must be called after the capture block exits")
        summary = self.summary()
        if require_complete and summary.capture_errors:
            raise PackageError(
                "Precompilation is incomplete because capture raised: "
                f"{list(summary.capture_errors)}. Re-run every example successfully, "
                "or pass require_complete=False to save the partial artifact."
            )
        if require_no_dropped_guards and summary.dropped_guards:
            raise PackageError(
                f"Precompilation dropped {len(summary.dropped_guards)} guard(s) that "
                f"were not serialized: {list(summary.dropped_guards)}. Rebinding any "
                f"of those sources between capture and load can silently serve a graph "
                f"traced against the old value. Pass require_no_dropped_guards=False "
                f"only to select the relaxed risky-drop policy."
            )
        if summary.risky_dropped_guards and require_no_risky_drops:
            raise PackageError(
                f"Precompilation dropped guard(s) that can affect dispatch on "
                f"{[n for _, n in summary.risky_dropped_guards]}. Each of those names "
                f"either a configuration-dependent identity slot or a guard discarded "
                f"by a custom filter. Nothing checks it at load time, so a different "
                f"value can silently select the wrong graph instead of recompiling. "
                f"Make the value reachable through a serializable guard, pin both "
                f"machines to the same value, or pass "
                f"require_no_risky_drops=False to accept the risk explicitly."
            )
        elif summary.risky_dropped_guards:
            # The caller explicitly accepted the risk. Measured on stock models: torchvision
            # resnet18 and mobilenet_v3 report none, timm's ViT reports one (a
            # re-exported torch._assert) and transformers' Qwen2 reports 33,
            # nearly all of them library internals that no deployment swaps.
            names = [n for _, n in summary.risky_dropped_guards]
            # The cut below is in guard-type order and says nothing about
            # severity: Qwen2's one genuinely config-selected drop sorts past
            # it, so a truncated report has to say where the rest are.
            rest = (
                ""
                if len(names) <= 8
                else f" Only the first 8 are shown, cut in guard-type order "
                f"rather than by severity; summary().risky_dropped_guards has "
                f"all {len(names)}."
            )
            log.warning(
                "precompile: %d dropped guard(s) can affect dispatch, so nothing "
                "checks them at load: %s.%s Audit them against "
                "your deployment; this warning appears only because "
                "require_no_risky_drops=False explicitly accepted them.",
                len(names),
                names[:8],
                rest,
            )
        if require_complete:
            if summary.guarded_codes == 0:
                raise PackageError(
                    "Precompilation captured no compiled code. Capture happens by "
                    "execution, so the callable must actually be run inside the "
                    "capture block. A call Dynamo could not turn into guarded code "
                    "is reported separately as an uncovered frame."
                )
            if summary.truncated:
                raise PackageError(
                    f"Precompilation is incomplete: at least "
                    f"{len(summary.truncated)} frame(s) exceeded recompile_limit "
                    f"(currently {self._recompile_limit}) and are missing variants: "
                    f"{list(summary.truncated)}. That list is a lower bound -- hitting "
                    f"the limit also puts every frame called beneath the named one "
                    f"into run-only mode, so those stop capturing too and never "
                    f"re-enter Dynamo to report it. A frame needs one slot per "
                    f"variant, and frames shared across module instances accumulate "
                    f"them. Raise recompile_limit, or pass require_complete=False to "
                    f"accept an artifact that is more incomplete than this list shows."
                )
            if summary.uncovered_frames:
                raise PackageError(
                    f"Precompilation exercised frame(s) that produced NO guarded code "
                    f"at all: {list(summary.uncovered_frames)}. Those paths are absent "
                    f"from the artifact, and such a frame is skipped at install and runs "
                    f"eager, so serving() cannot report that gap. This is expected for a "
                    f"frame that only dispatches to covered submodules; it also looks "
                    f"exactly like a frame Dynamo gave up on (check "
                    f"TORCH_LOGS=graph_breaks for gb0124). A frame that hit the recompile "
                    f"limit has working variants and is reported as truncated instead. "
                    f"Pass require_complete=False once you have confirmed which."
                )
            if summary.bypassed:
                raise PackageError(
                    f"Precompilation is incomplete: {len(summary.bypassed)} frame(s) "
                    f"were bypassed and will serve nothing: {list(summary.bypassed)}. "
                    f"This usually means their guards could not be serialized. Pass "
                    f"require_complete=False to accept a partial artifact."
                )
        if summary.wont_generalize:
            log.warning(
                "precompile: %d value(s) are pinned to what capture saw (%s). A call "
                "supplying anything else misses every graph, so exercise each value "
                "you need to serve inside the capture block.",
                len(summary.wont_generalize),
                list(summary.wont_generalize),
            )
        return summary

    def save(
        self,
        path: str,
        *,
        require_complete: bool = True,
        require_no_risky_drops: bool = True,
        require_no_dropped_guards: bool = False,
    ) -> PrecompileSummary:
        r"""Write the artifact with independent coverage and guard gates.

        ``require_complete`` gates exactly ``PrecompileSummary.complete``. The
        two guard rails are separate from it, because a dropped guard is a wrong
        answer rather than a gap in coverage. ``require_no_dropped_guards`` is
        False here and in the public ``torch.compiler.precompile`` facade:
        every model drops the identity guards precompile cannot serialize, so
        requiring none refuses essentially every real artifact. The rail that
        is on by default is ``require_no_risky_drops``, and the risky subset is
        deliberately only a lint, not a proof.

        ``path`` is a FILE, written exactly as given with parent directories
        created, and ``precompile_load`` takes it straight back. Same contract
        as ``invariants``.
        """
        if self._stack is not None:
            raise RuntimeError("save() must be called after the capture block exits")
        summary = self.summary()
        if require_complete and summary.capture_errors:
            raise PackageError(
                "Precompilation is incomplete because capture raised: "
                f"{list(summary.capture_errors)}. Re-run every example successfully, "
                "or pass require_complete=False to save the partial artifact."
            )
        if require_no_dropped_guards and summary.dropped_guards:
            raise PackageError(
                f"Precompilation dropped {len(summary.dropped_guards)} guard(s) that "
                f"were not serialized: {list(summary.dropped_guards)}. Rebinding any "
                f"of those sources between capture and load can silently serve a graph "
                f"traced against the old value. Pass require_no_dropped_guards=False "
                f"only to select the relaxed risky-drop policy."
            )
        if summary.risky_dropped_guards and require_no_risky_drops:
            raise PackageError(
                f"Precompilation dropped guard(s) that can affect dispatch on "
                f"{[n for _, n in summary.risky_dropped_guards]}. Each of those names "
                f"either a configuration-dependent identity slot or a guard discarded "
                f"by a custom filter. Nothing checks it at load time, so a different "
                f"value can silently select the wrong graph instead of recompiling. "
                f"Make the value reachable through a serializable guard, pin both "
                f"machines to the same value, or pass "
                f"require_no_risky_drops=False to accept the risk explicitly."
            )
        elif summary.risky_dropped_guards:
            # The caller explicitly accepted the risk. Measured on stock models: torchvision
            # resnet18 and mobilenet_v3 report none, timm's ViT reports one (a
            # re-exported torch._assert) and transformers' Qwen2 reports 33,
            # nearly all of them library internals that no deployment swaps.
            names = [n for _, n in summary.risky_dropped_guards]
            # The cut below is in guard-type order and says nothing about
            # severity: Qwen2's one genuinely config-selected drop sorts past
            # it, so a truncated report has to say where the rest are.
            rest = (
                ""
                if len(names) <= 8
                else f" Only the first 8 are shown, cut in guard-type order "
                f"rather than by severity; summary().risky_dropped_guards has "
                f"all {len(names)}."
            )
            log.warning(
                "precompile: %d dropped guard(s) can affect dispatch, so nothing "
                "checks them at load: %s.%s Audit them against "
                "your deployment; this warning appears only because "
                "require_no_risky_drops=False explicitly accepted them.",
                len(names),
                names[:8],
                rest,
            )
        if require_complete:
            if summary.guarded_codes == 0:
                raise PackageError(
                    "Precompilation captured no compiled code. Capture happens by "
                    "execution, so the callable must actually be run inside the "
                    "capture block. A call Dynamo could not turn into guarded code "
                    "is reported separately as an uncovered frame."
                )
            if summary.truncated:
                raise PackageError(
                    f"Precompilation is incomplete: at least "
                    f"{len(summary.truncated)} frame(s) exceeded recompile_limit "
                    f"(currently {self._recompile_limit}) and are missing variants: "
                    f"{list(summary.truncated)}. That list is a lower bound -- hitting "
                    f"the limit also puts every frame called beneath the named one "
                    f"into run-only mode, so those stop capturing too and never "
                    f"re-enter Dynamo to report it. A frame needs one slot per "
                    f"variant, and frames shared across module instances accumulate "
                    f"them. Raise recompile_limit, or pass require_complete=False to "
                    f"accept an artifact that is more incomplete than this list shows."
                )
            if summary.uncovered_frames:
                raise PackageError(
                    f"Precompilation exercised frame(s) that produced NO guarded code "
                    f"at all: {list(summary.uncovered_frames)}. Those paths are absent "
                    f"from the artifact, and such a frame is skipped at install and runs "
                    f"eager, so serving() cannot report that gap. This is expected for a "
                    f"frame that only dispatches to covered submodules; it also looks "
                    f"exactly like a frame Dynamo gave up on (check "
                    f"TORCH_LOGS=graph_breaks for gb0124). A frame that hit the recompile "
                    f"limit has working variants and is reported as truncated instead. "
                    f"Pass require_complete=False once you have confirmed which."
                )
            if summary.bypassed:
                raise PackageError(
                    f"Precompilation is incomplete: {len(summary.bypassed)} frame(s) "
                    f"were bypassed and will serve nothing: {list(summary.bypassed)}. "
                    f"This usually means their guards could not be serialized. Pass "
                    f"require_complete=False to accept a partial artifact."
                )
        if summary.wont_generalize:
            log.warning(
                "precompile: %d value(s) are pinned to what capture saw (%s). A call "
                "supplying anything else misses every graph, so exercise each value "
                "you need to serve inside the capture block.",
                len(summary.wont_generalize),
                list(summary.wont_generalize),
            )
        self._take_backend_artifacts()
        store = _SingleFileStore(self._backend_artifacts)
        if self._backend == "eager":
            # Eager "backends" are fx graphs with no compiled artifact of their
            # own, so they have to be handed to the store explicitly.
            for backend_id, backend in self._package.cached_backends.items():
                store.record_eager_backend(backend_id, backend)
        try:
            # save_cache_entry, not save_package: the latter also files the
            # package into the process-global PrecompileContext, which is for
            # the transparent cache. An artifact written to a path the caller
            # chose should not leave anything behind.
            store.save_cache_entry(self._package.cache_entry(), path)
        except RuntimeError as e:
            if "is not found in the given backends" not in str(e):
                raise
            raise PackageError(
                "Precompilation captured graphs but their compiled backends were "
                "never recorded, so there is nothing to serialize. AOTAutograd only "
                "records the bundled artifact once the BACKWARD compiles, so a "
                "forward-only capture with grad enabled records nothing on its own. "
                "Pass training=True to lower the backward eagerly (the joint "
                "trace synthesizes tangents, so no loss is needed), capture "
                "under torch.no_grad()/torch.inference_mode() for an inference "
                "artifact, or run .backward() inside the capture block."
            ) from e
        log.info("precompile: saved %s to %s", summary, path)
        return summary

    def _collect_backends(self) -> dict[str, Any]:
        """The compiled subgraphs this capture produced, keyed by backend id."""
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        self._take_backend_artifacts()
        collected = dict(self._backend_artifacts)
        if self._backend == "eager":
            # Eager "backends" are fx graphs with no compiled artifact of their
            # own, so they have to be gathered explicitly.
            for backend_id, backend in self._package.cached_backends.items():
                collected[backend_id] = EagerCacheArtifact(
                    key=backend_id, content=backend
                )
        entry = self._package.cache_entry()
        for backend_id in entry.backend_ids:
            if backend_id not in collected:
                artifact = PrecompileContext.take_artifact(backend_id)
                if artifact is not None:
                    collected[backend_id] = artifact
        missing = [b for b in entry.backend_ids if b not in collected]
        if missing:
            raise PackageError(
                "Precompilation captured graphs but their compiled backends were "
                "never recorded, so there is nothing to serialize. AOTAutograd only "
                "records the bundled artifact once the BACKWARD compiles, so a "
                "forward-only capture with grad enabled records nothing on its own. "
                "Pass training=True to lower the backward eagerly (the joint "
                "trace synthesizes tangents, so no loss is needed), capture "
                "under torch.no_grad()/torch.inference_mode() for an inference "
                "artifact, or run .backward() inside the capture block."
            )
        return {str(b): collected[b] for b in entry.backend_ids}

    def artifact(
        self,
        *,
        require_complete: bool = True,
        require_no_risky_drops: bool = True,
        require_no_dropped_guards: bool = False,
    ) -> tuple[str, bytes]:
        """Render this capture as ``(python_code, cache)``.

        Same contract and the same gates as the single-graph forms: python_code
        is a self-contained module exposing ``forward``, and cache is an
        acceleration the loader may ignore. The guard trees and transformed
        bytecode have no readable form and sit in labelled opaque sections.
        """
        summary = self._gated_summary(
            require_complete=require_complete,
            require_no_risky_drops=require_no_risky_drops,
            require_no_dropped_guards=require_no_dropped_guards,
            caller="artifact()",
        )
        from torch._precompile import _build_multigraph_artifact

        return _build_multigraph_artifact(
            self._package.cache_entry(),
            self._collect_backends(),
            summary,
            self._backend,
            _entry_fn_of(self._fn),
        )


class _SingleFileStore(DynamoStore):
    """
    One artifact, one file.

    DiskDynamoStore treats its path as a directory and writes ``entry`` inside
    it, which is right for a content-addressed cache that owns its own layout.
    An artifact you name explicitly is a file you hand around, so this writes
    exactly the path given. Only the two storage primitives differ; everything
    above them is the shared DynamoStore.
    """

    def __init__(self, backend_artifacts: dict[_BackendId, Any] | None = None) -> None:
        self._backend_artifacts = {} if backend_artifacts is None else backend_artifacts

    def clear(self) -> None:
        self._backend_artifacts.clear()

    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None:
        from torch._dynamo.precompile_context import EagerCacheArtifact

        self._backend_artifacts[backend_id] = EagerCacheArtifact(
            key=backend_id, content=backend
        )

    def save_cache_entry(self, cache_entry: _DynamoCacheEntry, key: str) -> None:
        from torch._dynamo.precompile_context import (
            BackendCacheArtifact,
            PrecompileContext,
        )

        for backend_id in cache_entry.backend_ids:
            if backend_id not in self._backend_artifacts:
                artifact = PrecompileContext.take_artifact(backend_id)
                if artifact is not None:
                    self._backend_artifacts[backend_id] = artifact
        missing = [
            backend_id
            for backend_id in cache_entry.backend_ids
            if backend_id not in self._backend_artifacts
        ]
        if missing:
            raise RuntimeError(
                f"Backend {missing[0]} is not found in the given backends"
            )
        backends = {
            backend_id: self._backend_artifacts[backend_id]
            for backend_id in cache_entry.backend_ids
        }
        if not all(
            isinstance(artifact, BackendCacheArtifact) for artifact in backends.values()
        ):
            raise AssertionError("Expected BackendCacheArtifact")
        self.write(PrecompileCacheEntry(cache_entry, backends), key)

    def load_cache_entry(self, key: str) -> PrecompileCacheEntry:
        return self.read(key)

    def write(self, cache_entry: PrecompileCacheEntry, path: str) -> None:
        from torch._inductor.codecache import write_atomic

        if os.path.isdir(path):
            raise PackageError(
                f"{path} is a directory. precompile artifacts are single files; "
                f"pass the path you want the artifact written to."
            )
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            write_atomic(path, pickle.dumps(cache_entry))
        except Exception as e:
            raise PackageError(f"Failed to write artifact to {path}: {e}") from e

    def read(self, path: str) -> PrecompileCacheEntry:
        if os.path.isdir(path):
            raise PackageError(
                f"{path} is a directory. precompile artifacts are single files; "
                f"pass the path save() was given."
            )
        try:
            with open(path, "rb") as handle:
                entry = pickle.loads(handle.read())
        except Exception as e:
            raise PackageError(f"Failed to read artifact from {path}: {e}") from e
        if not isinstance(entry, PrecompileCacheEntry):
            raise PackageError(f"{path} does not hold a precompile artifact")
        return entry


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
    r"""Load an artifact and return a callable ready to serve it.

    The wiring is order-sensitive -- the package has to be attached to the
    optimize context before its globals and guarded codes are installed -- so it
    is done here rather than left to callers.

    Installing mutates global state on the underlying code objects, so the result
    is also a context manager that unloads on exit. ``torch._dynamo.reset()`` is
    not that unload: it destroys the precompile entries while leaving the
    installed globals in place, so a call after a reset raises rather than
    silently recompiling. Load the artifact again instead.

    Runtime guards remain intact for any compilation allowed outside
    ``serving()``. ``guard_filter_fn`` affects only its serialized package state.
    """
    entry_fn = _entry_fn_of(fn)
    store = _SingleFileStore()
    cache_entry = store.load_cache_entry(path)
    _check_artifact_matches(cache_entry.dynamo, entry_fn, path)
    return serve_cache_entry(
        fn,
        cache_entry,
        backend=backend,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
    )


def serve_cache_entry(
    fn: Callable[..., object],
    cache_entry: PrecompileCacheEntry,
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
) -> PrecompiledCallable:
    """Wire an already-loaded cache entry onto ``fn`` and install it.

    Shared by ``precompile_load``, which reads the entry from a path, and by the
    multi-graph artifact whose driver carries the same entry inline: the wiring
    is order-sensitive (the package has to be attached to the optimize context
    before its globals and guarded codes are installed), so it lives in one
    place rather than being written twice.
    """
    entry_fn = _entry_fn_of(fn)
    package = CompilePackage(
        entry_fn,
        cache_entry.dynamo,
        serialization_guard_filter_fn=(
            default_guard_filter_fn if guard_filter_fn is None else guard_filter_fn
        ),
    )
    optimize_ctx = torch._dynamo.optimize(
        _PrecompileBackend(backend),
        package=package,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        isolate_recompiles=True,
    )
    compiled = optimize_ctx(fn)
    isolate_recompiles_id = getattr(optimize_ctx, "_isolate_recompiles_id", None)
    if not isinstance(isolate_recompiles_id, int):
        raise AssertionError("missing isolate_recompiles_id")
    package.install(cache_entry.backends, isolate_recompiles_id=isolate_recompiles_id)
    return PrecompiledCallable(compiled, package, isolate_recompiles_id)


def precompile_capture(
    fn: Callable[..., object],
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
    example_inputs: Sequence[ExampleInput | tuple[object, ...]] | None = None,
    invariants: str | None = None,
    keep_only: frozenset[tuple[str, str]] | None = None,
    training: bool = False,
) -> PrecompileSession:
    r"""Begin capturing ``fn`` into a multi-graph artifact.

    ``recompile_limit`` defaults well above Dynamo's usual 8 because a
    precompile deliberately wants one compiled variant per condition, whereas
    the normal limit exists to catch runaway recompilation. It also raises a
    lower ambient ``accumulated_recompile_limit`` for this capture so that the
    explicit API limit is the effective one.

    ``example_inputs`` are run on ``__enter__``, under ``torch.no_grad()``, so
    the ``with`` body is optional for an inference capture; calls you make in
    the body run in the ambient grad mode instead. Existing inference tensors in
    the inputs or an ``nn.Module``'s parameters and buffers are rejected because
    disabling inference mode does not make them ordinary tensors. ``invariants``
    names a file written when the block exits without an exception. A session is
    one-shot; create another capture instead of re-entering it.

    Runtime guards remain intact during capture. ``guard_filter_fn`` applies
    only to the serialized guard state, so every supplied example observes the
    same recompilation behavior as ordinary ``torch.compile``. ``save()`` refuses
    the risky subset by default rather than every drop, and every guard a custom
    filter drops counts as risky.
    """
    return PrecompileSession(
        fn,
        backend=backend,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        example_inputs=example_inputs,
        invariants=invariants,
        keep_only=keep_only,
        training=training,
    )


class PrecompiledCallable:
    """A loaded artifact. Call it, or use it as a context manager to scope it."""

    def __init__(
        self,
        compiled: Callable[..., object],
        package: CompilePackage,
        isolate_recompiles_id: int,
    ) -> None:
        self._compiled: Callable[..., object] | None = compiled
        self._package = package
        self._isolate_recompiles_id = isolate_recompiles_id
        from .pgo import _new_code_state

        self._pgo_state = _new_code_state()
        self._state = threading.Condition()
        self._active_calls = 0
        self._unloading = False
        self._loaded = True
        from .eval_frame import _register_explicit_compile_region

        _register_explicit_compile_region(isolate_recompiles_id, self)

    def __call__(self, *args: object, **kwargs: object) -> object:
        with self._state:
            if not self._loaded or self._unloading:
                raise RuntimeError("PrecompiledCallable has been unloaded")
            if self._package.installed_entries_dropped():
                raise PackageError(
                    "torch._dynamo.reset() cleared the precompiled code this "
                    "artifact installed. The installed globals are still in "
                    "place, so this call would silently recompile instead of "
                    "serving, or fail with an indistinguishable RecompileError "
                    "under serving(). Load the artifact again after the reset."
                )
            compiled = self._compiled
            if compiled is None:
                raise AssertionError("loaded callable is missing")
            self._active_calls += 1
        # An uncovered no-op branch must become a guarded variant rather than
        # Dynamo's ordinary eager-only SkipFrame. Otherwise one fallback call
        # permanently skips that frame and later serving() cannot detect it.
        try:
            # _make_closure_patcher rather than config.patch(): patch() builds a
            # fresh ConfigPatch class and a fresh ContextVar on every call, and
            # this runs on every served call. The patcher is built once at import
            # and costs a set/restore pair.
            revert = _ALLOW_EMPTY_GRAPHS()
            try:
                with _use_code_state(self._pgo_state):
                    return compiled(*args, **kwargs)
            finally:
                revert()
        finally:
            with self._state:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._state.notify_all()

    def __enter__(self) -> Self:
        with self._state:
            if not self._loaded or self._unloading:
                raise RuntimeError("PrecompiledCallable has been unloaded")
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        """Remove installed globals and precompile entries from the code objects."""
        with self._state:
            while self._unloading:
                self._state.wait()
            if not self._loaded:
                return
            self._unloading = True
            try:
                while self._active_calls:
                    self._state.wait()
            except BaseException:
                self._unloading = False
                self._state.notify_all()
                raise
            self._loaded = False
        # uninstall() forgets which code objects it installed onto, and a frame
        # reached through code_source was installed onto the live code the
        # running program resolves rather than the reconstructed twin the
        # package holds, so the set to clear has to be taken first.
        codes = self._package.region_codes()
        try:
            self._package.uninstall()
        finally:
            try:
                _clear_package_region(codes, self._isolate_recompiles_id)
            finally:
                from .eval_frame import _unregister_explicit_compile_region

                _unregister_explicit_compile_region(self._isolate_recompiles_id)
                with self._state:
                    self._unloading = False
                    self._compiled = None
                    self._package.cached_backends.clear()
                    self._pgo_state.clear()
                    self._state.notify_all()


@contextlib.contextmanager
def serving() -> Iterator[None]:
    """
    Forbid compilation, so a call the artifact does not cover raises instead of
    quietly recompiling.

    Scoped to the CALLING THREAD, not the process: a serving process is
    multi-threaded by definition, and a process-wide switch made one request
    handler's block reject a concurrent handler's legitimate recompile. The
    corollary is that work handed to other threads is NOT covered --
    ``with serving(): pool.map(model, batches)`` leaves the workers free to
    recompile.

    It is not a property of the artifact either: it applies to every call made
    on this thread while it is active.
    """
    from .eval_frame import (
        _enter_fail_on_recompile_override,
        _exit_fail_on_recompile_override,
    )

    _enter_fail_on_recompile_override()
    try:
        yield
    finally:
        _exit_fail_on_recompile_override()


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

    Identity is (defining module, qualname, co_name, first line). The first
    line is what separates two definitions of one name in one module -- the
    class under an ``if`` that a config flag picks -- which every other field
    agrees on and which the source checksum passes, since both branches are in
    the file it hashes. co_filename is deliberately NOT compared: the capture
    and serving machines check out to different absolute paths, and the module
    name already carries what the path would say.
    """
    code = getattr(entry_fn, "__code__", None)
    if code is None:
        raise PackageError(f"{entry_fn!r} has no __code__ to load {path} onto.")
    if not dynamo.codes:
        raise PackageError(f"Artifact at {path} contains no code entries.")
    entry = dynamo.codes[0]
    actual_module = _defining_module_name(code) or getattr(entry_fn, "__module__", None)
    if actual_module is not None and entry.python_module != actual_module:
        raise PackageError(
            f"Artifact at {path} was captured from a callable defined in "
            f"{entry.python_module!r} but is being loaded onto one defined in "
            f"{actual_module!r}. Loading it would serve the captured "
            f"function's graphs for this one."
        )
    expected = dynamo.fn_name
    actual = getattr(entry_fn, "__qualname__", None)
    if expected is not None and actual is not None and expected != actual:
        raise PackageError(
            f"Artifact at {path} was captured from {expected!r} but is being "
            f"loaded onto {actual!r}. Loading it would serve the captured "
            f"function's graphs for this one."
        )
    stored = entry.python_code
    if stored.co_name != code.co_name:
        raise PackageError(
            f"Artifact at {path} was captured from code object "
            f"{stored.co_name!r} but is being loaded onto {code.co_name!r}."
        )
    if stored.co_firstlineno != code.co_firstlineno:
        raise PackageError(
            f"Artifact at {path} was captured from a definition at "
            f"{entry.python_module}:{stored.co_firstlineno} but the callable of "
            f"that name here is defined at line {code.co_firstlineno}. Same "
            f"name, different definition -- a class defined under an `if` that "
            f"a config flag or environment variable resolves the other way on "
            f"this machine looks exactly like this, and nothing else catches "
            f"it, because both branches are in the source the checksum covers. "
            f"A module whose lines merely shifted also lands here."
        )


@functools.singledispatch
def _entry_fn_of(fn: object) -> Callable[..., object]:
    if not callable(fn):
        raise TypeError(f"expected a callable or nn.Module, got {type(fn).__name__}")
    if not hasattr(fn, "__code__"):
        raise TypeError(
            f"expected a function or nn.Module, got {type(fn).__name__}, which "
            f"has no __code__ for Dynamo to capture or to load an artifact "
            f"onto. Pass partial.func for a functools.partial, or obj.__call__ "
            f"for an object that only defines __call__."
        )
    return fn  # type: ignore[return-value]


@_entry_fn_of.register
def _(fn: torch.nn.Module) -> Callable[..., object]:
    forward = fn.forward
    if not hasattr(forward, "__code__"):
        raise TypeError(
            f"{type(fn).__name__}.forward is a {type(forward).__name__}, which "
            f"has no __code__ for Dynamo to capture or to load an artifact onto. "
            f"Binding it in __init__ -- self.forward = functools.partial(...) -- "
            f"shadows the class method and lands here; keep forward a method."
        )
    return forward
