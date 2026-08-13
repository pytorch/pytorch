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
    session.save("snapshots/model.pt")

``example_inputs`` runs the calls for you, under ``torch.no_grad()``, so an
inference capture with nothing conditional in it is one statement, and
``invariants`` writes a readable report of what the capture established::

    with precompile_capture(
        model,
        backend="inductor",
        example_inputs=[(x1,), (x2,)],  # or ExampleInput(args, kwargs)
        invariants="model.invariants",
    ) as compiled:
        pass

Calls in the block body run in the caller's grad mode rather than under
no_grad, so a training capture -- one that calls ``.backward()`` -- goes there
rather than in ``example_inputs``, and passing one call both ways compiles it
twice, once per grad mode.

Per frame, that report separates the guards that held in EVERY compiled variant
from the ones that differed. The first set are the preconditions the artifact is
only valid under -- a call violating one cannot be served by any graph in it --
and each is marked enforced or dropped, so a precondition nothing rechecks at
load is visible rather than implied. The second set is what tells the compiled
graphs apart. Intersection is per frame because guards from different frames are
not comparable: the entry frame guards its arguments, a resume frame guards
whatever crossed the break. See ``PrecompileSession.invariants``.

Both path kwargs name a FILE and write exactly it, creating parent directories:
``snapshots/invariants.txt`` is a text file you can commit and diff, and
``save("snapshots/model.pt")`` is the artifact itself, handed straight back to
``precompile_load``. That is deliberately unlike ``DiskDynamoStore``, whose path
is a directory it fills, because a content-addressed cache owns its layout while
an artifact you name is a file you move around.

    # later, in a fresh process
    compiled = precompile_load(model, path, backend="inductor")
    with serving():  # no compilation permitted
        compiled(*args)

Capture is by execution: a resume function only exists once the frame ahead of
it has actually run, so every variant must be exercised. Whatever you do not
run is not in the artifact.

Know these before relying on an artifact in production:

* Capture inference artifacts under ``torch.no_grad()`` or
  ``torch.inference_mode()``. With ``backend="inductor"`` and parameters that
  require grad, AOTAutograd only records a bundled backend once the BACKWARD
  compiles, so a forward-only capture with grad enabled -- the default, and what
  ``model.eval()`` still leaves you in -- records no backends and cannot be
  saved. Capturing a training step that calls ``.backward()`` works too.
* A non-tensor argument, and any value that crosses a graph break, is guarded
  by equality, so an int/bool/str argument or a break coming from ``.item()``
  yields an artifact that only serves calls reproducing those exact values.
  ``summary().wont_generalize`` lists them; exercise every value you need to
  serve inside the capture block, or expect poor coverage on new data.
  ``dynamic=True`` helps with shapes but not with pinned values.
* Identity guards cannot be serialized, so precompiling gives up on noticing
  that a guarded object was rebound. ``summary().dropped_guards`` is the
  authoritative list and ``risky_dropped_guards`` is a lint over it, not a
  proof -- see ``_is_risky_drop`` for what it does and does not catch. The lint
  REPORTS by default rather than refusing, because real models trip it on
  library internals: measured on stock models, torchvision resnet18 and
  mobilenet_v3 report none, timm's ViT reports one (a re-exported
  ``torch._assert``) and transformers' Qwen2 reports 33 built from a two-layer
  config, 55 for the pretrained 24-layer, of which only the
  attention-implementation registry looks genuinely config-selected. Report
  counts are per model, not per library: torchvision's efficientnet_b0 reports
  2 and timm's swin reports 5, one of which is a real config slot. Refusing
  by default would refuse real models and train users to switch the check off,
  so audit the list once for your model and pass
  ``require_no_risky_drops=True`` to enforce it thereafter.
* A transformers model does not round trip today, and some do not even capture:
  T5 raises ``PackageError: Cannot find module for code <code object __init__``
  from ``_get_code_source``, which is byte-identical to base and which plain
  ``caching_precompile`` also raises, so it is upstream too. For the families
  that do capture, save and load work,
  but the first served call recompiles and ``serving()`` therefore raises. The
  entry frame is a ``functools.wraps`` decorator defined in
  ``transformers/utils/generic.py`` (``can_return_tuple`` for Qwen2), so its
  code object lives in that file; ``CompilePackage._add_function`` records
  the module as the decorated callable's ``__module__``
  (``...models.qwen2.modeling_qwen2``), so ``_install_codes`` hands the guards a
  global scope that is not the one the frame runs in and every ``G[...]`` guard
  misses with a ``KeyError``. This is not specific to this module -- plain
  ``torch._dynamo.config.caching_precompile`` records the same mapping -- so it
  needs fixing in the loader, not here. Vision models (torchvision, timm) do
  round trip.
* A guard can also be KEPT and yet stop discriminating, which no rail reports.
  Guards are rebuilt at load against the loading process, so one whose source
  resolves through a reconstructed function's ``__globals__`` re-derives its
  expected value from the serving machine and compares that value to itself.
  A global rebound between capture and serve is then absorbed silently and the
  capture-time graph is served. This is upstream in guard serialization, not
  specific to this module -- the same shape reproduces through the untouched
  ``<locals>`` reconstruction path -- but note it applies to KEPT guards, where
  ``dropped_guards`` and ``risky_dropped_guards`` say nothing. In practice such
  a function is itself an identity drop, so the risky list does name it.
* SystemInfo checks Python, PyTorch, CUDA, Triton and GPU name at load, but NOT
  the CPU vector ISA. Inductor bakes the vector width into generated CPU code,
  so an artifact captured on an AVX-512 host and served on an AVX2 host can
  produce wrong numbers with no error. Pin the ISA across your fleet, or gate
  on it yourself, before deploying CPU artifacts.
* The model must live in an importable module. Source is checksummed, so a
  class defined in ``__main__`` or a REPL cannot be loaded elsewhere.
* ``install()`` patches code objects process-globally, so an artifact is not
  scoped to the object it was loaded onto: other instances of the same class
  are served from it too, and ``torch._dynamo.reset()`` stops it serving,
  though the globals install() wrote stay in the module until ``unload()``. Two
  artifacts for ONE CLASS cannot be loaded at once: they collide on the entry
  frame, whose entries clear en masse, so the second load evicts the first,
  with a warning. Load one artifact per class per process.

  Two artifacts for DIFFERENT models that merely share an inner frame -- two
  models containing the same library block -- do coexist: entries accumulate
  on the shared code object and guards pick the right one. That holds only as
  long as the guards can still tell the two apart. Ship an artifact that
  dropped the discriminating guard -- which the default allows, the risky-drop
  lint only reporting -- and dispatch becomes ambiguous: the first matching
  entry wins and one model is silently served the other's graph. Nothing
  warns, because the eviction warning covers entry frames only.
* While an artifact is loaded, a plain ``torch.compile`` of anything else in
  the same module can die with ``AssertionError: Name '__builtins_dict___1'
  already exists in scope``. Loading installs that name -- a counter value from
  the capture process -- into the module, and a local compile mints from a
  counter that starts over here, so the two eventually collide. ``unload()``
  removes it; until then keep loaded and freshly compiled callables in
  separate modules.

This wraps CompilePackage, which is the low-level component and is not meant to
be used directly.

Everything here is private and deliberately unexported: reach it as
``torch._dynamo.precompile_package`` and expect the names to move. It is also
neither of the other two things torch calls "precompile" --
``torch.compiler.precompile`` is make_fx AOT capture to Python source, and
``torch._dynamo.config.caching_precompile`` is transparent caching of
``torch.compile`` artifacts, which drives the same CompilePackage machinery this
module wraps but automatically, without an explicit capture block.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import hashlib
import logging
import os
import pickle
import re
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING
from typing_extensions import Self

import torch
import torch._functorch.config as functorch_config
from torch._guards import ChainedSource, Source

from .exc import PackageError
from .guards import CheckFunctionManager
from .package import (
    _DynamoCacheEntry,
    CompilePackage,
    DynamoStore,
    PrecompileCacheEntry,
)
from .source import AttrSource, DictGetItemSource, GlobalSource


if TYPE_CHECKING:
    import traceback

    from .types import GuardFilterEntry


log = logging.getLogger(__name__)

# Not a public surface -- see the module docstring. This exists so `from ...
# import *` in a debugging session pulls the entry points rather than every
# private helper, and so linters do not flag them as unused.
__all__ = [
    "ExampleInput",
    "FrameInvariants",
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
    REPORTS the ones that look load-bearing rather than refusing them --
    enforcement is opt-in via ``require_no_risky_drops=True``. See
    ``risky_dropped_guards``.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]


@dataclasses.dataclass(frozen=True)
class ExampleInput:
    """One call to make during capture, when args alone are not enough."""

    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class _GuardFact:
    """
    One guard as it appeared in one compilation.

    Identity is every field: type, source, normalized code, value fingerprint
    and whether the guard was enforced. The fingerprint carries what the code
    does not -- a TENSOR_MATCH's shape lives in the C++ leaf, an identity
    guard's object behind a normalized id -- so the same guard specialized two
    ways is two facts, which is what makes them fall out of the intersection
    instead of collapsing into it. See ``_value_fingerprint``.
    """

    guard_type: str
    source: str
    code: tuple[str, ...]
    value: str
    enforced: bool

    def render(self) -> str:
        # A fingerprint exists only when the code does not already say what the
        # guard checks, so print it whenever there is one: an identity guard
        # does render code, but the id in it has been normalized away.
        body = " ; ".join(self.code) if self.code else f"<{self.guard_type}>"
        if self.value:
            body = f"{body} {self.value}"
        where = f" on {self.source}" if self.source else ""
        return f"[{'enforced' if self.enforced else 'dropped '}] {body}{where}"


@dataclasses.dataclass(frozen=True)
class FrameInvariants:
    """What held in every compiled variant of one frame, and what did not."""

    frame: str
    filename: str
    lineno: int
    variants: int
    invariant: tuple[_GuardFact, ...]
    varying: tuple[_GuardFact, ...]
    # Guards this module cannot compare between variants. Not "they held" and
    # not "they differed" -- we do not know, and saying either would be a claim
    # we cannot support.
    undetermined: tuple[_GuardFact, ...]


@dataclasses.dataclass(frozen=True)
class PrecompileSummary:
    """What a capture actually produced. Assert on this in a build step."""

    frames: int
    resume_functions: int
    guarded_codes: int
    backend_graphs: int
    bypassed: tuple[str, ...]
    # Frames that hit recompile_limit. A LOWER BOUND: the limit also puts every
    # frame called beneath the offender into run-only mode, and those never
    # re-enter Dynamo, so they go short without being named here.
    truncated: tuple[str, ...] = ()
    # Frames Dynamo produced but compiled nothing for. The entry frame landing
    # here means the model runs eager despite the artifact existing.
    uncovered_frames: tuple[str, ...] = ()
    # Sources pinned to an exact value by an equality guard -- see
    # ``_pins_a_value``. These make the artifact serve only the calls it was
    # captured with.
    wont_generalize: tuple[str, ...] = ()
    # (guard_type, source_name) for every guard the filter discarded / retained.
    dropped_guards: tuple[tuple[str, str], ...] = ()
    kept_guards: tuple[tuple[str, str], ...] = ()
    # Subset of dropped_guards whose loss can plausibly change results.
    risky_dropped_guards: tuple[tuple[str, str], ...] = ()

    @property
    def complete(self) -> bool:
        return (
            not self.bypassed
            and not self.truncated
            and not self.uncovered_frames
            and self.guarded_codes > 0
        )

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
            base += f", >={len(self.truncated)} TRUNCATED: {list(self.truncated)}"
        if self.bypassed:
            base += f", {len(self.bypassed)} BYPASSED: {list(self.bypassed)}"
        return base


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


def _is_library_module(module_name: str | None) -> bool:
    """
    Owned by torch or the stdlib, so config on the serving machine does not
    choose between implementations. NB this trusts the OWNER, not the binding:
    a third party that monkeypatches ``F.gelu`` at import time still diverges,
    and that is called out in ``_is_risky_drop``'s KNOWN GAP.
    """
    if module_name is None:
        return False
    if module_name == "torch" or module_name.startswith("torch."):
        return True
    return module_name.partition(".")[0] in sys.stdlib_module_names


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


class _SingleFileStore(DynamoStore):
    """
    One artifact, one file.

    DiskDynamoStore treats its path as a directory and writes ``entry`` inside
    it, which is right for a content-addressed cache that owns its own layout.
    An artifact you name explicitly is a file you hand around, so this writes
    exactly the path given. Only the two storage primitives differ; everything
    above them is the shared DynamoStore.
    """

    def clear(self) -> None:
        pass

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


# Guards whose check IS object identity, directly or through a derived guard,
# which is the same test default_guard_filter_fn drops on.
_IDENTITY_GUARD_TYPES = frozenset(
    CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
)


def _object_identity(value: object) -> str:
    """
    A stable stand-in for the id ``_normalize`` stripped.

    A name rather than an address, so the file still diffs clean across runs.
    Two objects of one type still collapse -- two Linear instances are
    indistinguishable here -- but the case that matters separates: one slot
    holding a different callable in different variants.
    """
    if isinstance(value, types.ModuleType):
        return f"is module {value.__name__}"
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(name, str):
        return _normalize(f"is {_owning_module(value) or '?'}.{name}")[:160]
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
        f"{type(example).__name__}. Wrap keyword arguments in ExampleInput."
    )


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
    wont_generalize = tuple(sorted({n for t, n in kept if _pins_a_value(t, n)}))
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
        example_inputs: Sequence[ExampleInput | tuple[object, ...]] | None = None,
        invariants: str | None = None,
    ) -> None:
        self._example_inputs = tuple(example_inputs or ())
        self._invariants_path = invariants
        self._fn = fn
        self._backend = backend
        self._dropped_guards: set[tuple[str, str]] = set()
        self._kept_guards: set[tuple[str, str]] = set()
        self._risky_dropped_guards: set[tuple[str, str]] = set()
        # (co_name, co_filename, co_firstlineno) -> one fact set per compilation
        self._guard_sets: dict[tuple[str, str, int], list[frozenset[_GuardFact]]] = {}
        self._undetermined: dict[tuple[str, str, int], set[_GuardFact]] = {}
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
        # Capture is by execution, so example_inputs is just "run these for
        # me", except that they are forced under no_grad: they are the
        # inference path, and save() rejects a grad-enabled forward-only
        # capture. Done before yielding so the body can add calls on top.
        try:
            for example in self._example_inputs:
                args, kwargs = _example_call(example)
                with torch.no_grad():
                    self._compiled(*args, **kwargs)
        except BaseException:
            # A __enter__ that raises never gets its __exit__, so without this
            # the config patch above stays on for the life of the process and
            # the session is wedged: save() reports the block as still open.
            self._stack = None
            stack.close()
            raise
        return self._compiled

    def __exit__(self, *exc: object) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
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
            facts: set[_GuardFact] = set()
            undetermined: set[_GuardFact] = set()
            for keep, entry in zip(decisions, entries):
                target = self._kept_guards if keep else self._dropped_guards
                target.add((entry.guard_type, entry.name))
                if not keep and _is_risky_drop(entry, namespaces):
                    self._risky_dropped_guards.add((entry.guard_type, entry.name))
                unmodelled = entry.guard_type in _UNMODELLED_GUARD_TYPES
                fact = _GuardFact(
                    guard_type=entry.guard_type,
                    source=_normalize(entry.name),
                    code=_render_code(entry.orig_guard.code_list),
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
        no value of their own, so two variants that differ only in, say,
        autocast render identically and report as invariant with nothing
        varying. GRAD_MODE is fingerprinted because the capture path itself
        produces that split.
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
            "# dropped  = it could not be serialized, so it is a precondition",
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
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        log.info(
            "precompile: wrote invariants for %d frame(s) to %s", len(frames), path
        )

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
        require_no_risky_drops: bool = False,
        require_no_dropped_guards: bool = False,
    ) -> PrecompileSummary:
        """
        Write the artifact, refusing by default to write one that cannot serve
        what it claims: ``require_complete`` gates exactly
        ``PrecompileSummary.complete``. The two guard rails are separate from
        it, because a dropped guard is a wrong answer rather than a gap in
        coverage, and both are off: every capture drops identity guards, and
        the risky subset REPORTS through ``log.warning`` because real models
        trip the lint on library internals. Audit
        ``summary().risky_dropped_guards`` once for your model, then pass
        ``require_no_risky_drops=True`` to hold it.

        ``path`` is a FILE, written exactly as given with parent directories
        created, and ``precompile_load`` takes it straight back. Same contract
        as ``invariants``.
        """
        if self._stack is not None:
            raise RuntimeError("save() must be called after the capture block exits")
        summary = self.summary()
        if summary.risky_dropped_guards and require_no_risky_drops:
            raise PackageError(
                f"Precompilation dropped identity guard(s) on "
                f"{[n for _, n in summary.risky_dropped_guards]}. Each of those names "
                f"a slot whose occupant config chooses -- an instance attribute, a "
                f"closure cell, an aliased or cross-module import -- and identity "
                f"guards cannot be serialized, so nothing checks them at load time. "
                f"Identical source "
                f"is not enough: if config, a feature flag, or an environment variable "
                f"selects a different object on the serving machine, the artifact "
                f"serves the graph traced against the capture-time object and returns "
                f"a wrong answer with no error. Make the value reachable as data the "
                f"graph can guard, pin it so both machines agree, or drop "
                f"require_no_risky_drops=True -- reporting is the default -- to "
                f"accept the risk."
            )
        elif summary.risky_dropped_guards:
            # Advisory, not a gate. Measured on stock models: torchvision
            # resnet18 and mobilenet_v3 report none, timm's ViT reports one (a
            # re-exported torch._assert) and transformers' Qwen2 reports 33,
            # nearly all of them library internals that no deployment swaps.
            # Refusing by default would refuse real models and teach users to
            # switch the check off, so it reports and enforcement is opt-in.
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
                "precompile: %d dropped identity guard(s) sit at slots config can "
                "choose, so nothing checks them at load: %s.%s Audit them against "
                "your deployment and pass require_no_risky_drops=True to enforce.",
                len(names),
                names[:8],
                rest,
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
                    f"Precompilation produced no compiled code for entry frame(s) "
                    f"{list(summary.uncovered_frames)}, so install() will skip them "
                    f"and they will run eager -- and because a skipped frame never "
                    f"compiles, serving() cannot report the gap either. This is "
                    f"expected when the frame only dispatches to submodules that ARE "
                    f"covered; it also looks exactly like a frame Dynamo gave up on "
                    f"(check TORCH_LOGS=graph_breaks for gb0124). Pass "
                    f"require_complete=False once you have confirmed which."
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
        store = _SingleFileStore()
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
    example_inputs: Sequence[ExampleInput | tuple[object, ...]] | None = None,
    invariants: str | None = None,
) -> PrecompileSession:
    """
    Begin capturing ``fn`` into a multi-graph artifact.

    ``recompile_limit`` defaults well above Dynamo's usual 8 because a
    precompile deliberately wants one compiled variant per condition, whereas
    the normal limit exists to catch runaway recompilation.

    ``example_inputs`` are run on ``__enter__``, under ``torch.no_grad()``, so
    the ``with`` body is optional for an inference capture; calls you make in
    the body run in the ambient grad mode instead. ``invariants`` names a file
    written when the block exits without an exception.
    """
    return PrecompileSession(
        fn,
        backend=backend,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        example_inputs=example_inputs,
        invariants=invariants,
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
    store = _SingleFileStore()
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
    actual_module = getattr(entry_fn, "__module__", None)
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
