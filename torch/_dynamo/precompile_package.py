"""
Helpers behind the multi-graph precompile: the capture of a callable into one
artifact holding every frame Dynamo produces while the caller's calls run --
the entry frame, the ``torch_dynamo_resume_in_*`` continuations graph breaks
create, and the recompiled variants of each -- stored through CompilePackage
(``torch/_dynamo/package.py``), a low-level component not meant to be used
directly, the multi-graph counterpart of
``torch.compile(fn, fullgraph=True).aot_compile(...)``. It is not
``torch.compiler.precompile``, the ahead-of-time capture API this repository
already has (``torch/_precompile.py``), which does not call into this module
yet; nor ``torch._dynamo.config.caching_precompile``, which caches
``torch.compile`` artifacts transparently without an explicit capture and, when
set, wraps every guard filter, this module's included (see
``default_guard_filter_fn``).

``default_guard_filter_fn`` is the guard filter a capture's serialized guards
are written under. Around it sits the guard tooling that reports what that
filter dropped and the configuration a capture runs under. The filter lives
here, with that tooling, rather than beside the serializer's pre-check in
``guards.py``: it is the capture's policy over that pre-check, not part of it.
Everything here is internal; the filter alone is unprefixed because the capture
session passes it as the default a caller may name. ``PrecompileSession``, the
multi-graph Dynamo capture session that drives it, is added here; the
``torch.compiler.precompile.capture(..., tracer=DynamoTracer())`` entry point
that reaches it lands later in this stack, so nothing under ``torch/`` calls
into the session yet.

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
guarded variant it exercises.

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
  is still not a proof for unobserved deployments. The
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

The public surface is ``torch.compiler.precompile.capture(...)``, a caller-driven
capture used as a context manager: the caller's own calls inside the block drive
the capture, and the ``(python_code, cache)`` artifact is written to the given files
when the block exits (its default ``tracer=DynamoTracer()`` records many calls;
``tracer=MakeFxTracer()`` produces a self-contained Python source artifact from one
call); and ``torch.compiler.precompile.load``.
The helpers in this module, including the capture session, implement that surface
and remain internal.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import importlib.machinery
import logging
import os
import re
import site
import sys
import sysconfig
import threading
import types
from typing import Any, TYPE_CHECKING

import torch
import torch._functorch.config as functorch_config
from torch._guards import ChainedSource
from torch.compiler._precompile_types import (
    FrameInvariants,
    GuardFact as _GuardFact,
    PrecompileSummary,
)
from torch.utils._config_module import ConfigModule

from .aot_compile import _BUILTINS_DICT_PREFIX, _IMPORT_ALIAS_PREFIX
from .convert_frame import CatchErrorsWrapper
from .exc import PackageError
from .guards import CheckFunctionManager
from .package import CompilePackage
from .source import (
    AttrSource,
    DictGetItemSource,
    GetItemSource,
    GlobalSource,
    LocalSource,
)


if TYPE_CHECKING:
    import traceback
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from torch._guards import Source

    from .convert_frame import ConvertFrameReturn
    from .eval_frame import OptimizeContext
    from .package import _BackendId, _DynamoCacheEntry
    from .types import CacheEntry, DynamoFrameType, GuardFilterEntry
    from .variables.builder import FrameStateSizeEntry


log = logging.getLogger(__name__)


# Not a public surface -- see the module docstring. This exists so `from ...
# import *` in a debugging session pulls the entry points rather than every
# private helper, and so linters do not flag them as unused.
__all__ = [
    "default_guard_filter_fn",
    "FrameInvariants",
    "PrecompileSession",
    "PrecompileSummary",
    "precompile_capture",
]


def default_guard_filter_fn(guard_entries: Sequence[GuardFilterEntry]) -> list[bool]:
    """
    Drop every guard ``CheckFunctionManager.serialize_guards`` would refuse for
    its type or a derived type, and keep everything else.

    The refused types are ``UNSUPPORTED_SERIALIZATION_GUARD_TYPES``: the
    identity guards ID_MATCH, FUNCTION_MATCH, MODULE_MATCH, NN_MODULE,
    CLASS_MATCH and CLOSURE_MATCH (a function by its ``__code__`` id), plus
    DICT_VERSION and WEAKREF_ALIVE. Dropping one gives up on noticing that the
    guarded object was rebound, mutated or collected: rebind a global function
    between capture and load and the artifact serves the graph traced against
    the old one, with no error
    (``test_default_guard_filter_through_serialize_guards``). Every dropped
    slot is reported in ``PrecompileSummary.dropped_guards``, once however many
    variants dropped it.

    The criterion is the pre-check's own: a guard is dropped if its type is
    refused or a derived type is (a CONSTANT_MATCH on a code object runs
    through ID_MATCH), and TYPE_MATCH and BUILTIN_MATCH are kept whatever they
    derive, as the pre-check accepts them before it looks at derived types.
    That keeps BUILTIN_MATCH, an ``id_match_unchecked`` deriving ID_MATCH that
    the loaded artifact still checks against the loading process's builtins.

    A DICT_VERSION derived by a DICT_KEYS_MATCH is ignored. That compensates
    for a ``CheckFunctionManager`` artifact, not a property of the serializer:
    a filter sees the derived types of the pre-filter build, which runs with
    ``save_guards=False`` and so promotes the DICT_KEYS_MATCH on
    ``torch.utils._pytree.SUPPORTED_NODES`` to DICT_VERSION, while the save
    build keeps the DICT_KEYS_MATCH the pre-check accepts
    (``test_default_guard_filter_keeps_the_pytree_registry_keys_match``). A
    ``guards.py`` fix giving both builds one verdict would delete the two
    lines below. A DICT_KEYS_MATCH deriving another refused type, and any other
    type deriving DICT_VERSION, are dropped. Neither this keep nor the
    accepted-by-type branch holds under
    ``torch._dynamo.config.caching_precompile``: its wrapper around every guard
    filter trips over the same artifact and drops, with a warning, any guard
    of type ID_MATCH, CLOSURE_MATCH, WEAKREF_ALIVE or DICT_VERSION or deriving
    ID_MATCH or DICT_VERSION.

    Passing this filter does not mean the artifact serializes: the pre-check
    also refuses a kept TYPE_MATCH on a local-scope type, which cannot be
    pickled by name. The filter keeps such a guard on purpose, so serialization
    fails loudly; dropping the TYPE_MATCH on an ``nn.Module`` of a local class
    would ship an artifact that serves a module of any class whose guarded
    attributes match
    (``test_default_guard_filter_keeps_local_type_guards_for_a_loud_refusal``).
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    keep = []
    for g in guard_entries:
        derived = g.derived_guard_types
        if g.guard_type == "DICT_KEYS_MATCH":
            derived = tuple(d for d in derived if d != "DICT_VERSION")
        keep.append(
            # The pre-check's accepted-by-type pair, a literal in serialize_guards,
            # in test_aot_compile.py's keep_builtin_guards and in
            # test_precompile_package.py's _pre_check_accepts too; a type added
            # to one is not seen by the others.
            g.guard_type in ("TYPE_MATCH", "BUILTIN_MATCH")
            or (
                g.guard_type not in unsupported
                and not any(d in unsupported for d in derived)
            )
        )
    return keep


def _owning_module(value: object) -> str | None:
    if isinstance(value, types.ModuleType):
        return value.__name__
    owner = getattr(value, "__module__", None)
    return owner if isinstance(owner, str) else None


def _source_root(source: Source) -> Source:
    return source.get_base() if isinstance(source, ChainedSource) else source


# The list of Dynamo-generated resume functions every generated resume function
# takes as its first parameter (resume_execution.py and
# comprehension_graph_break.py mint the name; codegen_call_resume in
# symbolic_convert.py builds the list). Its entries are generated code, not a
# slot any config chooses, so an identity guard lost on one cannot diverge.
# Only the list and an entry read straight off it are covered: a resume
# function's closure cells carry the resumed frame's cell variables, so a guard
# Dynamo mints past an entry (type(__nested_resume_fns[0].__closure__[0]
# .cell_contents).__call__ for a callable an inner def captured) is the user's
# and is judged like any other value. Its sibling __nested_frame_values is NOT
# here: it carries the live stack and locals of the frames nested INSIDE the
# one resuming, which pops the last entry off it for the callee it resumes, so
# a guard rooted there is judged like the value it stands for.
_DYNAMO_SYNTHESIZED = ("__nested_resume_fns",)


def _is_dynamo_synthesized(source: Source) -> bool:
    if isinstance(source, GetItemSource) and isinstance(source.index, int):
        source = source.base
    return isinstance(source, LocalSource) and source.local_name in _DYNAMO_SYNTHESIZED


def _norm(path: str) -> str:
    """
    realpath then normcase. A relative path resolves against the process cwd,
    so a recorded ``__file__`` is gated with isabs before it gets here and every
    root candidate comes through ``_norm_absolute``; this module's own
    ``__file__`` is taken as read because it is always the path finder's
    spelling, absolute since bpo-43105 (3.10+): the finder that keeps a relative
    one, zipimport, cannot load torch (torch._C is an extension module), and a
    frozen torch has no ``__file__`` at all. os is different: frozen since 3.11,
    its ``__file__`` is spelled from sys._stdlib_dir, relative under a relative
    home until site.abs_paths() re-anchors it at startup, which -S skips.
    """
    return os.path.normcase(os.path.realpath(path))


def _norm_absolute(paths: Iterable[object]) -> set[str]:
    """
    The ``_norm`` of every absolute str among ``paths``; anything else is
    dropped rather than resolved. The interpreter's own metadata can be
    relative: a venv whose pyvenv.cfg ``home`` is relative or a relative
    PYTHONHOME leaves sys.base_prefix, sys._stdlib_dir and every sysconfig
    path relative (os.__file__ alone is re-anchored, by site.abs_paths()),
    a relative PYTHONUSERBASE the user site. Resolved, such a root would sit
    wherever the process cwd was at the first call and stay cached there,
    so a writer and a reader on one interpreter could classify differently.
    """
    return {_norm(p) for p in paths if isinstance(p, str) and os.path.isabs(p)}


@functools.cache
def _stdlib_roots() -> tuple[str, ...]:
    """
    Where this interpreter's own library lives, sorted, so the first root a path
    lies under is the outermost (``_classify_file`` reads it that way). os is
    unquestionably stdlib, so its directory is the direct evidence and the only
    one that stays right when the stdlib is a zip, where that root is the whole
    archive and a third party bundled into it is waived with the stdlib;
    sysconfig and sys._stdlib_dir cover a build where os is frozen with no
    __file__ (``_classify_file`` skips the stdlib arm under ``sys.frozen``, so
    an app bundle does not use them). An install root can nest inside one of
    these (see ``_install_roots``), so a path under both is third party: an
    install root wins over a stdlib root.
    """
    roots: list[object] = []
    os_file = getattr(os, "__file__", None)
    if isinstance(os_file, str) and os.path.isabs(os_file):
        # The directory the file resolves into, not the one it was imported
        # from: in a venv over a symlink-farm prefix (a Nix, Guix or Spack
        # profile) os.py is a per-file link into the store, sysconfig and
        # sys._stdlib_dir already name the farm, and every consumer normalizes
        # the file it asks about.
        roots.append(os.path.dirname(_norm(os_file)))
    roots.append(getattr(sys, "_stdlib_dir", None))  # 3.11+
    paths = sysconfig.get_paths()
    roots += [paths["stdlib"], paths["platstdlib"]]
    if sys.platform == "win32":
        # The stdlib's C extensions live beside Lib, not under it.
        roots.append(os.path.join(sys.base_prefix, "DLLs"))
    return tuple(sorted(_norm_absolute(roots)))


@functools.cache
def _install_roots() -> tuple[str, ...]:
    """
    Where a third party lands. This is the load-bearing exclusion: purelib is
    NESTED inside stdlib in a conda layout and inside platstdlib in a venv (on
    a --with-platlibdir=lib64 build it is platlib that nests, purelib living
    under lib instead), so without it every pip-installed package is under a
    stdlib root.
    """
    paths = sysconfig.get_paths()
    roots: list[object] = [paths["purelib"], paths["platlib"]]
    for name in ("getsitepackages", "getusersitepackages"):
        try:
            got = getattr(site, name)()
            roots += [got] if isinstance(got, str) else list(got)
        except Exception:
            continue  # an old-virtualenv site.py lacks it, or it cannot answer
    # On Windows getsitepackages() lists the bare prefix, which the whole stdlib
    # sits under; a directory a stdlib root lies strictly under is not an
    # install root. A candidate that IS a stdlib root stays one, on purpose:
    # dropping it would leave a third party installed into the stdlib directory
    # itself under no install root and waived with the stdlib, while keeping it
    # only reads the stdlib as third party.
    stdlib = _stdlib_roots()
    normed = _norm_absolute(roots)
    above = {r for r in normed for s in stdlib if s.startswith(r + os.sep)}
    return tuple(sorted(normed - above))


@functools.cache
def _torch_roots() -> tuple[str, ...]:
    """
    Every directory torch's own submodules come from. An editable build splits
    them -- torch/__init__.py out of the source tree, _C.so and version.py out
    of site-packages -- and torch.__path__ is exactly that set. The gate rules
    out a substituted sys.modules['torch'] only: its __path__ is ignored unless
    the torch package directory this file sits under (two levels up, past
    _dynamo) is among the entries, and then every absolute entry is adopted,
    one a third party appended to the real torch's included; a relative entry
    is dropped (``_norm_absolute``). That directory has two spellings, both of
    them roots: resolved as a directory, and two levels up from where this file
    resolves. They differ in a per-file symlink farm (see ``_stdlib_roots``),
    where torch.__path__ names the farm and every consumer asks about a file
    that resolves into the store. The second is taken only while the file still
    resolves to <root>/_dynamo/<file>: a link that flattens the depth would make
    an ancestor of unrelated code a torch root, and then the resolved file lies
    under no root rather than under too wide a one.
    """
    own_file = globals().get("__file__")
    if not own_file:
        return ()  # frozen torch: no directory to anchor to
    own_dir = os.path.dirname(own_file)
    own = os.path.dirname(own_dir)
    roots = {_norm(own)}
    resolved = _norm(own_file)
    tail = os.path.join(os.path.basename(own_dir), os.path.basename(own_file))
    if resolved.endswith(os.sep + os.path.normcase(tail)):
        roots.add(os.path.dirname(os.path.dirname(resolved)))
    search = getattr(sys.modules.get("torch"), "__path__", None) or ()
    listed = _norm_absolute(search)
    if roots & listed:
        roots |= listed
    return tuple(sorted(roots))


def _within(path: str, roots: tuple[str, ...]) -> bool:
    """Prefix test over ``_norm``-ed paths; the caller normalizes both sides."""
    return any(path == r or path.startswith(r + os.sep) for r in roots)


# Belt and braces for an install directory none of the roots name: whichever
# layout put it there, a pip target still ends in one of these. Matched below
# the stdlib root a file was found under, not on the whole path, so a stdlib
# that itself sits under a site-packages directory keeps its waiver.
_INSTALL_DIR_NAMES = frozenset({"site-packages", "dist-packages"})


@functools.cache
def _classify_file(file: str, stdlib: bool) -> bool | None:
    """
    Shipped here (True), shipped elsewhere (False), or no evidence (None), for
    one ``__file__`` judged against the torch roots (``stdlib=False``) or the
    stdlib roots minus the install roots (``stdlib=True``). None is a path that
    cannot be resolved, or any path asked about for the stdlib in a frozen app;
    past those gates the torch arm never answers None: with no torch root (a
    frozen torch) every path is elsewhere, and ``_is_library_module`` waives
    torch names before asking. Cached on the (file, flag) pair rather than on a
    module name: everything else this reads, the roots and ``sys.frozen``, is
    fixed once the process is running, so the answer for a path never changes,
    while the module a name resolves to can (a test that patches either clears
    the cache on both sides of the patch).
    """
    # Before 3.13 ntpath.isabs accepts a driveless \Lib\x.py (its LEGACY BUG
    # comment), which realpath then resolves against the current drive.
    driveless = sys.platform == "win32" and not os.path.splitdrive(file)[0]
    if driveless or not os.path.isabs(file):
        # Resolving it would be against a cwd that is not the one it was
        # recorded under, so it is evidence in neither direction.
        return None
    if "\x00" in file:
        # No file has one. posixpath.realpath raises ValueError on it from lstat;
        # from gh-106242 on, ntpath.realpath swallows that and returns the path
        # unresolved, whose prefix _within would then match.
        return None
    path = _norm(file)
    if not stdlib:
        # Not gated on sys.frozen: the torch root is torch's own package
        # directory, taken from where this file sits, so in a bundle nothing
        # outside torch lies under it, whereas the stdlib root below is the
        # bundle root itself.
        return _within(path, _torch_roots())
    if getattr(sys, "frozen", False):
        # A frozen app bundles the stdlib and every third party under one root
        # with no site-packages component between them, and PyInstaller names
        # that root in sys._stdlib_dir and in the __file__ it gives the
        # CPython-frozen modules (its _fixup_frozen_stdlib), so every bundled
        # file would lie under a stdlib root: no path there says which of the
        # two a file is. _located's loader arm still waives what the interpreter
        # binary itself carries. Truthiness, as multiprocessing.spawn reads it:
        # PyInstaller and cx_Freeze set the flag to True, py2exe to a string
        # ("console_exe", "windows_exe" or "dll").
        return None
    if _within(path, _install_roots()):
        return False
    # The roots are sorted, so the first match is the outermost and the part
    # below it the longest: the strictest reading of _INSTALL_DIR_NAMES. Case
    # folded because normcase is the identity on posix, and a macOS filesystem
    # is case-insensitive.
    for root in _stdlib_roots():
        if _within(path, (root,)):
            below = path[len(root) + 1 :].lower().split(os.sep)
            return _INSTALL_DIR_NAMES.isdisjoint(below)
    return False


def _located(module: object, name: str, stdlib: bool) -> bool | None:
    """
    Shipped here (True), shipped elsewhere (False), or neither (None), for a
    sys.modules entry, which need not be a module. None covers no ``__file__``
    and no known loader, a ``__file__`` that ``_classify_file`` cannot place
    (relative, non-string, NUL, or any stdlib path in a frozen app) with no
    known loader, and a read below that raised; ``_is_library_module`` refuses
    None for a top-level name and keeps the package's waiver for an inner one,
    so an unplaceable inner ``__file__`` counts as no evidence there. A
    ``__file__`` that ``_classify_file`` can place decides, and only then is
    the loader read: with importlib on disk, importlib/__init__.py gives
    importlib._bootstrap a
    stdlib ``__file__`` next to a FrozenImporter whose table knows it only as
    _frozen_importlib (with importlib frozen too it has no ``__file__``, and
    the loader arm answers False under the dotted name: a lost waiver, not a
    wrong drop). The built-in and frozen arms answer on the caller's ``name``
    alone, not the module's ``__name__``, under either ``stdlib``: both importers
    precede the path finder, so no sys.path entry can shadow such a module,
    and its code is fixed by the interpreter binary, which is what either
    waiver needs. The ``stdlib=False`` case is an embedding that registers
    torch._C through PyImport_AppendInittab while torch's Python files stay on
    disk, so ``_torch_roots`` is non-empty and ``_is_library_module`` does not
    waive torch on its name.
    """
    # The module dict rather than getattr: a class attribute says nothing about
    # where the module came from (torch.ops is a ModuleType subclass carrying
    # the class's relative "_ops.py" as its __file__), and on an entry whose
    # dict is missing one of these dunders getattr would fall through to a PEP
    # 562 module __getattr__, user code a lint must not run (a raise is caught
    # below, its side effects are not). And object's read of the dict, not the
    # module's: importlib.util.LazyLoader leaves a _LazyModule whose
    # __getattribute__ executes the module body on ANY attribute read, __dict__
    # included, while its dict already carries the seeded __file__, __spec__ and
    # __loader__. That read defeats a __getattr__ or a __getattribute__
    # override, not a __dict__ descriptor defined on the type. sys.modules can
    # hold any object, so such a descriptor, and spec.loader on a hand-rolled
    # spec, are user code, and are caught, as is the AttributeError object
    # itself raises for a slotted entry with no __dict__ at all. The try also
    # spans _classify_file and the loader arms, so a raise out of the
    # classifier's own gates or out of FrozenImporter.find_spec (ImportError on
    # an excluded or invalid frozen table entry) lands here as None rather than
    # escaping a lint.
    try:
        attrs = object.__getattribute__(module, "__dict__")
        file = attrs.get("__file__")
        # Only a str is classifiable: a bytes or PathLike __file__ would pass
        # isabs and raise from the NUL check instead.
        if isinstance(file, str):
            verdict = _classify_file(file, stdlib)
            if verdict is not None:
                return verdict
        # The loader rather than spec.origin: both importers' find_spec pass
        # origin=cls._ORIGIN to spec_from_loader, so the two never disagree,
        # and the class is the stronger signal. A __loader__ in the dict skips
        # the spec's loader, the only user-code half of this; compared with None
        # rather than for truth, so no __bool__ of user code runs. A hand-made
        # ModuleType has the None its __init__ seeds and falls through (an
        # imported module's dict carries spec.loader instead, copied in by
        # _init_module_attrs).
        loader = attrs.get("__loader__")
        if loader is None:
            loader = getattr(attrs.get("__spec__"), "loader", None)
        # Both arms compare by identity: under == a __loader__ whose __eq__
        # answers true for anything would take the waiver.
        if loader is importlib.machinery.BuiltinImporter:
            # Statically linked, and BuiltinImporter precedes PathFinder on
            # sys.meta_path, so on import no file on sys.path is reachable
            # under this name; a spec assigned straight into sys.modules is
            # taken at its word. The inittab is keyed on the full dotted name.
            return name in sys.builtin_module_names
        if loader is importlib.machinery.FrozenImporter:
            # frozen also precedes the path finder
            return importlib.machinery.FrozenImporter.find_spec(name) is not None
    except Exception:
        return None
    return None  # namespace package, or exec'd in memory with no loader


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
    path finder cannot shadow. That is required of the TOP-LEVEL name: not
    imported, or without location evidence (a namespace package has none), it
    is untrusted. An imported inner name only has to not be located ELSEWHERE:
    the package it was found in is already located, and a real submodule can
    carry no evidence of its own: pyexpat.errors, which pyexpat's C init
    registers with neither ``__file__`` nor ``__spec__``, and torch.ops, a
    ModuleType subclass whose ``__file__`` is a class attribute the module dict
    never sees. The torch arm alone has an escape hatch: a frozen torch gives
    ``_torch_roots`` no directory to anchor to, and then every torch name is
    waived on its name, with no location evidence at all.
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
        # Unimported or unlocatable, an inner name has nothing to check, and
        # the package it would have to be found in has already been located.
        if module is not None and _located(module, name, stdlib) is False:
            return False
    return True


def _defined_where_read(
    value: object, global_name: str, user_stack: traceback.StackSummary | None
) -> bool:
    """
    Whether ``global_name`` is a def or class statement of that name living in
    the file that read it.

    ``global_name`` is the read's ``GlobalSource.global_name``; the
    ``GuardFilterEntry.name`` spelling keeps its ``G[...]`` wrapper and is not
    it. The reading file is the OUTERMOST frame of the guard's ``user_stack``:
    a bare GlobalSource denotes the root frame's globals (an inlined frame with
    other globals reads through an ``__import_`` alias or an
    ``___unnamed_scope`` dict entry instead), while the stack is stamped at
    first use, so its innermost frame can be a helper inlined from another
    file. A def bound under its own name in the reading file is the one binding
    the inlined-source checksum of that file covers. ``from impl_a import op``
    takes only a conditional import in the reader, which no checksum sees, and
    ``act = _impl_a if cfg.fast else _impl_b`` is a slot however close to home
    the def is; so are ``op = Ops.op`` and a def returned by a factory, which is
    why the name compared is ``__qualname__``, backed by the code object's own
    name: functools.wraps copies ``__qualname__`` onto a wrapper but cannot
    forge its ``co_qualname``, so a same-file ``wraps`` decorator a flag turns
    on is a slot in both arms (before 3.11 only ``co_name`` exists, so a
    wrapper def named after what it wraps slips through there). Only a plain
    function or a class is judged; a bound method or any other proxy that
    forwards ``__qualname__`` and ``__code__`` is refused before an attribute
    is read, which also keeps a proxy that answers reads by raising, such as
    ``torch.classes.<ns>``, out of the value slot. The file is read off the
    code object, not off ``__module__``: functools.wraps copies ``__module__``
    along with ``__name__`` and ``__qualname__``, so ``op = torch.compile(op)``
    behind a flag claims the reader's module while its code lives in
    eval_frame.py. The object does not tell that shape from an unconditional
    cross-file decorator, so ``@torch.no_grad()`` on a same-file def is not
    waived either.
    A class has no code object, and its ``__module__`` is no better: namedtuple
    and ``type()`` stamp it from the calling frame (make_dataclass does from
    3.12) under a BARE ``__qualname__`` (a def or class statement inside the
    factory would carry ``factory.<locals>.``), so a ``namedtuple("Point", ...)``
    called in the reader looks exactly like a class statement -- the frame
    stamped is the caller's, so the collision needs a factory called from the
    reading file, a cross-file ``lib.make_point()`` coming back stamped ``lib``.
    Its methods can tell: a class statement compiles its defs in its own file
    under its own ``__qualname__`` prefix, so a class is waived when at least one
    function in its own ``__dict__``, stored under key ``k`` with
    ``__qualname__`` ``Cls.k``
    (a staticmethod or classmethod is unwrapped through ``__func__`` and a
    property through ``fget``, by type rather than by ``getattr``, which a proxy
    attribute such as ``torch.classes.<ns>`` answers by raising; a
    cached_property keeps its function under ``.func`` and does not count), was
    compiled in the reading file. A function attached afterwards keeps its bare
    qualname, and one attached through functools.wraps keeps its code object's
    own name, so an imported class the reader extends (``Point.extra =
    _extra``) or patches (``Point.norm = wraps(Point.norm)(_norm)``) and a
    factory fed same-file methods (``type(name, bases, {"area": _area})``,
    ``make_dataclass(..., namespace=...)``) are refused, and ``class Marker:
    pass`` fails closed. So does a class statement whose only functions are
    generated: a fields-only ``@dataclass``'s and a ``NamedTuple``'s are defs
    of a factory (dataclasses' ``__create_fn__``, ``namedtuple``), so their
    code objects' own qualname carries the factory's ``<locals>.`` prefix (on
    3.10, where only ``co_name`` exists, the ``<string>`` or stdlib file they
    compile in refuses them instead), and an ``Enum``'s arrive in the subclass
    ``__dict__`` under ``Enum.`` qualnames the key rule refuses; so a plain
    config dataclass read as a global is reported. The one function the
    compiler itself puts in a class ``__dict__``, the PEP 649 annotate function
    3.14 stores for an annotated class body, compiles in the reading file, but
    under key ``__annotate_func__`` with ``__qualname__`` ``Cls.__annotate__``,
    so the key rule refuses it and the verdict is the same on every version;
    both annotate keys are skipped outright as well, against a version that
    stores it under its own name. Nothing else without a code object is
    waived, because a C-implemented wrapper such as functools.lru_cache claims
    the reader's module the same way. A ``co_filename`` is not always a path:
    an exec records ``<string>``, a REPL ``<stdin>``, and ``_norm`` would
    resolve either against the cwd, so a def exec'd under ``<string>`` and read
    from an exec-generated frame would collide with it and be waived (on 3.10 a
    fields-only dataclass too, whose ``__init__`` compiles in ``<string>``
    under a ``co_name`` that cannot tell it from a class statement's def). Only
    absolute filenames on both sides compare; anything else fails closed. What this
    cannot see is a same-name fork inside the reading file -- ``try: from x
    import impl as op`` / ``except ImportError: def op``, or a class statement
    under the same ``if`` -- which binds a different def per machine under one
    checksum; that is the conditional-bind KNOWN GAP recorded in
    ``_is_risky_drop``.
    """
    if not user_stack or not isinstance(value, (type, types.FunctionType)):
        return False
    if value.__qualname__ != global_name:
        return False

    # functools.wraps copies __qualname__ but not the code object's own name:
    # co_qualname from 3.11, before that co_name, its last component only.
    def compiled_as(fn: types.FunctionType, qualname: str) -> bool:
        if sys.version_info >= (3, 11):
            return fn.__code__.co_qualname == qualname
        return fn.__code__.co_name == qualname.rpartition(".")[2]

    if isinstance(value, type):
        # 3.14 stores the PEP 649 annotate function under __annotate_func__
        # with qualname Cls.__annotate__, which the key rule refuses; the skip
        # covers a version that stores it under its own name.
        skip = ("__annotate__", "__annotate_func__")
        files: list[str] = []
        for key, attr in vars(value).items():
            if isinstance(attr, (staticmethod, classmethod)):
                attr = attr.__func__
            elif isinstance(attr, property):
                attr = attr.fget
            if not isinstance(attr, types.FunctionType) or key in skip:
                continue
            qualname = f"{global_name}.{key}"
            if attr.__qualname__ == qualname and compiled_as(attr, qualname):
                files.append(attr.__code__.co_filename)
    else:
        files = [value.__code__.co_filename] if compiled_as(value, global_name) else []
    read = user_stack[0].filename
    if not os.path.isabs(read):
        return False
    try:
        here = _norm(read)
        return any(os.path.isabs(f) and _norm(f) == here for f in files)
    except ValueError:  # an embedded NUL, which posixpath.realpath lets through
        return False


def _dynamo_alias_module(global_name: str) -> types.ModuleType | None:
    """
    The module behind an ``__import_a_dot_b`` alias, mirroring the ordinary
    branch of import_source; a torch_package module is aliased without the
    prefix and comes back None here, which fails closed.

    The OutputGraph's import_sources table is authoritative, but a guard entry
    does not carry it; unmangling collides only for a module whose name
    contains ``_dot_`` (``pkg.sub_dot_mod`` is aliased as
    ``__import_pkg_dot_sub_dot_mod``, which comes back as ``pkg.sub.mod`` when
    that module is loaded too), and that collision fails open (the callers judge
    the module returned).
    """
    if not global_name.startswith(_IMPORT_ALIAS_PREFIX):
        return None
    tail = global_name[len(_IMPORT_ALIAS_PREFIX) :]
    return sys.modules.get(tail.replace("_dot_", "."))


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
    ``import mypkg``, and the ``__import_x`` alias Dynamo installs to reach an
    inlined function's own globals -- or if it is an attribute of a trusted
    module under a name that module already owns: its own ``__name__`` (a plain
    ``import own_sub`` inside the parent) or the parent's plus the attribute
    (``from . import sub``, and ``import mypkg.layers``, which Dynamo guards as
    ``G['mypkg'].layers``). An ALIASED user module is none of those:
    ``if flag: import impl_b as impl`` picks what ``impl.op`` resolves to per
    machine, and so does the same alias spelled ``from . import impl_b as
    impl`` in a package __init__. Inheriting the parent's trust without
    checking the name is what let that shape through before.

    Library ownership is trusted under ANY binding, ``import torch.nn.functional
    as F`` included; the price, taken deliberately because flagging ``F`` would
    flag every model, is that an alias config picks between two torch modules
    is waived too (see ``_is_risky_drop``'s KNOWN GAP). A config module is the
    one namespace whose bindings config chooses by definition, so it is never
    trusted, whoever owns it and however it is reached (a recovered
    ``__import_x`` alias included). Keys are source names, and the consumer
    looks a read's ``source.base.name`` up exactly, never by prefix.
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
        if name in trusted:
            return trusted[name]
        found = modules.get(name)
        ok = False
        if found is not None:
            source, module = found
            if isinstance(module, ConfigModule):
                ok = False
            elif _is_library_module(module.__name__):
                ok = True
            elif isinstance(source, GlobalSource):
                ok = (
                    source.global_name == module.__name__
                    or _dynamo_alias_module(source.global_name) is module
                )
            elif isinstance(source, AttrSource):
                outer = modules.get(source.base.name)
                ok = (
                    outer is not None
                    and is_trusted(source.base.name)
                    and module.__name__
                    in (source.member, f"{outer[1].__name__}.{source.member}")
                )
        trusted[name] = ok
        return ok

    return {name: module for name, (_, module) in modules.items() if is_trusted(name)}


def _reads_a_builtin(source: Source, value: object) -> bool:
    """
    ``len`` or ``sorted`` reached the ordinary way, through the builtins dict
    Dynamo installs to resolve them. That dict is the frame's live
    ``builtins.__dict__``, not a table of the real builtins, so a shim's
    ``builtins.py2_sum = sum`` is a binding under it that another machine's
    shim can point elsewhere; only a builtin ``builtins`` itself owns, read
    under its own name, is waived (``IOError`` and ``EnvironmentError``,
    CPython's aliases of ``OSError``, fail closed), and only one CPython built:
    functools.wraps copies ``__module__`` and ``__name__`` onto ``builtins.sum =
    wraps(sum)(logged_sum)``, a Python function whose CLOSURE_MATCH is dropped,
    and a class statement exec'd with the builtins namespace as its globals (or
    handed ``__module__ = "builtins"``) claims the module outright, so the value
    must be a builtin function or a static type as well, one carrying
    ``Py_TPFLAGS_IMMUTABLETYPE``, which a class statement or a ``type()`` call
    does not carry. The flag is read through ``type``'s own descriptor, so a
    metaclass can neither shadow it nor make the read raise, and a static
    type's ``__module__`` is derived from its C name; a builtin function's
    ``__module__`` is a writable member, though, so ``builtins.getcwd =
    os.getcwd`` plus ``os.getcwd.__module__ = "builtins"`` is waived -- evidence
    rather than proof on that branch, which an advisory lint over a
    dropped-guard set does not defend against. Every callable in
    ``builtins.__dict__`` is one of the two kinds apart from the
    ``_sitebuiltins`` objects, instances of neither, and beside the aliases
    (``WindowsError`` is a third on Windows) three of them fail closed:
    ``open``, the one builtin function ``builtins`` does not own (its
    ``__module__`` is ``_io``, ``io`` before 3.12, so a dropped guard on one of
    the most mainstream builtins here is reported), and the heap types
    ``ExceptionGroup`` (3.11+) and ``__loader__``, the latter refused under a
    name that is not its own as well. The exposure is
    narrow either way: a registered builtin is id-matched into a BUILTIN_MATCH
    the serializer keeps, so only a deregistered, polyfilled one (``sum``,
    ``enumerate``, ``all``, ``any``) or a shim reaches the dropped set this
    lint examines.

    A builtin parked in a slot -- ``self.act = abs``, straight out of an
    ACT2FN-style table -- is a slot like any other, so this deliberately keys
    on where the read comes FROM rather than on who owns the value.
    """
    return (
        isinstance(source, DictGetItemSource)
        and isinstance(source.base, GlobalSource)
        and source.base.global_name.startswith(_BUILTINS_DICT_PREFIX)
        and (
            isinstance(value, types.BuiltinFunctionType)
            # Py_TPFLAGS_IMMUTABLETYPE: a static type CPython built, not a
            # class statement or type() call, whose __module__ is writable. Read
            # through type's descriptor, past any metaclass shadowing __flags__.
            or (
                isinstance(value, type)
                and bool(vars(type)["__flags__"].__get__(value) & (1 << 8))
            )
        )
        and _owning_module(value) == "builtins"
        and getattr(value, "__name__", None) == source.index
    )


# Built once: config.patch() allocates a class and a ContextVar each time it is
# called, and this runs on every frame Dynamo compiles for a package.
_ALLOW_EMPTY_GRAPHS = torch._dynamo.config._make_closure_patcher(
    allow_empty_graphs=True
)


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


def _compose_with_default(
    user: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]] | None,
) -> Callable[[Sequence[GuardFilterEntry]], tuple[Sequence[bool], Sequence[bool]]]:
    """AND a caller's filter with the default rather than replacing it.

    ``default_guard_filter_fn`` is not a default in the "sensible starting point"
    sense -- it is what drops the identity guards that CANNOT be serialized at
    all. Replacing it means a caller who wanted to drop three of their own guards
    silently re-admits every unserializable one, and the failure surfaces as
    "ID_MATCH guard cannot be serialized" in frames that have nothing to do with
    their filter. A custom filter can only ever want to drop MORE, so composing
    is the only reading that makes sense. ``user=None`` is no custom filter at
    all, where the default's decisions ARE the composition.

    Both are returned: the recorder judges a drop against the default's own
    verdict, and returning it here is what keeps the default to one call per
    compile.
    """

    def composed(
        entries: Sequence[GuardFilterEntry],
    ) -> tuple[Sequence[bool], Sequence[bool]]:
        base = default_guard_filter_fn(entries)
        if user is None:
            return base, base
        chosen = user(entries)
        if len(chosen) != len(entries):
            raise ValueError(
                f"guard_filter_fn returned {len(chosen)} decisions for "
                f"{len(entries)} guards; it must return one per entry."
            )
        return [bool(a) and bool(b) for a, b in zip(base, chosen)], base

    return composed


_OBJ_ID = re.compile(r"(?<=, )\d+(?=\), type=)")


_SAVED_HOOK_IDS = re.compile(r"(?<=top_saved_tensors_hooks ids == )\(\d+(?:, \d+)*\)")


_DYNAMO_COUNTER = re.compile(
    r"(__builtins_dict__|__compiled_fn|__resume_at)_*\d+(_\d+)?"
)


# OutputGraph.install_global_by_id names a global "<prefix>_<id(value)>_c<n>",
# so a guard reading one carries BOTH an address and a compile counter inside
# an identifier, where neither pattern above can see it. Real models reach this
# -- transformers' Qwen2 installs three -- and the report then differs run to
# run, which is exactly what the "commit and diff" contract rules out.
_DYNAMO_GLOBAL_BY_ID = re.compile(r"_\d{9,}_c\d+\b")


# A default __repr__ renders the object's address, which no pattern above sees.
_HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def _normalize(text: str) -> str:
    text = _SAVED_HOOK_IDS.sub("(<ids>)", text)
    text = _HEX_ADDRESS.sub("<addr>", text)
    text = _DYNAMO_GLOBAL_BY_ID.sub("_<id>_c<n>", _OBJ_ID.sub("<id>", text))
    return _DYNAMO_COUNTER.sub(r"\1_<n>", text)


def _render_code(code_list: Sequence[str] | None) -> tuple[str, ...]:
    # Keep the _dynamo_*_indices parts: they carry TENSOR_MATCH's dimension
    # marking, so mark_static on one variant and not the next shows up only here.
    return tuple(_normalize(part) for part in (code_list or ()))


# Guards whose drop changes what a graph is reused FOR instead of crashing in a
# kernel: a shape, a pinned Python value or type, a container fact, the default
# device. _warn_risky_drops surfaces these first for that reason.
_SHAPE_BEARING_GUARD_TYPES = frozenset(
    {
        "TENSOR_MATCH",
        "SEQUENCE_LENGTH",
        "CONSTANT_MATCH",
        "EQUALS_MATCH",
        "DUPLICATE_INPUT",
        "HASATTR",
        "TYPE_MATCH",
        "FAKE_SCRIPT_TYPE_MATCH",
        "DEFAULT_DEVICE",
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


# Guards this report cannot model a comparable value for (FSDP_TRAINING_STATE is
# per param group and GlobalStateGuard does not snapshot it), so their facts are
# reported as undetermined rather than compared.
_UNMODELLED_GUARD_TYPES = frozenset(
    {
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


# Guard types whose GuardBuilder method is `pass`: the guard is a marker, and
# the check it names is made by GLOBAL_STATE's leaf. Nothing about them is
# serialized or dropped, so they never appear in a dropped-guard report --
# listing GRAD_MODE as "a precondition nothing checks" would be false, since
# GlobalStateGuard checks it on every call. Their facts ARE compared, on what a
# marker renders, which is its type and source alone.
_NOOP_GUARD_TYPES = frozenset({"DETERMINISTIC_ALGORITHMS", "GRAD_MODE"})


def _is_noop_guard_type(guard_type: str) -> bool:
    # EMPTY_NN_MODULE_HOOKS_DICT is a no-op by config: under
    # skip_nnmodule_hook_guards, the default, GuardBuilder emits nothing for it.
    return guard_type in _NOOP_GUARD_TYPES or (
        guard_type == "EMPTY_NN_MODULE_HOOKS_DICT"
        and torch._dynamo.config.skip_nnmodule_hook_guards
    )


def _render_fact(fact: _GuardFact) -> str:
    """Render one guard as a stable, human-readable line for the report."""
    body = " ; ".join(fact.code) if fact.code else f"<{fact.guard_type}>"
    if fact.value:
        body = f"{body} {fact.value}"
    where = f" on {fact.source}" if fact.source else ""
    label = "enforced" if fact.enforced else "dropped"
    return f"[{label:<8}] {body}{where}"


def _fact_order(fact: _GuardFact) -> tuple[str, str, str, str]:
    # value is part of the key: once the boilerplate code parts are filtered a
    # TENSOR_MATCH renders no code, so two shape specializations would otherwise
    # tie and sort unstably, making the file differ run to run.
    return (fact.source, fact.guard_type, " ".join(fact.code), fact.value)


# A compiled frame: what the report prints for it, then the identity of the
# package entry that owns its code object (one entry per code object). The
# printed triple alone collides -- two lambdas on one line, or two frames exec'd
# under "<string>" -- and colliding frames would have their guard sets
# intersected as if they were variants of one frame.
_FrameKey = tuple[str, str, int, int]


def _summarize(
    entry: _DynamoCacheEntry,
    dropped: set[tuple[str, str]],
    kept: set[tuple[str, str]],
    policy_dropped: set[tuple[str, str]],
    risky: set[tuple[str, str]],
    uncovered: frozenset[str],
    capture_errors: Sequence[str],
    guard_sets: Mapping[_FrameKey, Sequence[frozenset[_GuardFact]]],
    dropped_code: Mapping[tuple[str, str], str],
) -> PrecompileSummary:
    # truncated and wont_generalize keep their empty defaults: the recompile-limit
    # bookkeeping and the value-pinning analysis behind them are not part of this
    # build.
    return PrecompileSummary(
        frames=len(entry.codes),
        resume_functions=sum(1 for c in entry.codes if c.install_to_global),
        guarded_codes=sum(len(c.guarded_codes) for c in entry.codes),
        backend_graphs=len(entry.backend_ids),
        bypassed=tuple(c.python_code.co_name for c in entry.codes if c.bypassed),
        uncovered_frames=tuple(sorted(uncovered)),
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


def _entry_fn_of(fn: object) -> Callable[..., object]:
    if isinstance(fn, torch.nn.Module):
        forward = fn.forward
        if not hasattr(forward, "__code__"):
            raise TypeError(
                f"{type(fn).__name__}.forward is a {type(forward).__name__}, which "
                f"has no __code__ for Dynamo to capture or to load an artifact onto. "
                f"Binding it in __init__ -- self.forward = functools.partial(...) -- "
                f"shadows the class method and lands here; keep forward a method."
            )
        return forward
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


def _identify_graph(gm: torch.fx.GraphModule) -> str:
    """Name a graph well enough to find it, from inside a backend.

    The module class alone does not: every graph a model produces reports the
    same one, so a capture that recompiled nine graphs at serve time said
    "GraphModule" nine times. The compile id keys tlparse, the backend id keys
    the artifact, and the first node carrying a stack trace names the user line
    -- including, for a continuation, the resume frame Dynamo minted for it,
    which is the only thing that tells one break in a chain from another.
    """
    parts: list[str] = []
    compile_id = torch._guards.CompileContext.current_compile_id()
    if compile_id is not None:
        parts.append(f"compile id {compile_id}")
    backend_id = gm.meta.get("backend_id") or getattr(gm, "_backend_id", None)
    if backend_id is not None:
        parts.append(f"backend id {backend_id}")
    for node in gm.graph.nodes:
        if node.op not in ("placeholder", "output") and node.stack_trace:
            first = node.stack_trace.strip().splitlines()[0].strip()
            parts.append(f"first traced at {first}")
            break
    return f" Graph: {'; '.join(parts)}." if parts else ""


class _PrecompileBackend:
    """Give one explicit session its own Dynamo cache identity."""

    def __init__(
        self, backend: str, keep_graphs: bool = False, serving: bool = False
    ) -> None:
        inner = torch._dynamo.lookup_backend(backend)
        self._torchdynamo_orig_backend = inner
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

    def __call__(self, gm: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Any:
        if self.serving:
            self.serve_time_compiles += 1
            log.warning(
                "precompile: serving compiled a NEW graph -- no captured variant "
                "matched this call, so the artifact is serving less than it was "
                "measured to. Recapture with an example that covers it.%s",
                _identify_graph(gm),
            )
        if self._keep_graphs:
            backend_id = gm.meta.get("backend_id")
            if backend_id is not None and str(backend_id) not in self.graphs:
                # Deep-copy before the inner backend runs: inductor lowering
                # mutates the graph it is handed, and a rendered copy has to be
                # the graph Dynamo produced, not the leftovers.
                placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
                # Render against the placeholders' FAKES, never the real inputs
                # below: compile_fx re-fakifies real tensors into a fresh symbol
                # set and dedups by value, which silently unifies a batch dim
                # with any other dim of the same size.
                fakes = [
                    n.meta.get("example_value", n.meta.get("val")) for n in placeholders
                ]
                if all(f is not None for f in fakes):
                    self.graphs[str(backend_id)] = (copy.deepcopy(gm), fakes)
        return self._torchdynamo_orig_backend(gm, inputs)

    def get_compiler_config(self) -> Any:
        getter = getattr(self._torchdynamo_orig_backend, "get_compiler_config", None)
        return None if getter is None else getter()


def _optimize_isolated(
    backend: _PrecompileBackend,
    package: CompilePackage,
    *,
    recompile_limit: int,
    dynamic: bool | None,
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]] | None,
) -> OptimizeContext:
    from .eval_frame import OptimizeContext

    optimize_ctx = torch._dynamo.optimize(
        backend,
        package=package,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        guard_filter_fn=guard_filter_fn,
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


def _warn_risky_drops(risky: Sequence[tuple[str, str]]) -> None:
    """Report accepted risky drops, shape-bearing ones first.

    Ordering by type only puts the candidates where they can be seen; whether a
    guarded VALUE can differ at serve time is not something the type answers.
    """
    by_type: dict[str, list[str]] = {}
    for guard_type, name in sorted(risky):
        by_type.setdefault(guard_type, []).append(name)

    # Grouped rather than a flat cut, and capped PER TYPE: a flat list is
    # dominated by whichever type happens to be most numerous, which can bury a
    # lone SEQUENCE_LENGTH behind a crowd of CONSTANT_MATCH and CLOSURE_MATCH.
    def render(types: list[str], per_type: int) -> str:
        parts = []
        for t in types:
            names = by_type[t]
            shown = ", ".join(names[:per_type])
            more = f", +{len(names) - per_type} more" if len(names) > per_type else ""
            parts.append(f"{t} x{len(names)}: {shown}{more}")
        return "; ".join(parts)

    shape_types = [t for t in by_type if t in _SHAPE_BEARING_GUARD_TYPES]
    other_types = [t for t in by_type if t not in _SHAPE_BEARING_GUARD_TYPES]
    # Says "could" rather than "can", and points at the distinction that
    # actually decides it. This is a classification by guard TYPE, and the
    # question a reader has is whether the guarded VALUE can differ at serve
    # time -- which the type does not answer. The first four this ordering
    # surfaced on a real model were all reached through a class or function
    # definition (__mro__ walks to __defaults__, __code__) and were therefore
    # compile-time constants that no batch could change. Distinguishing those
    # properly needs the structured source, not the name.
    shape_report = (
        f" COULD BEAR ON SHAPE ({sum(len(by_type[t]) for t in shape_types)}), "
        f"unlike the rest, so check these first -- but check whether each one "
        f"can actually differ at serve time: a guard reached through a class or "
        f"function definition (an __mro__ walk, __defaults__, __code__) is a "
        f"compile-time constant and cannot: {render(shape_types, 3)}."
        if shape_types
        else ""
    )
    log.warning(
        "precompile: %d dropped guard(s) can affect dispatch, so nothing checks "
        "them at load.%s The rest are identity slots to audit: %s. "
        "summary().risky_dropped_guards has all of them; this warning appears "
        "only because require_no_risky_drops=False explicitly accepted them.",
        len(risky),
        shape_report,
        render(other_types, 2) or "none",
    )


def _missing_backends_message(
    total: int, missing: Sequence[object], backend: str = "inductor"
) -> str:
    """Why some compiled subgraphs never reached the artifact.

    Reports the recorded/total split rather than asserting nothing was
    recorded: a single missing id is fatal here, and saying so as "never
    recorded" reads as total failure when most of the capture succeeded.
    """
    shown = ", ".join(str(b) for b in missing[:8])
    if len(missing) > 8:
        shown += f", ... ({len(missing) - 8} more)"
    if backend not in ("inductor", "eager"):
        # A session takes any backend Dynamo can resolve, but only these two
        # leave something a served artifact can run: "eager" keeps the fx
        # graphs and "inductor" bundles compiled code. Anything else captures
        # cleanly and records nothing, so say so here rather than let it read
        # as a defect in the model. aot_eager is the one people reach for,
        # since it is how you isolate AOTAutograd.
        return (
            f"Precompilation recorded {total - len(missing)} of {total} "
            f"compiled backend(s) because backend={backend!r} does not produce "
            f"anything serializable; precompile can record only 'inductor' or "
            f"'eager'. To isolate AOTAutograd without inductor, use plain "
            f"torch.compile(backend='aot_eager') -- that needs no precompile."
        )
    from torch._dynamo.utils import counters

    # Rendering re-enters AOTAutograd outside the pinned bypass_autograd_cache_key
    # config and bypasses there routinely, so this is a diagnostic, not a diagnosis.
    bypasses = counters["aot_autograd"].get("autograd_cache_bypass", 0)
    bypass_note = (
        f" It bypassed {bypasses} time(s) here, which rendering does for any "
        "graph the cache cannot key, and which does not by itself explain a gap."
        if bypasses
        else ""
    )
    return (
        f"Precompilation recorded {total - len(missing)} of {total} compiled "
        f"backend(s), so {len(missing)} graph(s) would reach the artifact with "
        f"no code behind them: {shown}. Capture pins functorch's "
        "bypass_autograd_cache_key, so AOTAutograd keys every graph it lowers "
        "and no longer declines to record one it cannot address."
        + bypass_note
        + " A gap therefore means a graph whose backward never compiled, which "
        "is a forward-only capture with grad enabled. Pass training=True to "
        "lower the backward eagerly (the joint trace synthesizes tangents, so "
        "no loss is needed), capture under torch.no_grad() / "
        "torch.inference_mode() for an inference artifact, or run .backward() "
        "inside the capture block. Re-run with "
        "TORCH_LOGS=+torch._functorch._aot_autograd to see each graph as it "
        "lowers."
    )


class PrecompileSession:
    """
    A caller-driven capture in progress. Enter as a context manager to get the
    callable to exercise and invoke it with real inputs inside the block. The
    compiled region stays alive for the whole block, so every call reuses the
    variants the earlier ones produced.
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
        training: bool = False,
        keep_graphs: bool = False,
        invariants: str | None = None,
    ) -> None:
        self._fn = fn
        self._backend = backend
        # A training capture traces with grad on and lowers the backward
        # eagerly, so the artifact carries AOTAutograd's CompiledFunction and
        # calling .backward() on a served output runs precompiled code.
        self._training = training
        # Retaining a graph deepcopies it for the session; the guard probe and
        # captures whose graphs are never rendered leave it off.
        self._keep_graphs = keep_graphs
        self._invariants_path = invariants
        self._backend_obj: _PrecompileBackend | None = None
        # Slots dropped by the invariance policy. The policy itself is not part
        # of this build, so the set stays empty; summary() subtracts it anyway.
        self._policy_dropped_guards: set[tuple[str, str]] = set()
        # slot -> the check it rendered as, for every slot dropped by any
        # route. See PrecompileSummary.dropped_guard_code for why the slot
        # tuple alone cannot be audited.
        self._dropped_guard_code: dict[tuple[str, str], str] = {}
        self._dropped_guards: set[tuple[str, str]] = set()
        self._kept_guards: set[tuple[str, str]] = set()
        self._risky_dropped_guards: set[tuple[str, str]] = set()
        self._capture_errors: list[str] = []
        # How many capture errors predate the capture block. Set at __enter__;
        # a render counts only the errors raised since.
        self._gate_error_mark = 0
        self._recorded_exception_keys: set[tuple[type[BaseException], str]] = set()
        # frame -> one fact set per compilation of it
        self._guard_sets: dict[_FrameKey, list[frozenset[_GuardFact]]] = {}
        self._undetermined: dict[_FrameKey, set[_GuardFact]] = {}
        self._guard_filter_fn = self._recording_filter(
            _compose_with_default(guard_filter_fn)
        )
        self._recompile_limit = recompile_limit
        self._dynamic = dynamic
        self._entry_fn = _entry_fn_of(fn)
        # The guard filter rides on the optimize context rather than the
        # package, so it applies to the live guards as well as the serialized
        # ones, exactly as caching_precompile does today.
        self._package = CompilePackage(self._entry_fn)
        self._backend_artifacts: dict[_BackendId, Any] = {}
        self._entered = False
        self._optimized: Callable[..., object] | None = None
        self._compiled: Callable[..., object] | None = None
        self._state = threading.Condition()
        self._active_calls = 0
        self._closing = False
        self._finished = False

    def _take_backend_artifacts(self) -> None:
        from torch._dynamo.output_graph import noop_graph_call
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        for backend_id in self._package.cache_entry().backend_ids:
            artifact = PrecompileContext.take_artifact(backend_id)
            if artifact is not None:
                self._backend_artifacts[backend_id] = artifact
            elif self._package.cached_backends.get(backend_id) is noop_graph_call:
                # output_graph short-circuits an empty graph to noop_graph_call
                # without filing anything under its id, which the bytecode still
                # names. Record the no-op so the served frame dispatches to it
                # rather than running eager. Here rather than in
                # _collect_backends: teardown clears cached_backends first.
                self._backend_artifacts[backend_id] = EagerCacheArtifact(
                    key=backend_id, content=noop_graph_call
                )

    def _record_capture_error(self, error: BaseException) -> None:
        message = str(error)
        key = (type(error), message)
        if key in self._recorded_exception_keys:
            return
        self._recorded_exception_keys.add(key)
        self._capture_errors.append(f"{type(error).__name__}: {message}")

    def _release(self) -> None:
        # The compiled variants stay in the entry's ordinary Dynamo cache, as
        # they would after torch.compile; clearing them per capture needs the
        # region-scoped cache entries that are not part of this build. The
        # eager backends stay until the render collects them.
        self._optimized = None
        if self._backend != "eager":
            self._package.cached_backends.clear()

    def _call(self, *args: object, **kwargs: object) -> object:
        with self._state:
            if self._compiled is None or self._closing:
                raise RuntimeError("PrecompileSession is not active")
            compiled = self._compiled
            self._active_calls += 1
        try:
            with _capture_config(self._training):
                result = compiled(*args, **kwargs)
        except BaseException as e:
            self._record_capture_error(e)
            raise
        finally:
            with self._state:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._state.notify_all()
        return result

    def __enter__(self) -> Callable[..., object]:
        if self._finished:
            raise RuntimeError("PrecompileSession cannot be re-entered")
        if self._entered:
            raise PackageError(
                "PrecompileSession is already active: a session runs one capture "
                "block at a time, so serialize concurrent entries."
            )
        self._gate_error_mark = len(self._capture_errors)
        # The grad-mode/config patch is per call, in _call, not block-level:
        # user code between calls (optimizer.step, data loading) must run in
        # the ambient mode, not the capture's.
        self._entered = True
        try:
            if self._optimized is None:
                self._backend_obj = _PrecompileBackend(self._backend, self._keep_graphs)
                optimize_ctx = _optimize_isolated(
                    self._backend_obj,
                    self._package,
                    recompile_limit=self._recompile_limit,
                    dynamic=self._dynamic,
                    guard_filter_fn=self._guard_filter_fn,
                )
                self._optimized = optimize_ctx(self._fn)
            self._compiled = self._optimized
        except BaseException as e:
            self._record_capture_error(e)
            # A __enter__ that raises never gets its __exit__, so without this
            # the session is wedged: the block reads as still open.
            self._entered = False
            self._compiled = None
            # Drain in-flight calls FIRST, before any teardown, exactly as
            # __exit__ does: a concurrent call can still be compiling against a
            # borrowed cache entry, and teardown mutates state it reads. The
            # cleanup chain sits in the drain's finally so an interrupt raised
            # out of wait() (e.g. KeyboardInterrupt) still releases the session
            # rather than leaking it until process exit.
            try:
                with self._state:
                    self._closing = True
                    while self._active_calls:
                        self._state.wait()
            finally:
                # _closing marks a drain in progress, so both teardown paths
                # clear it once their own drain is done.
                self._closing = False
                try:
                    self._take_backend_artifacts()
                except BaseException as teardown:
                    self._record_capture_error(teardown)
                finally:
                    self._release()
                    self._finished = True
                    with self._state:
                        self._state.notify_all()
            raise
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
                # An interrupted drain leaves the session open, so _closing goes
                # back to False here and in __enter__'s error path alike.
                self._closing = False
                self._state.notify_all()
                raise
            self._closing = False
            entered = self._entered
            self._entered = False
            self._compiled = None
        if entered:
            try:
                self._take_backend_artifacts()
            except BaseException as teardown:
                # Recorded, never raised: a teardown failure must not replace
                # the exception the caller's block is already propagating, and
                # the capture errors are what a render gates on.
                self._record_capture_error(teardown)
            finally:
                self._release()
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

    def _record_dropped_code(
        self, slot: tuple[str, str], code: Sequence[str] | None
    ) -> None:
        """Remember what a dropped slot actually checked.

        The FIRST rendering, as PrecompileSummary.dropped_guard_code documents:
        one rendering however many variants dropped the slot. A check that embeds
        its value (EQUALS_MATCH renders L['n'] == 3) therefore tells the form of
        the check rather than every value the slot took, and the field cannot
        grow with the variant count.
        """
        rendered = " ; ".join(_render_code(code))
        if rendered:
            with self._state:
                self._dropped_guard_code.setdefault(slot, rendered)

    def _recording_filter(
        self,
        inner: Callable[
            [Sequence[GuardFilterEntry]], tuple[Sequence[bool], Sequence[bool]]
        ],
    ) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
        """
        Remember which guard types were discarded. A dropped guard does not
        fail at serving time, it silently widens what a graph is reused for, so
        the set has to be inspectable rather than invisible.
        """

        # One object per distinct fact, shared by every compilation that
        # produced it. A recompiled frame repeats nearly all of its guards, so
        # storing a copy per compilation would make the session grow with
        # variants rather than with facts.
        pool: dict[_GuardFact, _GuardFact] = {}

        def filter_fn(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
            # A custom filter composes with the default, so the identity drops
            # the default makes anyway are judged as they always are; only a
            # drop the custom filter ADDED is risky by construction, because
            # nothing here can say what the caller gave up. The composition
            # hands back the default's own verdicts, so it runs once per compile.
            decisions, default_kept = inner(entries)
            # A no-op type's check, where it has one, is GLOBAL_STATE's leaf,
            # made whatever the filter said about the marker: it is dropped
            # only with GLOBAL_STATE.
            global_state_kept = any(
                keep and e.guard_type == "GLOBAL_STATE"
                for keep, e in zip(decisions, entries)
            )
            facts: set[_GuardFact] = set()
            undetermined: set[_GuardFact] = set()
            kept_slots: set[tuple[str, str]] = set()
            dropped_slots: set[tuple[str, str]] = set()
            risky_slots: set[tuple[str, str]] = set()
            for keep, by_default, entry in zip(decisions, default_kept, entries):
                # Normalized where the slot is RECORDED, not where it is read: a
                # Dynamo per-process counter (__builtins_dict___14) otherwise
                # makes one logical slot appear once per compilation, under names
                # that change every run, and the slot lists, the risky subset and
                # the rendered code end up spelling it three different ways.
                slot = (entry.guard_type, _normalize(entry.name))
                enforced = keep or (
                    _is_noop_guard_type(entry.guard_type) and global_state_kept
                )
                if not enforced:
                    self._record_dropped_code(slot, entry.orig_guard.code_list)
                (kept_slots if enforced else dropped_slots).add(slot)
                # Risky here means a drop the default filter would not have made;
                # the finer lint over identity guards is not part of this build.
                if not enforced and by_default:
                    risky_slots.add(slot)
                unmodelled = entry.guard_type in _UNMODELLED_GUARD_TYPES
                fact = _GuardFact(
                    guard_type=entry.guard_type,
                    source=slot[1],
                    code=_render_code(entry.orig_guard.code_list),
                    value="" if unmodelled else _plain_value(entry),
                    enforced=enforced,
                )
                fact = pool.setdefault(fact, fact)
                # Never compared, so never claimed to hold: see
                # _UNMODELLED_GUARD_TYPES.
                (undetermined if unmodelled else facts).add(fact)
            # One filter call is one compilation, and only the package knows
            # which frame is being compiled. Without it there is no frame to
            # attribute the facts to, so they go unrecorded rather than into a
            # made-up one.
            compiling = self._package._current_entry
            # Published under the lock a reader takes, in one step, because this
            # runs on whatever thread is compiling.
            with self._state:
                self._kept_guards |= kept_slots
                self._dropped_guards |= dropped_slots
                self._risky_dropped_guards |= risky_slots
                if compiling is not None:
                    code = compiling.python_code
                    key = (
                        code.co_name,
                        code.co_filename,
                        code.co_firstlineno,
                        id(compiling),
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

        GLOBAL_STATE, TORCH_FUNCTION_STATE and FSDP_TRAINING_STATE carry no
        value of their own, so nothing here can say whether two variants agreed
        on, say, autocast. They are listed as UNDETERMINED rather than compared
        -- see _UNMODELLED_GUARD_TYPES, and note that calling them equal is
        precisely how the report would assert a precondition that does not
        hold. GRAD_MODE and DETERMINISTIC_ALGORITHMS are compared like any other
        guard, on what their marker renders, which is the type and source alone.
        """
        # Snapshotted under the lock because a compile on another thread records
        # into these dicts; the facts themselves are immutable, so the rendering
        # below needs no lock.
        with self._state:
            recorded = [
                (key, list(sets), set(self._undetermined.get(key, ())))
                for key, sets in self._guard_sets.items()
            ]
        out = []
        # Ordered by the printed triple alone: the identity tail is an address, so
        # ordering on it would reshuffle the report run to run. Ties (frames that
        # share all three) keep the order they were first compiled in.
        for key, sets, undetermined in sorted(recorded, key=lambda item: item[0][:3]):
            name, filename, lineno, _ = key
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
                    undetermined=tuple(sorted(undetermined, key=_fact_order)),
                )
            )
        return tuple(out)

    def write_invariants(self, path: str, /) -> None:
        """
        Write :meth:`invariants` to ``path`` in human-readable form.

        ``path`` is a FILE, written exactly as given, with parent directories
        created -- ``snapshots/invariants.txt`` is a text file.

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
                lines.append(f"  invariant {_render_fact(fact)}")
            for fact in f.varying:
                lines.append(f"  varies    {_render_fact(fact)}")
            for fact in f.undetermined:
                lines.append(f"  unknown   {_render_fact(fact)}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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
        frames = self.invariants()
        # Risky only when the SAME source held DIFFERENT values across variants;
        # merely being absent from one variant (a MODULE_MATCH a branch does not
        # touch) flags every ordinary multi-branch capture. Grouped by source,
        # not (guard_type, source): a rebind can change the guard type too.
        values_by_source: dict[str, set[str]] = {}
        for frame in frames:
            for fact in frame.varying:
                if not fact.enforced:
                    values_by_source.setdefault(fact.source, set()).add(fact.value)
        varying_dropped = {
            (fact.guard_type, fact.source)
            for frame in frames
            for fact in frame.varying
            if not fact.enforced and len(values_by_source.get(fact.source, ())) > 1
        }
        # Snapshotted under the lock, for the reason invariants() takes it. The
        # slots are already normalized, so every list here spells one slot the
        # same way and risky_dropped_guards really is a subset of dropped_guards.
        with self._state:
            dropped = set(self._dropped_guards)
            kept = self._kept_guards - self._policy_dropped_guards
            policy_dropped = set(self._policy_dropped_guards)
            risky = self._risky_dropped_guards | (dropped & varying_dropped)
            dropped_code = dict(self._dropped_guard_code)
            guard_sets = dict(self._guard_sets)
            capture_errors = list(self._capture_errors)
        from .package import SerializedCode

        entry = self._package.cache_entry()
        return _summarize(
            entry,
            dropped,
            kept,
            policy_dropped,
            risky,
            # An uncovered frame is one the package recorded without a single
            # guarded variant.
            frozenset(
                SerializedCode.to_code_object(code.python_code).co_name
                for code in entry.codes
                if not code.bypassed and not code.guarded_codes
            ),
            capture_errors,
            guard_sets,
            dropped_code,
        )

    def _gated_summary(
        self,
        *,
        require_complete: bool,
        require_no_risky_drops: bool,
        require_no_dropped_guards: bool,
    ) -> PrecompileSummary:
        """Run the coverage and guard gates, or raise saying which one failed.

        Callable mid-block, while the compiled region is still live.
        """
        summary = self.summary()
        # Only the errors raised since the block was entered. A call that failed
        # already raised to the caller, who saw it and carried on; counting a
        # pre-block error here would refuse a render over something the caller
        # already handled.
        fresh_errors = list(summary.capture_errors)[self._gate_error_mark :]
        if require_complete and fresh_errors:
            raise PackageError(
                "Precompilation is incomplete because capture raised: "
                f"{fresh_errors}. Re-run every example successfully, "
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
            # The caller explicitly accepted the risk.
            _warn_risky_drops(summary.risky_dropped_guards)
        if require_complete:
            if summary.guarded_codes == 0:
                raise PackageError(
                    "Precompilation captured no compiled code. Capture happens by "
                    "execution, so the callable must actually be run inside the "
                    "capture block. A call Dynamo could not turn into guarded code "
                    "is reported separately as an uncovered frame."
                )
            if summary.backend_graphs == 0:
                raise PackageError(
                    "Precompilation compiled no graph: every captured frame was "
                    "empty, so the artifact carries no compiled compute. This is "
                    "what a callable whose whole body sits behind "
                    "torch._dynamo.disable looks like. Pass require_complete=False "
                    "to write the guards-only artifact anyway."
                )
            if summary.uncovered_frames:
                raise PackageError(
                    f"Precompilation exercised frame(s) that produced NO guarded code "
                    f"at all: {list(summary.uncovered_frames)}. Those paths are absent "
                    f"from the artifact, and such a frame is skipped at install and runs "
                    f"eager, so serving() cannot report that gap. This is expected for a "
                    f"frame that only dispatches to covered submodules; it also looks "
                    f"exactly like a frame Dynamo gave up on (check "
                    f"TORCH_LOGS=graph_breaks for gb0124). Pass require_complete=False "
                    f"once you have confirmed which."
                )
            if summary.bypassed:
                raise PackageError(
                    f"Precompilation is incomplete: {len(summary.bypassed)} frame(s) "
                    f"were bypassed and will serve nothing: {list(summary.bypassed)}. "
                    f"This usually means their guards could not be serialized. Pass "
                    f"require_complete=False to accept a partial artifact."
                )
        return summary

    def rendered_backends(
        self, backend_ids: Sequence[str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Compiled subgraphs as READABLE source, and the reason each of the rest
        stayed pickled, both keyed by backend id.

        The pickled bundle is the fallback, not the goal: a subgraph is Inductor
        output, which has a source form (the make_fx tracer emits exactly this),
        unlike the guard trees and transformed bytecode beside it. Anything that
        fails to render -- an effectful op, a graph with no compute, a training
        shape the composer refuses -- stays pickled, and its reason is warned
        here and written into the artifact header so the fallback is visible.

        Rendering re-runs AOTAutograd + Inductor on the retained graph, so it is
        a second lowering, paid once per subgraph that reaches the artifact.
        """
        from torch._functorch import aot_autograd

        if self._backend_obj is None or self._backend == "eager":
            return {}, {}
        rendered: dict[str, str] = {}
        refused: dict[str, str] = {}
        for backend_id in backend_ids:
            held = self._backend_obj.graphs.get(str(backend_id))
            if held is None:
                continue
            gm, fakes = held
            try:
                # grad_enabled is what makes AOTAutograd emit the joint
                # forward+backward for a training capture; without it the
                # backward is silently absent and the served output loses its
                # grad_fn.
                source, _ = aot_autograd.compile_to_python(gm, fakes)
            except Exception as e:
                reason = " ".join(f"{type(e).__name__}: {e}".split())
                log.warning(
                    "precompile: subgraph %s stays pickled, not rendered as source: %s",
                    backend_id,
                    reason,
                )
                refused[str(backend_id)] = reason
                continue
            rendered[str(backend_id)] = source
        from torch._functorch._aot_autograd.to_standalone_python import (
            namespace_module_names,
        )

        keys = list(rendered)
        namespaced = namespace_module_names([rendered[k] for k in keys])
        return dict(zip(keys, namespaced)), refused

    def snapshot_artifact(
        self,
        *,
        require_complete: bool = True,
        require_no_risky_drops: bool = True,
        require_no_dropped_guards: bool = False,
    ) -> tuple[str, bytes]:
        """Render everything captured SO FAR, leaving this session able to capture more.

        The render reads the package's records without consuming them, which is
        what lets ``save()`` be called repeatedly within one capture block.
        """
        summary = self._gated_summary(
            require_complete=require_complete,
            require_no_risky_drops=require_no_risky_drops,
            require_no_dropped_guards=require_no_dropped_guards,
        )
        from torch._precompile import _build_multigraph_artifact

        backends = self._collect_backends()
        entry = self._package.cache_entry()
        rendered, refused = self.rendered_backends(list(backends))
        return _build_multigraph_artifact(
            entry,
            backends,
            summary,
            self._backend,
            _entry_fn_of(self._fn),
            rendered,
            refused,
        )

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
                _missing_backends_message(
                    len(entry.backend_ids), missing, self._backend
                )
            )
        return {str(b): collected[b] for b in entry.backend_ids}


def precompile_capture(
    fn: Callable[..., object],
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
    training: bool = False,
    keep_graphs: bool = False,
    invariants: str | None = None,
) -> PrecompileSession:
    r"""Begin capturing ``fn`` into a multi-graph artifact.

    ``recompile_limit`` defaults well above Dynamo's usual 8 because a
    precompile deliberately wants one compiled variant per condition, whereas
    the normal limit exists to catch runaway recompilation. Nothing raises the
    ambient ``accumulated_recompile_limit`` (256), which Dynamo checks first and
    counts across every isolated region on the code object, so that ceiling --
    not this argument -- is the effective cap above 256 variants.

    The capture is caller-driven: enter the session to get a callable, invoke it
    exactly as you would ``fn`` inside the ``with`` body, and the calls fold into
    the artifact in the ambient grad mode. The compiled region stays alive for
    the whole block, so every call reuses the variants the earlier ones
    produced. ``invariants`` names a file written when the block exits without
    an exception.

    Runtime guards remain intact during capture. ``guard_filter_fn`` applies
    only to the serialized guard state, so every call observes the same
    recompilation behavior as ordinary ``torch.compile``.
    """
    return PrecompileSession(
        fn,
        backend=backend,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        training=training,
        keep_graphs=keep_graphs,
        invariants=invariants,
    )


def _plain_value(entry: GuardFilterEntry) -> str:
    """What a guard's check compares, rendered without the guarded value itself.

    GuardFact.value's contract, and the report is a file write_invariants tells
    the user to commit: a guarded value is user data (a tensor's elements, a
    prompt, a path), so a tensor renders as dtype, shape and device, a small
    builtin literal as itself, and anything else as its type. That also keeps
    the value a GUARD comparison -- two calls whose tensors differ only in their
    elements produce one fact, not a spurious varying pair -- and keeps a CUDA
    tensor off the host, where repr() would sync the device on every compile.

    Nothing here may raise: guards.py calls the filter unguarded, so a property
    or a __repr__ that fails would turn a capture into a compile failure.
    """
    try:
        if not entry.has_value:
            return ""
        value = entry.value
        if isinstance(value, torch.Tensor):
            return f"{value.dtype} {tuple(value.shape)} {value.device}"
        if value is None or isinstance(value, (bool, int, float, complex, str)):
            text = _normalize(repr(value))
            # A long literal is the data itself rather than a fingerprint of it,
            # so it degrades to its type instead of being cut into the report.
            return text if len(text) <= 64 else f"<{type(value).__name__}>"
        return f"<{type(value).__name__}>"
    except Exception:
        return "<unavailable>"
