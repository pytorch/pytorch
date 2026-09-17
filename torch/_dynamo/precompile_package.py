"""
Ahead-of-time precompilation of a callable into MANY graphs: the multi-graph
counterpart of ``torch.compile(fn, fullgraph=True).aot_compile(...)``. Every
frame Dynamo produces while the caller's calls run and the package can record
-- the entry frame, each ``torch_dynamo_resume_in_*`` continuation created by a
graph break, and every recompiled variant of each -- is captured into one
serializable artifact whose frames are stored through CompilePackage
(``torch/_dynamo/package.py``), a low-level component that is not meant to be
used directly; a frame it could not record is listed in
``PrecompileSummary.bypassed``.

Everything here is internal, and the module fills in over several commits: the
guard filter the serialized copy is written under (``default_guard_filter_fn``)
comes first, then the lint over the guards it drops (``_is_risky_drop``), the
guard-type classification and fingerprints behind the ``PrecompileSummary``
report, the per-frame comparison of captured variants and the summary builder
(``_varying_guard_slots``, ``_summarize``), and the compiler configuration a
capture runs under (``_capture_config``). The capture session that calls them
-- the ``torch.compiler.precompile.capture`` / ``accumulate`` / ``load`` entry
points, its ``guard_filter_fn`` and ``require_no_dropped_guards`` options and
the runnable ``load`` returns -- is a follow-up, so until it lands nothing
under ``torch/`` calls into this module; the module docstring of
``torch/_precompile.py``, rewritten near the top of this stack, carries the
caller-facing usage. Until that rewrite, the file documents the single-graph
callable API those entry points replace, ``torch.compiler.precompile(fn,
*example_inputs)`` and its ``load(python_code, cache)``. This is distinct from
``torch._dynamo.config.caching_precompile``, which caches ``torch.compile``
artifacts transparently without an explicit capture.

Capture is by execution, and the caller drives it: the session hands back a
callable, the caller invokes it with real inputs inside their own loop, and
every frame Dynamo produces is recorded. The session keeps the runtime guards
intact during capture, so later calls trigger the same recompilations as
ordinary ``torch.compile``, and applies its ``guard_filter_fn`` to the
serialized copy only; that needs a serialization-only filter in
``CheckFunctionManager`` (a ``serialization_guard_filter_fn`` beside the
runtime ``guard_filter_fn``), which arrives with the session. Nothing in this
stack has it: ``torch.compile``'s ``guard_filter_fn``, the only hook here and
the one this stack's tests pass ``default_guard_filter_fn`` through, removes a
rejected guard from the runtime check as well (``CheckFunctionManager.__init__``
builds the runtime guards from the filtered list), so a capture run that way
does not recompile when a guarded global function is rebound, serves the stale
graph, and records fewer variants than plain ``torch.compile`` would. Every
dropped guard is reported in ``PrecompileSummary.dropped_guards``.

The follow-up's session refuses an artifact by default when serialization
dropped a guard that looks configuration-dependent, rather than writing one
with variants whose dispatch would be ambiguous after load; nothing in this
stack refuses, it computes the inputs to that decision. Two tests feed it, and
their union is ``PrecompileSummary.risky_dropped_guards``. One is the
risky-drop lint over the dropped guard's binding site (``_is_risky_drop``,
later in this stack); it waives a builtin read, a
read off a torch- or stdlib-owned namespace and a global bound to a same-name
``def``, so a config-selected binding read off a trusted namespace passes it,
and it is a lint, not a proof. The other is decided per frame, by comparing the
guards of every captured variant of one frame: a guard that held identically
in every variant is a precondition the artifact is only valid under, one that
differed is what tells its graphs apart, and a dropped guard of the second kind
is what makes dispatch ambiguous. A serializable guard the session drops
because it held identically is reported apart, in
``PrecompileSummary.policy_dropped_guards``. Guards from different frames are
not comparable -- an entry frame guards its arguments, a resume frame guards
whatever crossed the break -- so the comparison never crosses frames.

Because capture is by execution, a resume function only exists once the frame
ahead of it has actually run, so every variant must be exercised. Whatever you
do not run is not in the artifact, and ``PrecompileSummary.complete`` means
complete only for the observed capture, not for every possible input to the
callable. A captured call that raises marks the session incomplete even if
caller code catches it.

Know these before relying on an artifact in production:

* Calls run with the grad mode the caller sets; capture forces neither
  ``no_grad()`` nor ``enable_grad()``. For an inference artifact run the calls
  under ``torch.no_grad()``. For a training artifact capture with
  ``training=True`` (see ``_capture_config``), which lowers the backward
  eagerly so the artifact carries one and a served output can be
  backpropagated -- without it, AOTAutograd defers the backward to the first
  ``.backward()`` call, so a grad-enabled capture that never makes one records
  no backends and cannot be written. No loss is needed: the joint trace
  synthesizes tangents from the forward outputs' own metadata.
* A non-tensor argument, and any value that crosses a graph break, is guarded
  by equality, so an int/bool/str argument or a break coming from ``.item()``
  yields an artifact that only serves calls reproducing those exact values.
  ``PrecompileSummary.wont_generalize`` lists them; exercise every value you
  need to serve with a captured call, or expect poor coverage on new data.
  ``dynamic=True`` un-pins an int or float, argument or crossed value alike,
  by making it symbolic from the first call, as the default's automatic
  dynamic does for an int from its second distinct value; a bool or str stays
  pinned either way.
* Identity guards (and the dict-version and weakref-liveness guards) cannot be
  serialized, so precompiling gives up on noticing that a guarded object was
  rebound, mutated or collected. ``PrecompileSummary.dropped_guards`` is the
  authoritative list. ``PrecompileSummary.risky_dropped_guards`` includes
  every drop observed to distinguish captured variants plus a lint for
  configuration-like sources (``_is_risky_drop``); it is still not a proof for
  unobserved deployments. The follow-up's public ``torch.compiler.precompile``
  facade rejects the RISKY subset by default. Refusing every drop is opt-in: every
  model drops the identity guards precompile cannot serialize, so
  ``require_no_dropped_guards=True`` refuses essentially every real artifact.
  Some models trip the lint on library internals. Measured on stock models
  when this was written, torchvision resnet18 and mobilenet_v3 report none,
  timm's ViT reports one (a re-exported ``torch._assert``) and transformers'
  Qwen2 reports 33 built from a two-layer config, 55 for the pretrained
  24-layer, of which only the attention-implementation registry looks genuinely
  config-selected. Report counts are per model, not per library: torchvision's
  efficientnet_b0 reports 2 and timm's swin reports 5, one of which is a real
  config slot. Audit the list before relying on the relaxed dropped-guard
  default, and before relaxing the risky-drop rail on top of it.
* Some models do not capture yet. For example, T5 raises ``PackageError: Cannot
  find module for code <code object __init__`` from ``_get_code_source`` in
  ``torch/_dynamo/package.py``, which plain ``caching_precompile`` also raises.
* The model must live in an importable module. Source is checksummed, so a
  class defined in ``__main__`` or a REPL cannot be loaded elsewhere.
* ``CompilePackage.install()`` writes compiled and resume functions into module
  globals and pushes the guarded entries onto the captured code objects, where
  every caller of those code objects dispatches through them. In this stack it
  begins with ``CompilePackage.uninstall()``, which clears the entry code
  object's entries and nothing pushed onto shared inner or resume code
  objects, so a second artifact installed on the same entry code takes the
  first's place. The follow-up scopes each entry to the isolated compile
  region owned by the runnable ``load`` returns (an ``isolate_recompiles_id``
  on the entry that the frame lookup matches against its own region only), so
  that loaded artifacts can share entry, inner and resume code objects without
  taking each other's entries, and the runnable's ``unload()`` removes only its
  own region and the globals it still owns; call that runnable rather than
  another instance of the same class.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import os
import re
import site
import sys
import sysconfig
import types
from typing import TYPE_CHECKING

import torch
from torch._guards import ChainedSource, Source
from torch.utils._config_module import ConfigModule

from .aot_compile import _BUILTINS_DICT_PREFIX, _IMPORT_ALIAS_PREFIX
from .guards import CheckFunctionManager
from .source import AttrSource, DictGetItemSource, GlobalSource, LocalSource


if TYPE_CHECKING:
    import traceback
    from collections.abc import Mapping, Sequence

    from torch.compiler._precompile_types import GuardFact as _GuardFact

    from .types import GuardFilterEntry


def default_guard_filter_fn(entries: Sequence[GuardFilterEntry], /) -> Sequence[bool]:
    """
    Drop every guard ``CheckFunctionManager.serialize_guards`` would refuse,
    and keep everything else.

    Read this before trusting an artifact. The refused set is the IDENTITY
    guards -- ID_MATCH, FUNCTION_MATCH, CLOSURE_MATCH, MODULE_MATCH, NN_MODULE,
    CLASS_MATCH -- plus DICT_VERSION (dict mutation) and WEAKREF_ALIVE
    (liveness); the module docstring says what dropping them gives up. Most
    such guards are on modules and are stable in practice, but one on a global
    holding a function is not: rebind it between capture and load and the
    artifact serves the graph traced against the old one, with no error.
    Keeping them instead makes serialization raise for essentially every
    function, so every drop is recorded with its source name in
    ``PrecompileSummary.dropped_guards``, and the only rail on by default is
    ``PrecompileSummary.risky_dropped_guards``, the drops the module's
    risky-drop lint flags or that differed between captured variants of one
    frame; a lint is not a proof.

    This mirrors the type tests of the three branches of the serializer's
    pre-check, the if/elif chain at the top of ``serialize_guards``, with one
    type added to the first branch's (DICT_KEYS_MATCH, below) and nothing past
    the chain: not the raise inside the first branch, which is the local-scope
    refusal below. A guard of a refused type is dropped, and so is
    a guard of another type that DERIVES one (a CONSTANT_MATCH on a code object
    runs through ID_MATCH), except that TYPE_MATCH and BUILTIN_MATCH are kept
    whatever they derive, as the chain takes their branch first: BUILTIN_MATCH
    is an ``id_match_unchecked`` that records ID_MATCH, but the builtin pickles
    by name, the artifact carries only save-time copies of the builtins dict
    (the guards' copy ``serialize_guards`` prunes to the names guards read, and
    the pickle-filtered copy ``get_runtime_env`` records for the bytecode), and
    both loaders rebind the guard's key to the dict derived from the loading
    scope's own ``__builtins__`` (``CompilePackage.install``,
    ``AOTCompiledFunction._seed_guard_scope``), the LIVE ``builtins.__dict__``
    unless the caller pre-bound another, so the guard rebuilt at load catches
    a builtin swapped afterwards (``test_aot_compile.py``
    ``test_kept_builtin_match_guard_reads_the_seeded_builtins_dict`` pins the
    live dict). DICT_KEYS_MATCH is kept by type for another reason. The derived
    types an entry carries are the UNSAVED build's: ``CheckFunctionManager``
    runs the pre-filter build with ``save_guards=False``, and a DICT_KEYS_MATCH
    on ``torch.utils._pytree.SUPPORTED_NODES`` promotes itself to DICT_VERSION
    in exactly that build, whereas the save build pins it to a keys-match
    (``guard._force_dict_keys_match``) that the pre-check accepts; dropping it
    on the unsaved build's DICT_VERSION discards a guard the artifact can carry
    and pads ``dropped_guards`` with it. It is not the only guard on that
    registry, and neither notices a change made before load: a load rebuilds
    the guard tree against the loading process's registry, so both bake its
    keys and length at that point. After load, the kept guards on the entries
    ``tree_flatten`` reads put a DictGuardManager over the registry whose
    length check notices a node registered (a late import) either way; the
    keys-match is what notices a same-count change of keys (one node
    deregistered, another registered).
    Check any new refused derived type against the save build before dropping
    on it. Past the chain the filter mirrors nothing, and the refusal that
    matters there is of local-scope types, which cannot be pickled by name. It
    has two paths: the chain's TYPE_MATCH/BUILTIN_MATCH branch raises when
    ``guard._unserializable`` is set, and ``GuardsStatePickler.reducer_override``,
    once none of its earlier branches has rebuilt the object, refuses a plain
    non-tuple instance of such a type anywhere in the guard tree, so a kept
    guard whose source walks through one fails there. A tuple instance passes
    that check: a local tuple subclass then fails in plain pickle, a
    PackageError wrapping the AttributeError, but only a filter that drops
    TYPE_MATCH gets there (every tuple-subclass value Dynamo reads carries its
    own TYPE_MATCH, which under this filter refuses it on the first path), and
    a local namedtuple has no refusal at all: its guard is a SEQUENCE_LENGTH,
    whose ``_unserializable`` the first branch never reads, its type is
    rebuilt from its fields by an earlier branch, and the rebuilt type is a
    new class, so the artifact ships and its type check never passes for the
    caller's own. FAKE_SCRIPT_TYPE_MATCH sets the same flag, which nothing outside
    that branch reads, so the pre-check passes a local-scope opaque-object
    type; its only installer, the opaque-object path of ``VariableBuilder``,
    guards a plain instance, which the pickler then refuses with the same
    message, so nothing ships. The pickler's refusal is not universal:
    the branches before it rebuild a local function by value, a local
    namedtuple type from its fields, and an ``nn.Module`` of a local class with
    the default ``__getstate__`` as a plain ``torch.nn.Module``, so for a
    module argument of a local class the kept TYPE_MATCH is the only refusal;
    drop it and the artifact serializes, then serves a differently typed module
    whose guarded attributes match, with no error. That is why the filter keeps
    every local-type guard on purpose although ``orig_guard._unserializable``
    would tell for the chain's path: dropping a guard on a type the artifact
    cannot pickle ships an artifact that never checks the type, whereas keeping
    it makes serialization refuse loudly. Passing this filter therefore does
    not mean the artifact serializes.
    Under ``torch._dynamo.config.caching_precompile``,
    ``CheckFunctionManager.__init__`` wraps the ``guard_filter_fn`` it is
    given, this one included: the caller's verdicts come first, then it also
    drops ID_MATCH, CLOSURE_MATCH, WEAKREF_ALIVE, DICT_VERSION and anything
    deriving ID_MATCH or DICT_VERSION, so its drops win and both exemptions
    above are void there: BUILTIN_MATCH goes through its derived ID_MATCH and
    the SUPPORTED_NODES DICT_KEYS_MATCH through the unsaved build's
    DICT_VERSION. That list reaches most refused types, since NN_MODULE,
    FUNCTION_MATCH, CLASS_MATCH and MODULE_MATCH all derive ID_MATCH (the first
    two through ID_MATCH, the other two through ``id_match_unchecked`` directly,
    which records the name ID_MATCH), but it keys on what the unsaved build
    recorded, so a refused guard whose build recorded nothing is dropped here
    by type and kept there, into a ``serialize_guards`` that refuses it by
    type: a CLASS_MATCH rooted at a TypeSource, which ``id_match_unchecked``
    turns into a TYPE_MATCH on a fresh guard (of those four only CLASS_MATCH
    can be rooted there, a TypeSource's value being a class), and an NN_MODULE
    that ``build_guards`` skipped under ``guard_nn_modules=False``.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type in ("TYPE_MATCH", "BUILTIN_MATCH", "DICT_KEYS_MATCH")
        or (
            g.guard_type not in unsupported
            and not any(d in unsupported for d in g.derived_guard_types)
        )
        for g in entries
    ]


def _owning_module(value: object) -> str | None:
    if isinstance(value, types.ModuleType):
        return value.__name__
    owner = getattr(value, "__module__", None)
    return owner if isinstance(owner, str) else None


def _source_root(source: Source) -> Source:
    return source.get_base() if isinstance(source, ChainedSource) else source


# The list of Dynamo-generated resume functions a nested resume function takes
# as its first parameter (resume_execution.py and comprehension_graph_break.py
# mint the name; codegen_call_resume in symbolic_convert.py builds the list). Its
# entries are generated code, not a slot any config chooses, so an identity
# guard lost on one cannot diverge. Its sibling __nested_frame_values is NOT
# here: it carries each enclosing frame's live stack and locals, so a guard
# rooted there is judged like the value it stands for.
_DYNAMO_SYNTHESIZED = ("__nested_resume_fns",)


def _is_dynamo_synthesized(source: Source) -> bool:
    root = _source_root(source)
    return isinstance(root, LocalSource) and root.local_name in _DYNAMO_SYNTHESIZED


# Belt and braces for an install directory none of the roots name: whichever
# layout put it there, a pip target still ends in one of these. Matched below
# the stdlib root a file was found under, not on the whole path, so a stdlib
# that itself sits under a site-packages directory keeps its waiver.
_INSTALL_DIR_NAMES = frozenset({"site-packages", "dist-packages"})


def _norm(path: str) -> str:
    """
    realpath then normcase. A relative path resolves against the process cwd,
    so a recorded ``__file__`` is gated with isabs before it gets here.
    """
    return os.path.normcase(os.path.realpath(path))


@functools.cache
def _stdlib_roots() -> tuple[str, ...]:
    """
    Where this interpreter's own library lives, sorted, so the first root a path
    lies under is the outermost (``_classify_file`` reads it that way). os is
    unquestionably stdlib, so its directory is the direct evidence and the only
    one that stays right when the stdlib is a zip, where that root is the whole
    archive and a third party bundled into it is waived with the stdlib;
    sysconfig and sys._stdlib_dir cover a build where os is frozen with no
    __file__. An install root can nest inside one of these (see
    ``_install_roots``), so a path under both is third party: an install root
    wins over a stdlib root.
    """
    roots = []
    os_file = getattr(os, "__file__", None)
    if os_file:
        # The directory the file resolves into, not the one it was imported
        # from: in a venv over a symlink-farm prefix (a Nix, Guix or Spack
        # profile) os.py is a per-file link into the store, sysconfig and
        # sys._stdlib_dir already name the farm, and every consumer normalizes
        # the file it asks about.
        roots.append(os.path.dirname(_norm(os_file)))
    frozen_dir = getattr(sys, "_stdlib_dir", None)  # 3.11+
    if frozen_dir:
        roots.append(frozen_dir)
    paths = sysconfig.get_paths()
    roots += [paths["stdlib"], paths["platstdlib"]]
    if sys.platform == "win32":
        # The stdlib's C extensions live beside Lib, not under it.
        roots.append(os.path.join(sys.base_prefix, "DLLs"))
    return tuple(sorted({_norm(p) for p in roots}))


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
    roots = [paths["purelib"], paths["platlib"]]
    for name in ("getsitepackages", "getusersitepackages"):
        try:
            got = getattr(site, name)()
            found = [got] if isinstance(got, str) else list(got)
        except Exception:
            continue  # an old-virtualenv site.py lacks it, or it cannot answer
        roots += [p for p in found if isinstance(p, str)]
    # On Windows getsitepackages() lists the bare prefix, which the whole stdlib
    # sits under; a directory a stdlib root lies under is not an install root.
    stdlib = _stdlib_roots()
    normed = {_norm(p) for p in roots}
    above = {r for r in normed for s in stdlib if s == r or s.startswith(r + os.sep)}
    return tuple(sorted(normed - above))


@functools.cache
def _torch_roots() -> tuple[str, ...]:
    """
    Every directory torch's own submodules come from. An editable build splits
    them -- torch/__init__.py out of the source tree, _C.so and version.py out
    of site-packages -- and torch.__path__ is exactly that set. The gate rules
    out a substituted sys.modules['torch'] only: its __path__ is ignored unless
    the torch package directory this file sits under (two levels up, past
    _dynamo) is among the entries, and then every entry is adopted, one a third
    party appended to the real torch's included.
    """
    own_file = globals().get("__file__")
    if not own_file:
        return ()  # frozen torch: no directory to anchor to
    own = _norm(os.path.dirname(os.path.dirname(own_file)))
    roots = {own}
    search = getattr(sys.modules.get("torch"), "__path__", None) or ()
    listed = {_norm(p) for p in search if isinstance(p, str)}
    if own in listed:
        roots |= listed
    return tuple(sorted(roots))


def _within(path: str, roots: tuple[str, ...]) -> bool:
    """Prefix test over ``_norm``-ed paths; the caller normalizes both sides."""
    return any(path == r or path.startswith(r + os.sep) for r in roots)


@functools.cache
def _classify_file(file: str, stdlib: bool) -> bool | None:
    """
    Shipped here (True), shipped elsewhere (False), or no evidence (None), for
    one ``__file__`` judged against the torch roots (``stdlib=False``) or the
    stdlib roots minus the install roots (``stdlib=True``). None is only ever
    a path that cannot be resolved; past those gates the torch arm never
    answers None: with no torch root (a frozen torch) every path is elsewhere,
    and ``_is_library_module`` waives torch names before asking. Cached on the
    __file__ string rather than on the module name: the roots are fixed for the
    process, so the answer for a path never changes, while the module a name
    resolves to can.
    """
    # Before 3.13 ntpath.isabs accepts a driveless \Lib\x.py (its LEGACY BUG
    # comment), which realpath then resolves against the current drive.
    driveless = sys.platform == "win32" and not os.path.splitdrive(file)[0]
    if driveless or not os.path.isabs(file):
        # Resolving it would be against a cwd that is not the one it was
        # recorded under, so it is evidence in neither direction.
        return None
    try:
        path = _norm(file)
    except ValueError:  # an embedded NUL, which posixpath.realpath lets through
        return None
    if not stdlib:
        return _within(path, _torch_roots())
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
    Shipped here (True), shipped elsewhere (False), or no evidence (None), for
    a sys.modules entry, which need not be a module. A ``__file__`` that
    ``_classify_file`` can place decides, and only then is the loader read:
    with importlib on disk, importlib/__init__.py gives importlib._bootstrap a
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
    # The module dict rather than getattr: a PEP 562 module __getattr__ is user
    # code, and a module that raises on an unknown attribute would take the
    # capture session down from inside a lint. sys.modules can hold any object,
    # so the __dict__ and spec.loader reads that remain are user code on the
    # wrong one and are caught.
    try:
        attrs = getattr(module, "__dict__", None) or {}
        file = attrs.get("__file__")
        if isinstance(file, str):  # os.path.isabs raises TypeError on anything else
            verdict = _classify_file(file, stdlib)
            if verdict is not None:
                return verdict
        # The loader rather than spec.origin: both importers build the spec as
        # spec_from_loader(name, cls, origin=cls._ORIGIN), so the two never
        # disagree, and the class is the stronger signal.
        spec = attrs.get("__spec__")
        loader = attrs.get("__loader__") or getattr(spec, "loader", None)
    except Exception:
        return None
    if loader is importlib.machinery.BuiltinImporter:
        # Statically linked, and BuiltinImporter precedes PathFinder on
        # sys.meta_path, so on import no file on sys.path is reachable under
        # this name; a spec assigned straight into sys.modules is taken at its
        # word. The inittab is keyed on the full dotted name.
        return name in sys.builtin_module_names
    if loader is importlib.machinery.FrozenImporter:
        # frozen also precedes the path finder
        return importlib.machinery.FrozenImporter.find_spec(name) is not None
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
    path finder cannot shadow. That is required of the TOP-LEVEL name: not
    imported, a namespace package, or without location evidence, it is
    untrusted. An imported inner name only has to not be located ELSEWHERE:
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
    why the name compared is ``__qualname__``. The file is read off the code
    object, not off ``__module__``: functools.wraps copies ``__module__`` along
    with ``__name__`` and ``__qualname__``, so ``op = torch.compile(op)`` behind
    a flag claims the reader's module while its code lives in eval_frame.py.
    The object does not tell that shape from an unconditional cross-file
    decorator, so ``@torch.no_grad()`` on a same-file def is not waived either.
    A class has no code object, and its ``__module__`` is no better: namedtuple
    and ``type()`` stamp it from the calling frame (make_dataclass does from
    3.12) under a BARE ``__qualname__`` (a def or class statement inside the
    factory would carry ``factory.<locals>.``), so ``Point = lib.make_point()``
    in the reader looks exactly like a class statement. Its methods can tell: a
    class statement compiles its defs in its own file under its own
    ``__qualname__`` prefix, so a class is waived when at least one function in
    its own ``__dict__``, stored under key ``k`` with ``__qualname__`` ``Cls.k``
    (a staticmethod or classmethod is unwrapped through ``__func__`` and a
    property through ``fget``, by type rather than by ``getattr``, which a proxy
    attribute such as ``torch.classes.<ns>`` answers by raising; a
    cached_property keeps its function under ``.func`` and does not count), was
    compiled in the reading file. A function attached afterwards keeps its bare
    qualname, so an imported class the reader extends (``Point.extra =
    _extra``) and a factory fed same-file methods (``type(name, bases, {"area":
    _area})``, ``make_dataclass(..., namespace=...)``) are refused, and ``class
    Marker: pass`` fails closed. So does a class statement whose only functions are
    generated -- a fields-only ``@dataclass``, a ``NamedTuple``, an ``Enum``
    -- because those methods compile in ``<string>`` or the stdlib, so a plain
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
    resolve either against the cwd, so a fields-only dataclass read from an
    exec-generated frame would collide with it and be waived. Only absolute
    filenames on both sides compare; anything else fails closed. What this
    cannot see is a same-name fork inside the reading file -- ``try: from x
    import impl as op`` / ``except ImportError: def op``, or a class statement
    under the same ``if`` -- which binds a different def per machine under one
    checksum; that is the conditional-bind KNOWN GAP recorded in
    ``_is_risky_drop``.
    """
    if not user_stack or getattr(value, "__qualname__", None) != global_name:
        return False
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
            if attr.__qualname__ == f"{global_name}.{key}":
                files.append(attr.__code__.co_filename)
    else:
        code = getattr(value, "__code__", None)
        files = [code.co_filename] if isinstance(code, types.CodeType) else []
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
    does not carry it; unmangling collides only for a module literally named
    ``a_dot_b``.
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
    ``import mypkg.layers``, and the ``__import_x`` alias Dynamo installs to
    reach an inlined function's own globals -- or if it is an attribute of a
    trusted module under a name that module already owns: its own ``__name__``
    (a plain ``import own_sub`` inside the parent) or the parent's plus the
    attribute (``from . import sub``). An ALIASED user module is none of those:
    ``if flag: import impl_b as impl`` picks what ``impl.op`` resolves to per
    machine, and so does the same alias spelled ``from . import impl_b as
    impl`` in a package __init__. Inheriting the parent's trust without
    checking the name is what let that shape through before.

    Library ownership is trusted under ANY binding, ``import torch.nn.functional
    as F`` included; the price, taken deliberately because flagging ``F`` would
    flag every model, is that an alias config picks between two torch modules
    is waived too (see ``_is_risky_drop``'s KNOWN GAP). A config module is the
    one namespace whose bindings config chooses by definition, so it is never
    trusted whoever owns it. Keys are source names, and the consumer looks a
    read's ``source.base.name`` up exactly, never by prefix.
    """
    modules = {
        e.orig_guard.originating_source.name: (e.orig_guard.originating_source, e.value)
        for e in entries
        if isinstance(e.value, types.ModuleType)
        and not isinstance(e.value, ConfigModule)
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
            if _is_library_module(module.__name__):
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
    shim can point elsewhere; only a builtin read under its own name is waived
    (``IOError``, CPython's alias of ``OSError``, fails closed), and only one
    CPython built: functools.wraps copies ``__module__`` and ``__name__`` onto
    ``builtins.sum = wraps(sum)(logged_sum)``, a Python function whose CLOSURE_MATCH
    is dropped, so the value must be a builtin function or a type as well
    (every callable in ``builtins.__dict__`` is one, the ``_sitebuiltins``
    objects aside, and those fail the ``__module__`` test). The exposure is
    narrow either way: a registered builtin is id-matched into a BUILTIN_MATCH
    the serializer keeps, so only a deregistered, polyfilled one (``sum``,
    ``enumerate``) or a shim reaches the dropped set this lint examines.

    A builtin parked in a slot -- ``self.act = abs``, straight out of an
    ACT2FN-style table -- is a slot like any other, so this deliberately keys
    on where the read comes FROM rather than on who owns the value.
    """
    return (
        isinstance(source, DictGetItemSource)
        and isinstance(source.base, GlobalSource)
        and source.base.global_name.startswith(_BUILTINS_DICT_PREFIX)
        and isinstance(value, (types.BuiltinFunctionType, type))
        and _owning_module(value) == "builtins"
        and getattr(value, "__name__", None) == source.index
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
    ``own_helpers.call`` is waived because own_helpers owns a def of that same
    name, subject to the gap below. ``mypkg.op`` re-exported from
    ``mypkg.impl_b``, ``dispatch.op``, ``own_helpers.act`` bound to some other
    def, and ``mypkg.impl.op`` where ``mypkg/__init__`` did ``from . import
    impl_b as impl`` are not waived: the import or assignment that chose the
    implementation lives in a file the inlined-source checksum never sees, so
    capture and serve can disagree with every other rail passing.
    ``test_risky_drop_decision_table`` in test_precompile_package.py pins these
    shapes and every other one found so far.

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
      torch function it wraps. So does an alias config picks between two torch
      modules: ``F`` bound to torch.nn.functional or torch._refs.nn.functional
      is trusted either way, and ``F.gelu`` names a different function.

    An ``allow_in_graph`` function passes too, and Dynamo traces it opaquely so
    the inlined-source checksum never covers it either. Nothing at capture time
    distinguishes a conditional bind from an unconditional one -- only the
    resulting object is visible -- so this is a limit of the approach rather
    than a missing check. ``dropped_guards`` is the authoritative list; this
    predicate is a lint over it, not a proof of safety.
    """
    source = entry.orig_guard.originating_source
    value = entry.value
    stack = entry.orig_guard.user_stack
    if _is_dynamo_synthesized(source):
        return False
    if source.name in namespaces:
        return False
    if _reads_a_builtin(source, value):
        return False
    if not entry.has_value:
        return True  # nothing to judge ownership by
    if isinstance(source, AttrSource):
        namespace = namespaces.get(source.base.name)
        if namespace is not None:
            return not (
                _is_library_module(namespace.__name__)
                or (
                    _owning_module(value) == namespace.__name__
                    and getattr(value, "__name__", None) == source.member
                )
            )
    if (
        isinstance(source, GlobalSource)
        and getattr(value, "__name__", None) == source.global_name
    ):
        return not (
            _is_library_module(_owning_module(value))
            or _defined_where_read(value, source.global_name, stack)
        )
    return True


# The guards that pin a Python value by equality. CONSTANT_MATCH is the front
# door: it delegates a bool to BOOL_MATCH, None to NONE_MATCH, a code object to
# ID_MATCH and everything else -- ints included -- to EQUALS_MATCH, and it is
# the only guard method that derives into another one of these, so a slot's
# top-level guard type is enough here. CONSTANT_SUBCLASS_MATCH pins the base
# value of an int/float/str subclass argument; the two iterator guards pin an
# iterator argument's exact position. Length and key-set guards
# (SEQUENCE_LENGTH, TUPLE_ITERATOR_LEN, MAPPING_KEYS_CHECK) pin a container's
# structure rather than a value and are deliberately not counted.
_VALUE_EQUALITY_GUARD_TYPES = frozenset(
    {
        "CONSTANT_MATCH",
        "CONSTANT_SUBCLASS_MATCH",
        "COUNT_ITERATOR_MATCH",
        "EQUALS_MATCH",
        "RANGE_ITERATOR_MATCH",
    }
)


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
# Dynamo appends a per-process counter to the builtins dict it installs, so the
# same guard reads __builtins_dict___6 in one compilation and ___8 in the next.
# Of the globals Dynamo mints, only the three families in
# aot_compile._MINTED_GLOBAL_PREFIXES can root a serializable guard: this one and
# the two install_global_by_id shapes the pattern below covers. __compiled_fn_*
# and __resume_at_* are codegen-only LOAD_GLOBAL targets, never guard subjects.
_DYNAMO_COUNTER = re.compile(re.escape(_BUILTINS_DICT_PREFIX) + r"_\d+")
# OutputGraph.install_global_by_id names a global "<prefix>_<id(value)>_c<n>",
# so a guard reading one carries BOTH an address and a compile counter inside
# an identifier, where neither pattern above can see it. Real models reach this
# -- transformers' Qwen2 installs three -- and the report then differs run to
# run, which is exactly what the "commit and diff" contract rules out. The
# prefix can be empty (torch itself is installed as "_<id>_c<n>"), so the digit
# width is the anchor: it is what keeps a user identifier such as w_1_c2 intact.
_DYNAMO_GLOBAL_BY_ID = re.compile(r"_\d{9,}_c\d+\b")


def _normalize(text: str) -> str:
    text = _SAVED_HOOK_IDS.sub("(<ids>)", text)
    text = _DYNAMO_GLOBAL_BY_ID.sub("_<id>_c<n>", _OBJ_ID.sub("<id>", text))
    return _DYNAMO_COUNTER.sub(_BUILTINS_DICT_PREFIX + "_<n>", text)


def _render_code(code_list: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(_normalize(part) for part in (code_list or ()))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# Guards whose check IS object identity, directly or through a derived guard,
# which is the same test default_guard_filter_fn drops on.
_IDENTITY_GUARD_TYPES = frozenset(
    CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
)


# Ellipsis and NotImplemented repr by name, so they are as stable as a literal.
_STABLE_CONST_TYPES = (
    str,
    int,
    float,
    complex,
    bytes,
    bool,
    type(None),
    type(Ellipsis),
    type(NotImplemented),
)


def _stable_consts(consts: tuple[object, ...]) -> tuple[object, ...]:
    """
    co_consts reduced to the part that reprs the same in every process.

    A nested code object reprs with its ADDRESS, so it cannot go into a digest
    that ends up in a file meant to be committed and diffed. Containers are
    filtered recursively rather than dropped whole: two lambdas differing only
    in a tuple or frozenset constant -- ``x * (1, 2)`` against ``x * (1, 3)`` --
    are genuinely different variants, and dropping the container is what let
    them collide. A const of any other type keeps its SLOT as a type marker:
    ``x[..., 0]`` and ``x[0, ...]`` fold to one const tuple at the same index,
    so with the slot dropped they would collide the same way.
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
        else:
            out.append(f"<{type(c).__name__}>")
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

    A qualname alone does not separate the case this exists for: an ACT2FN-style
    table whose entries are all ``<lambda>`` in one module, where two variants
    holding different entries would render identically and the CLOSURE_MATCH
    that split them would be reported as an invariant of both. So a callable is
    also named by where it is DEFINED (basename, so no checkout path) and by a
    digest of its body, both source-derived and so stable across processes.

    The discriminating part goes FIRST: truncation bounds this string, and a
    transformers lambda nested in a long module path exceeds the limit on the
    qualname alone, so a digest appended at the end would be cut off exactly on
    the names that need it most.

    Only the code object is named, not the data bound to it. Two closures from
    one factory (``make(2)`` and ``make(3)``), two functions differing only in
    ``__defaults__``, bound methods of two instances, and every
    ``functools.partial`` render the same, as do two instances of one class.
    """
    if isinstance(value, types.ModuleType):
        return f"is module {value.__name__}"
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(name, str):
        code = getattr(value, "__code__", None)
        where = ""
        if isinstance(code, types.CodeType):
            filename = os.path.basename(code.co_filename or "?")
            where = f"@{filename}:{code.co_firstlineno}#{_code_fingerprint(code)} "
        return _normalize(f"is {where}{_owning_module(value) or '?'}.{name}")[:160]
    return f"is a {type(value).__module__}.{type(value).__qualname__}"[:160]


# Guards that pin an input's SHAPE, VALUE or KIND, never policy-dropped even
# when they held identically across every captured variant. A drop is licensed
# by "it discriminated nothing", but with a single example nothing CAN
# discriminate, and what would disappear is the check that the runtime input
# looks like the captured one at all. A dropped shape guard crashes inside a
# kernel on inductor and can quietly miscompute on eager; a dropped value guard
# serves the captured branch to every other value with correct-looking numerics.
_SHAPE_BEARING_GUARD_TYPES = frozenset(
    {
        "TENSOR_MATCH",
        "SEQUENCE_LENGTH",
        # Python values the graph specialized on: an int or bool argument,
        # module.training, an .item() result, mask=None.
        "CONSTANT_MATCH",
        "EQUALS_MATCH",
        # Pins that two inputs alias, so a graph traced under `x is y` is never
        # served two distinct tensors.
        "DUPLICATE_INPUT",
        # hasattr is a branch like any other. Reachable on the DEFAULT gates: a
        # single-variant capture makes every slot look invariant and the drop
        # is not classed risky.
        "HASATTR",
        # An input's KIND: a graph traced for one class and served to another
        # returns the first one's answer, with no shape to crash on. Upstream
        # leans on this guard specifically: VariableBuilder.wrap_tensor relaxes
        # an AsyncCollectiveTensor's class guards (UnwrapCollectiveTensorSource)
        # so an ACT-traced graph serves the resolved tensor, and
        # BuiltinVariable.call_isinstance reinstalls TYPE_MATCH where the class
        # is observed. FAKE_SCRIPT_TYPE_MATCH is the same pin for a
        # reference-type opaque object.
        "TYPE_MATCH",
        "FAKE_SCRIPT_TYPE_MATCH",
        # The graph specialized on utils_device.CURRENT_DEVICE: captured under
        # the default None and served under set_default_device("cuda"), it
        # returns CPU tensors with no refusal.
        "DEFAULT_DEVICE",
        # Membership, key-set, length and iterator-position facts, each a branch
        # the graph specialized on. A module-owned dict (self.opts = {}) is
        # environment-rooted, which is exactly where a policy would drop it.
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
        # A SEQUENCE_LENGTH on a module's hook dicts when
        # skip_nnmodule_hook_guards is off, and nothing under the default: so
        # either there is nothing to drop, or what there is pins a value.
        "EMPTY_NN_MODULE_HOOKS_DICT",
        # Pins a folded torch._C._is_cow_tensor branch. Kept, a capture that
        # folded one fails at serialization with the builder's own error (the
        # tensor comes back fake and COW_TENSOR_MATCH rejects that); dropped, it
        # would serve the folded branch to the other kind of tensor silently.
        "COW_TENSOR_MATCH",
    }
)


# Guards whose C++ leaf compares something no fingerprint in this module reads:
# subclass metadata, a DTensor placement, an opaque object's guard values, a raw
# DispatchKeySet, the symbolic shape environment, or process-wide state the leaf
# snapshots for itself (GlobalStateGuard's state, the torch-function mode
# stack). Calling two of these equal is how the report ends up asserting a
# precondition that does not hold, so they are never compared and are reported
# as undetermined. They are never dropped either: a policy may drop only what
# its droppable set names, and these are in no such set.
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
    # EMPTY_NN_MODULE_HOOKS_DICT is classified shape-bearing for the config
    # where it emits a check; under skip_nnmodule_hook_guards, the default,
    # GuardBuilder emits nothing for it, so a report must not call it a
    # precondition.
    return guard_type in _NOOP_GUARD_TYPES or (
        guard_type == "EMPTY_NN_MODULE_HOOKS_DICT"
        and torch._dynamo.config.skip_nnmodule_hook_guards
    )


# The ONLY guard types the invariance policy may drop, and only when proven
# invariant across every captured variant: the identity guards the default
# filter drops anyway as unserializable, and process-wide compiler state. The
# four sets are a total, disjoint classification of GuardBuilder's guard
# methods, pinned by test_guard_policy_classification_is_total: a guard type in
# none of them -- any type added to GuardBuilder after this list -- is KEPT
# unconditionally until someone classifies it, so a new value-pinning guard can
# never become silently droppable. Guards installed outside GuardBuilder (the
# root manager's DuplicateInputs and StorageOverlap exprs, the dimension-marking
# lambda) never reach the guard filter and are outside the policy as well.
_INVARIANT_DROPPABLE_GUARD_TYPES = _IDENTITY_GUARD_TYPES | frozenset(
    {
        "AUTOGRAD_SAVED_TENSORS_HOOKS",
        # An identity match on a builtin, which the default filter keeps.
        "BUILTIN_MATCH",
        "DUAL_LEVEL",
        "FUNCTORCH_STACK_MATCH",
    }
)


def _saved_hooks_fingerprint() -> str:
    """
    Name the installed saved-tensors hooks the way the guard compares them.

    The guard stores ``tuple(map(id, hooks))`` when both hooks are fx
    GraphModules and ``None`` otherwise, so plain-Python hooks, and no hooks at
    all, are one value to it and must be one value here, or the report shows a
    'varies' line for a guard that passes either way. Inlineable hooks are
    named by their rendered graph rather than by address, since an id cannot
    go in a committed, diffable file. KNOWN TRADEOFF: two distinct GraphModules
    with identical code read as one hook set here while the guard tells them
    apart, so a policy may drop that guard on the strength of this fingerprint.
    """
    try:
        from torch._functorch._aot_autograd.utils import (
            saved_tensors_hooks_are_inlineable,
            top_saved_tensors_hooks,
        )

        hooks = top_saved_tensors_hooks()
        if not saved_tensors_hooks_are_inlineable(hooks):
            return "hooks=None"
        return "hooks=(" + ", ".join(_hash_text(hook.code) for hook in hooks) + ")"
    except Exception:
        # Distinct from the "" that means "the rendered code already names the
        # check": a failed read must not merge two variants.
        return "hooks=<unreadable>"


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
    as dtype does. KNOWN GAP: that leaf checks nothing for a dim the compile
    made dynamic, so under ``dynamic=True`` the concrete shape here is narrower
    than the guard and a shape-generic TENSOR_MATCH is reported as varying
    rather than invariant.

    An identity guard needs one too, because ``_normalize`` strips the id its
    code renders: without a name for the object, two variants holding different
    callables at one source collapse into one fact and are reported as an
    invariant neither of them holds.

    Every other guard takes its value from its own rendered code, which names
    it, so fingerprinting it again SPLITS identical guards: TYPE_MATCH on an
    unspecialized int checks only that the int is an int, and stamping 1 on one
    variant and 2 on the next demotes a real invariant into two identical
    'varies' lines. So every branch dispatches on the guard type, never on the
    value's: the NOT_NONE_MATCH Dynamo installs on an optimizer's .grad, or a
    TYPE_MATCH on a tensor attribute, holds a tensor whose shape and dtype it
    never checks.
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
    if entry.guard_type == "TENSOR_MATCH" and isinstance(value, torch.Tensor):
        # Render exactly what TensorCheck stores (notably the TLS-adjusted
        # dispatch key set, not the tensor's own) rather than reconstructing it.
        # The fact's source already names the tensor, so no name goes in.
        from .guards import convert_to_concrete_values, get_tensor_guard_code_part

        try:
            return get_tensor_guard_code_part(
                value,
                "<value>",
                convert_to_concrete_values(value.size()),
                convert_to_concrete_values(value.stride()),
                type(value),
                torch._C._dispatch_keys(value),
            )
        except Exception:
            # A subclass whose __torch_function__ refuses attribute reads got
            # here; type() is the one read that cannot raise.
            return f"type={type(value).__name__}, <unrenderable>"
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

    Cancellation is per frame, like every comparison in this module: a bare
    name means nothing across frames, and ``___stack0`` names whatever crossed
    the break in EVERY resume frame -- a tensor in one, an ``.item()`` int in
    the next -- so a generic mention elsewhere must not erase a real pin here.
    ``GuardFact.source`` and a kept slot's name share one spelling, the
    ``GuardFilterEntry.name`` with local scope stripped (``L['x']`` -> ``x``).
    """
    pinned = {n for t, n in kept if _pins_a_value(t, n)}
    if not pinned:
        return ()
    # A source survives if SOME frame pins it and never serves it generically;
    # a frame that does both cancels only its own pin, not another frame's.
    survivors: set[str] = set()
    for variants in guard_sets.values():
        pins: set[str] = set()
        generic: set[str] = set()
        for facts in variants:
            here = {f.source for f in facts if _pins_a_value(f.guard_type, f.source)}
            pins |= here
            # A variant that reached the source without pinning it is the
            # graph that serves other values.
            generic |= {f.source for f in facts} - here
        survivors |= pins - generic
    return tuple(sorted(pinned & survivors))
