"""
Helpers behind the multi-graph precompile: the capture of a callable into one
artifact holding every frame Dynamo produces while the caller's calls run --
the entry frame, the ``torch_dynamo_resume_in_*`` continuations graph breaks
create, and the recompiled variants of each -- stored through CompilePackage
(``torch/_dynamo/package.py``), a low-level component not meant to be used
directly. It is not ``torch.compiler.precompile``, the ahead-of-time capture
API this repository already has (``torch/_precompile.py``), which does not call
into this module; nor ``torch._dynamo.config.caching_precompile``, which caches
``torch.compile`` artifacts transparently without an explicit capture and, when
set, wraps every guard filter, this module's included (see
``default_guard_filter_fn``).

Over the stack that adds it, this module comes to hold the guard filter for
the serialized guards (``default_guard_filter_fn``), the lint over the identity
guards it drops (``_is_risky_drop``), the fingerprints and the guard-type
classification behind the ``PrecompileSummary`` report -- which also decides
the only guards an invariance policy may drop
(``_INVARIANT_DROPPABLE_GUARD_TYPES``) -- the per-frame comparison of captured
variants and the summary builder (``_varying_guard_slots``, ``_summarize``),
and the compiler configuration and frame converter a capture runs under
(``_capture_config``, ``_AllowEmptyGraphsConvertFrame``). The filter lives
here, with the rest of the capture's guard tooling, rather than beside the
serializer's pre-check in ``guards.py``: it is the capture's policy over that
pre-check, not part of it.
Everything here is internal; the filter alone is unprefixed because the
capture session passes it as the default a caller may name. The multi-graph
Dynamo capture session that drives them is a follow-up stack; nothing under
``torch/`` calls into this module yet.
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
from torch._guards import ChainedSource
from torch.utils._config_module import ConfigModule

from .aot_compile import _BUILTINS_DICT_PREFIX, _IMPORT_ALIAS_PREFIX
from .guards import CheckFunctionManager
from .source import (
    AttrSource,
    DictGetItemSource,
    GetItemSource,
    GlobalSource,
    LocalSource,
)


if TYPE_CHECKING:
    import traceback
    from collections.abc import Iterable, Sequence

    from torch._guards import Source
    from torch.compiler._precompile_types import GuardFact as _GuardFact

    from .types import GuardFilterEntry


def default_guard_filter_fn(guard_entries: Sequence[GuardFilterEntry]) -> list[bool]:
    """
    Drop every guard ``CheckFunctionManager.serialize_guards`` would refuse for
    its type or a derived type, and keep everything else.

    The refused types are ``UNSUPPORTED_SERIALIZATION_GUARD_TYPES``: ID_MATCH,
    FUNCTION_MATCH, MODULE_MATCH, NN_MODULE and CLASS_MATCH, which check the
    guarded object's identity, CLOSURE_MATCH, which checks a function by its
    ``__code__`` id, plus DICT_VERSION and WEAKREF_ALIVE. Dropping one gives up
    on noticing that the guarded object was rebound, mutated or collected:
    rebind a global function between capture and load and the artifact serves
    the graph traced against the old one, with no error
    (``test_default_guard_filter_through_serialize_guards``). Every dropped
    slot is reported in ``PrecompileSummary.dropped_guards``, once however many
    variants dropped it.

    The criterion is the serializer's own pre-check over the entry's type and
    derived types: a guard is dropped if its type is refused or one of its
    derived types is (a CONSTANT_MATCH on a code object runs through
    ID_MATCH), and TYPE_MATCH and BUILTIN_MATCH are kept whatever they derive,
    as the pre-check accepts them before it looks at derived types. That is
    what keeps BUILTIN_MATCH, an ``id_match_unchecked`` deriving ID_MATCH; the
    loaded artifact checks the builtin against the loading process's builtins,
    so it still notices one swapped after load. Neither that accepted-by-type
    branch nor the DICT_KEYS_MATCH keep below holds under
    ``torch._dynamo.config.caching_precompile``: ``CheckFunctionManager``
    wraps every guard filter under that setting and drops, with a warning, any
    guard of type ID_MATCH, CLOSURE_MATCH, WEAKREF_ALIVE or DICT_VERSION and
    any guard deriving ID_MATCH or DICT_VERSION. The one departure from the
    pre-check is a DICT_VERSION derived by a DICT_KEYS_MATCH, which is
    ignored: the entries this filter sees carry the derived types of the build
    ``CheckFunctionManager`` runs before filtering, with ``save_guards=False``,
    where a DICT_KEYS_MATCH on ``torch.utils._pytree.SUPPORTED_NODES`` is
    promoted to a DICT_VERSION, while the save build pins it to the keys-match
    the pre-check accepts
    (``test_default_guard_filter_keeps_the_pytree_registry_keys_match``). That
    pair only: a DICT_KEYS_MATCH deriving another refused type, and any other
    type deriving DICT_VERSION, are dropped as the pre-check would refuse them
    (``test_default_guard_filter_drops_the_unserializable_types``).

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
    # spans _classify_file, so a raise out of the classifier's own gates lands
    # here as None rather than escaping a lint.
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
        # and the class is the stronger signal. A truthy __loader__ skips the
        # spec's loader, the only user-code half of this; a falsy one falls
        # through to it, and a hand-made ModuleType has the None its __init__
        # seeds (an imported module's dict carries spec.loader instead, copied
        # in by _init_module_attrs).
        spec = attrs.get("__spec__")
        loader = attrs.get("__loader__") or getattr(spec, "loader", None)
    except Exception:
        return None
    # Both arms compare by identity: under == a __loader__ whose __eq__ answers
    # true for anything would take the waiver, and one whose __eq__ raises would
    # escape this function, the catch being closed above.
    if loader is importlib.machinery.BuiltinImporter:
        # Statically linked, and BuiltinImporter precedes PathFinder on
        # sys.meta_path, so on import no file on sys.path is reachable under
        # this name; a spec assigned straight into sys.modules is taken at its
        # word. The inittab is keyed on the full dotted name.
        return name in sys.builtin_module_names
    if loader is importlib.machinery.FrozenImporter:
        # frozen also precedes the path finder
        return importlib.machinery.FrozenImporter.find_spec(name) is not None
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
