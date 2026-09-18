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
import importlib.machinery
import os
import site
import sys
import sysconfig
import types
from typing import TYPE_CHECKING

from .aot_compile import _BUILTINS_DICT_PREFIX, _IMPORT_ALIAS_PREFIX
from .guards import CheckFunctionManager
from .source import DictGetItemSource, GetItemSource, GlobalSource, LocalSource


if TYPE_CHECKING:
    import traceback
    from collections.abc import Iterable, Sequence

    from torch._guards import Source

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
