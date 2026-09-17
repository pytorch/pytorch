"""
Helpers behind the multi-graph precompile: the capture of a callable into one
artifact holding every frame Dynamo produces while the caller's calls run --
the entry frame, the ``torch_dynamo_resume_in_*`` continuations graph breaks
create, and the recompiled variants of each -- stored through CompilePackage
(``torch/_dynamo/package.py``), a low-level component not meant to be used
directly.

This module holds the guard filter for the serialized guards
(``default_guard_filter_fn``), the lint over the guards it drops
(``_is_risky_drop``), the guard-type classification and fingerprints behind
the ``PrecompileSummary`` report, the per-frame comparison of captured
variants and the summary builder (``_varying_guard_slots``, ``_summarize``),
and the compiler configuration and frame converter a capture runs under
(``_capture_config``, ``_AllowEmptyGraphsConvertFrame``). Everything here is
internal. The capture session that drives them, ``torch.compiler.precompile``,
is a follow-up; nothing under ``torch/`` calls into this module yet. It is
distinct from ``torch._dynamo.config.caching_precompile``, which caches
``torch.compile`` artifacts transparently without an explicit capture.
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

from torch._guards import ChainedSource, Source
from torch.utils._config_module import ConfigModule

from .aot_compile import _BUILTINS_DICT_PREFIX, _IMPORT_ALIAS_PREFIX
from .guards import CheckFunctionManager
from .source import AttrSource, DictGetItemSource, GlobalSource, LocalSource


if TYPE_CHECKING:
    import traceback
    from collections.abc import Sequence

    from .types import GuardFilterEntry


def default_guard_filter_fn(entries: Sequence[GuardFilterEntry], /) -> Sequence[bool]:
    """
    Drop every guard ``CheckFunctionManager.serialize_guards`` would refuse for
    its type or a derived type, and keep everything else.

    The refused types are ``UNSUPPORTED_SERIALIZATION_GUARD_TYPES``: the
    identity guards ID_MATCH, FUNCTION_MATCH, CLOSURE_MATCH, MODULE_MATCH,
    NN_MODULE and CLASS_MATCH, plus DICT_VERSION and WEAKREF_ALIVE. Dropping
    one gives up on noticing that the guarded object was rebound, mutated or
    collected: rebind a global function between capture and load and the
    artifact serves the graph traced against the old one, with no error
    (``test_default_guard_filter_through_serialize_guards``). Every drop is
    reported in ``PrecompileSummary.dropped_guards``.

    The test is the serializer's own pre-check over the entry's type and
    derived types: a guard is dropped if its type is refused or one of its
    derived types is (a CONSTANT_MATCH on a code object runs through
    ID_MATCH), and TYPE_MATCH and BUILTIN_MATCH are kept whatever they derive,
    as the pre-check accepts them before it looks at derived types. That is
    what keeps BUILTIN_MATCH, an ``id_match_unchecked`` deriving ID_MATCH; the
    loaded artifact checks the builtin against the loading process's builtins,
    so it still notices one swapped after load. The one departure from the
    pre-check is a DICT_VERSION derived by a DICT_KEYS_MATCH, which is
    ignored: the entries this filter sees carry the derived types of the build
    ``CheckFunctionManager`` runs before filtering, with ``save_guards=False``,
    where a DICT_KEYS_MATCH on ``torch.utils._pytree.SUPPORTED_NODES`` is
    promoted to a DICT_VERSION, while the save build pins it to the keys-match
    the pre-check accepts
    (``test_default_guard_filter_keeps_the_pytree_registry_keys_match``).

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
    for g in entries:
        derived = g.derived_guard_types
        if g.guard_type == "DICT_KEYS_MATCH":
            derived = tuple(d for d in derived if d != "DICT_VERSION")
        keep.append(
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


def _norm(path: str) -> str:
    """
    realpath then normcase. A relative path resolves against the process cwd,
    so a recorded ``__file__`` or a ``__path__`` entry is gated with isabs
    before it gets here.
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
    would resolve against the process cwd and make it a torch root.
    """
    own_file = globals().get("__file__")
    if not own_file:
        return ()  # frozen torch: no directory to anchor to
    own = _norm(os.path.dirname(os.path.dirname(own_file)))
    roots = {own}
    search = getattr(sys.modules.get("torch"), "__path__", None) or ()
    listed = {_norm(p) for p in search if isinstance(p, str) and os.path.isabs(p)}
    if own in listed:
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
    if "\x00" in file:
        # No file has one. posixpath.realpath raises ValueError on it from lstat;
        # from gh-106242 on, ntpath.realpath swallows that and returns the path
        # unresolved, whose prefix _within would then match.
        return None
    path = _norm(file)
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
    # code a lint must not run (torch.ops's setattrs a fresh _OpNamespace for
    # any name asked of it). And object's read of the dict, not the module's:
    # importlib.util.LazyLoader leaves a _LazyModule whose __getattribute__
    # executes the module body on ANY attribute read, __dict__ included, while
    # its dict already carries the seeded __file__, __spec__ and __loader__.
    # sys.modules can hold any object, so that read on a slotted proxy and
    # spec.loader on a hand-rolled spec are user code, and are caught.
    try:
        attrs = object.__getattribute__(module, "__dict__")
        file = attrs.get("__file__")
        if isinstance(file, str):  # os.path.isabs raises TypeError on anything else
            verdict = _classify_file(file, stdlib)
            if verdict is not None:
                return verdict
        # The loader rather than spec.origin: both importers' find_spec pass
        # origin=cls._ORIGIN to spec_from_loader, so the two never disagree,
        # and the class is the stronger signal. __loader__ first: a module
        # that carries one never has its spec read.
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
    ``a_dot_b``, whose alias comes back as ``a.b`` when that module is loaded
    too, and that one collision fails open (the callers judge the module
    returned).
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
    ``builtins.sum = wraps(sum)(logged_sum)``, a Python function whose
    CLOSURE_MATCH is dropped, and a class statement exec'd with the builtins
    namespace as its globals (or handed ``__module__ = "builtins"``) claims the
    module outright, so the value must be a builtin function or a static type
    as well, one carrying ``Py_TPFLAGS_IMMUTABLETYPE``, which no class statement
    or ``type()`` call gets (every callable in ``builtins.__dict__`` is one of
    the two, apart from the ``_sitebuiltins`` objects, which fail the
    ``__module__`` test, and ``ExceptionGroup``, the one heap type among them,
    which fails closed). The exposure is
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
        and (
            isinstance(value, types.BuiltinFunctionType)
            # Py_TPFLAGS_IMMUTABLETYPE: a static type CPython built, never a
            # class statement or type() call, whose __module__ is writable
            or (isinstance(value, type) and bool(value.__flags__ & (1 << 8)))
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
