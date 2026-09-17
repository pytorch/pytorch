"""
Ahead-of-time precompilation of a callable into MANY graphs: the multi-graph
counterpart of ``torch.compile(fn, fullgraph=True).aot_compile(...)``. Every
frame Dynamo produces while the caller's calls run -- the entry frame, each
``torch_dynamo_resume_in_*`` continuation created by a graph break, and every
recompiled variant of each -- is captured into one serializable artifact whose
frames are stored through CompilePackage (``torch/_dynamo/package.py``), a
low-level component that is not meant to be used directly.

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
caller-facing usage. This is distinct from
``torch._dynamo.config.caching_precompile``, which caches ``torch.compile``
artifacts transparently without an explicit capture.

Capture is by execution, and the caller drives it: the session hands back a
callable, the caller invokes it with real inputs inside their own loop, and
every frame Dynamo produces is recorded. Runtime guards stay intact during
capture, so later calls trigger the same recompilations as ordinary
``torch.compile``: the session's ``guard_filter_fn`` applies only to the
serialized copy, unlike ``torch.compile``'s option of the same name, which
removes a rejected guard from the runtime check as well
(``CheckFunctionManager.__init__`` builds the runtime guards from the filtered
list). Every dropped guard is reported in ``PrecompileSummary.dropped_guards``.

If serialization drops a configuration-dependent guard, the artifact is refused
by default rather than written with variants whose dispatch would be ambiguous
after load. Which drops those are is decided per frame, by comparing the guards
of every captured variant of one frame: a guard that held identically in every
variant is a precondition the artifact is only valid under, one that differed
is what tells its graphs apart. Guards from different frames are not
comparable -- an entry frame guards its arguments, a resume frame guards
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
  ``dynamic=True`` helps with shapes but not with pinned values.
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
  globals, but guarded dispatch is scoped to the isolated compile region owned
  by the runnable ``load`` returns. Call that runnable rather than another
  instance of the same class. Multiple loaded artifacts can share entry, inner,
  and resume code objects without taking each other's entries; the runnable's
  ``unload()`` removes only its own region and the globals it still owns.
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

from .guards import CheckFunctionManager
from .source import AttrSource, DictGetItemSource, GlobalSource, LocalSource


if TYPE_CHECKING:
    import traceback
    from collections.abc import Sequence

    from .types import GuardFilterEntry


def default_guard_filter_fn(entries: Sequence[GuardFilterEntry], /) -> Sequence[bool]:
    """
    Drop every guard ``CheckFunctionManager.serialize_guards`` would refuse,
    and keep everything else.

    Read this before trusting an artifact. The refused set is the IDENTITY
    guards -- ID_MATCH, FUNCTION_MATCH, CLOSURE_MATCH, MODULE_MATCH, NN_MODULE,
    CLASS_MATCH -- plus DICT_VERSION (dict mutation) and WEAKREF_ALIVE
    (liveness), so precompiling inherently gives up on noticing that a guarded
    object was REBOUND to a different object of the same shape, MUTATED or
    COLLECTED. Most such guards are on modules and are stable in practice, but
    one on a global holding a function is not: rebind it between capture and
    load and the artifact serves the graph traced against the old one, with no
    error. Keeping them instead makes serialization raise for essentially every
    function, so every drop is recorded with its source name in
    ``PrecompileSummary.dropped_guards`` and the only rail on by default is the
    module's risky-drop lint over them; a lint is not a proof.

    This mirrors the serializer's test branch for branch. A guard of a refused
    type is dropped, and so is a guard of another type that DERIVES one (a
    CONSTANT_MATCH on a code object runs through ID_MATCH), except that
    TYPE_MATCH and BUILTIN_MATCH are kept whatever they derive, as the
    serializer takes their branch first: BUILTIN_MATCH is an
    ``id_match_unchecked`` that records ID_MATCH, but the builtin pickles by
    name and the guard is rebuilt at load against what the name resolves to, so
    the kept guard catches a builtin swapped after load
    (``test_guard_serialization.py`` ``test_builtin_match``). The one divergence
    is the direction the filter cannot mirror soundly: a TYPE_MATCH on a
    local-scope class passes here and the serializer still refuses it, because
    the type cannot be pickled. ``orig_guard._unserializable`` would tell, but
    dropping the guard would ship an artifact that never checks the type,
    whereas keeping it makes ``serialize_guards`` refuse loudly.
    ``CheckFunctionManager.__init__`` applies a different policy inline under
    ``torch._dynamo.config.caching_precompile`` (drop ID_MATCH, CLOSURE_MATCH,
    WEAKREF_ALIVE, DICT_VERSION and anything deriving ID_MATCH or
    DICT_VERSION). The two agree on every refused type, since NN_MODULE,
    FUNCTION_MATCH, CLASS_MATCH and MODULE_MATCH all run through ID_MATCH and
    derive it; they differ on BUILTIN_MATCH, which that policy drops through
    its derived ID_MATCH and this keeps, and on one of those four rooted at a
    TypeSource, which ``id_match_unchecked`` turns into a TYPE_MATCH on a fresh
    guard so the original derives nothing: dropped here by type, kept there.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type in ("TYPE_MATCH", "BUILTIN_MATCH")
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
# here: it carries the enclosing frames' live locals, so a guard rooted there
# is judged like the local it stands for.
_DYNAMO_SYNTHESIZED = ("__nested_resume_fns",)


def _is_dynamo_synthesized(source: Source) -> bool:
    root = _source_root(source)
    return isinstance(root, LocalSource) and root.local_name in _DYNAMO_SYNTHESIZED


# Belt and braces for an install directory none of the roots name: whichever
# layout put it there, a pip target still ends in one of these.
_INSTALL_DIR_NAMES = frozenset({"site-packages", "dist-packages"})


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


@functools.cache
def _stdlib_roots() -> tuple[str, ...]:
    """
    Where this interpreter's own library lives. os is unquestionably stdlib, so
    its directory is the direct evidence and the only one that stays right when
    the stdlib is a zip; sysconfig and sys._stdlib_dir cover a build where os is
    frozen with no __file__. An install root can nest inside one of these (see
    ``_install_roots``), so a path under both is third party: an install root
    wins over a stdlib root.
    """
    roots = []
    os_file = getattr(os, "__file__", None)
    if os_file:
        roots.append(os.path.dirname(os_file))
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
    NESTED inside stdlib in a conda layout and inside platstdlib in a venv, so
    without it every pip-installed package is under a stdlib root.
    """
    paths = sysconfig.get_paths()
    roots = [paths["purelib"], paths["platlib"]]
    for name in ("getsitepackages", "getusersitepackages"):
        try:
            got = getattr(site, name)()
            found = [got] if isinstance(got, str) else list(got or ())
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
    of site-packages -- and torch.__path__ is exactly that set. It is only
    trusted if the directory this file is running from is in it, so a
    sys.modules['torch'] that is not us cannot nominate its own roots.
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
    Cached on the __file__ string rather than on the module name: the roots are
    fixed for the process, so the answer for a path never changes, while the
    module a name resolves to can.
    """
    if not os.path.isabs(file):
        # Resolving it would be against a cwd that is not the one it was
        # recorded under, so it is evidence in neither direction.
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
    # code, and a module that raises on an unknown attribute would take the
    # capture session down from inside a lint.
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
        # The inittab is keyed on the full dotted name.
        return name in sys.builtin_module_names
    if origin == "frozen" or loader is importlib.machinery.FrozenImporter:
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
    carry no evidence of its own: pyexpat.errors has none, and torch.ops is a
    ModuleType subclass whose ``__file__`` is a class attribute, so its module
    dict has none either.
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
    Whether ``global_name`` is a def of that name living in the file that read it.

    ``global_name`` is the read's ``GlobalSource.global_name``; the
    ``GuardFilterEntry.name`` spelling keeps its ``G[...]`` wrapper and is not
    it. The reading file is the OUTERMOST frame of the guard's ``user_stack``:
    a bare GlobalSource denotes the root frame's globals (an inlined frame with
    other globals reads through an ``__import_`` alias or an
    ``___unnamed_scope`` dict entry instead), while the
    stack is stamped at first use, so its innermost frame can be a helper
    inlined from another file. A def bound under its own name in the reading
    file is the one binding the inlined-source checksum of that file covers.
    ``from impl_a import op`` takes only a conditional import in the reader,
    which no checksum sees, and ``act = _impl_a if cfg.fast else _impl_b`` is a
    slot however close to home the def is; so are ``op = Ops.op`` and a def
    returned by a factory, which is why the name compared is ``__qualname__``.
    The file is read off the code object, not off ``__module__``:
    functools.wraps copies ``__module__`` along with ``__name__`` and
    ``__qualname__``, so ``op = torch.compile(op)`` behind a flag claims the
    reader's module while its code lives in eval_frame.py. The object does not
    tell that shape from an unconditional cross-file decorator, so
    ``@torch.no_grad()`` on a same-file def is not waived either. A class has
    no code object and falls back to its ``__module__``'s file; nothing else
    does, because a C-implemented wrapper such as functools.lru_cache claims
    the reader's module the same way. What this cannot see is a same-name fork
    inside the reading file -- ``try: from x import impl as op`` / ``except
    ImportError: def op`` -- which binds a different def per machine under one
    checksum; that is the conditional-bind KNOWN GAP recorded in
    ``_is_risky_drop``.
    """
    if not user_stack or getattr(value, "__qualname__", None) != global_name:
        return False
    if hasattr(value, "__code__"):
        file = getattr(value.__code__, "co_filename", None)
    elif isinstance(value, type):
        file = getattr(sys.modules.get(_owning_module(value) or ""), "__file__", None)
    else:
        return False
    return isinstance(file, str) and _norm(file) == _norm(user_stack[0].filename)


def _dynamo_alias_module(global_name: str) -> types.ModuleType | None:
    """
    The module behind an ``__import_a_dot_b`` alias, mirroring the ordinary
    branch of import_source; a torch_package module is aliased without the
    prefix and comes back None here, which fails closed.

    The OutputGraph's import_sources table is authoritative, but a guard entry
    does not carry it; unmangling collides only for a module literally named
    ``a_dot_b``.
    """
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
