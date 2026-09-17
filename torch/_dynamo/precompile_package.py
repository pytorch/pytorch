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
caller-facing usage. Until that rewrite, the file documents the single-graph
callable API those entry points replace, ``torch.compiler.precompile(fn,
*example_inputs)`` and its ``load(python_code, cache)``. This is distinct from
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
import os
import site
import sys
import sysconfig
import types
from typing import TYPE_CHECKING

from torch._guards import ChainedSource, Source

from .guards import CheckFunctionManager
from .source import DictGetItemSource, GlobalSource, LocalSource


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
    (liveness); the module docstring says what dropping them gives up. Most
    such guards are on modules and are stable in practice, but one on a global
    holding a function is not: rebind it between capture and load and the
    artifact serves the graph traced against the old one, with no error.
    Keeping them instead makes serialization raise for essentially every
    function, so every drop is recorded with its source name in
    ``PrecompileSummary.dropped_guards``, and the only rail on by default is
    the module's risky-drop lint over them; a lint is not a proof.

    This mirrors the three branches of the serializer's pre-check, the if/elif
    chain at the top of ``serialize_guards``, and nothing past it. A guard of a
    refused type is dropped, and so is a guard of another type that DERIVES one
    (a CONSTANT_MATCH on a code object runs through ID_MATCH), except that
    TYPE_MATCH and BUILTIN_MATCH are kept whatever they derive, as the chain
    takes their branch first: BUILTIN_MATCH is an ``id_match_unchecked`` that
    records ID_MATCH, but the builtin pickles by name and the builtins dict
    travels as a reference resolved in the loading process
    (``GuardsStatePickler._globals_snapshot``), so the guard rebuilt at load
    catches a builtin swapped afterwards (``test_guard_serialization.py``
    ``test_builtin_match``). Past the chain the filter mirrors nothing, and the
    refusal that matters there is of local-scope types, which cannot be
    pickled. It has more than one path: the chain's TYPE_MATCH/BUILTIN_MATCH
    branch raises when ``guard._unserializable`` is set, FAKE_SCRIPT_TYPE_MATCH
    sets the same flag but takes no branch of the chain, and
    ``GuardsStatePickler.reducer_override`` refuses any non-tuple object of
    such a type wherever it sits in the guard tree, so a kept guard whose source
    walks through an instance of one fails there. Passing this filter therefore
    does not mean the artifact serializes. The filter keeps all of these on
    purpose: ``orig_guard._unserializable`` would tell for the first two, but
    dropping a guard on a type the artifact cannot pickle ships an artifact that
    never checks the type, whereas keeping it makes serialization refuse loudly.
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
    no code object, and its ``__module__`` is no better: namedtuple,
    make_dataclass and ``type()`` all stamp it from the calling frame under a
    BARE ``__qualname__`` (only a nested def gets ``factory.<locals>.``), so
    ``Point = lib.make_point()`` in the reader looks exactly like a class
    statement. Its methods can tell: a class written here compiled its defs
    here, so a class is waived when at least one function in its own
    ``__dict__`` was compiled in the reading file, and ``class Marker: pass``
    fails closed. So does a class statement whose only functions are generated
    -- a fields-only ``@dataclass``, a ``NamedTuple``, an ``Enum`` -- because
    those methods compile in ``<string>`` or the stdlib, so a plain config
    dataclass read as a global is reported. Nothing else without a code object
    is waived, because a
    C-implemented wrapper such as functools.lru_cache claims the reader's
    module the same way. What this cannot see is a same-name fork inside the
    reading file -- ``try: from x import impl as op`` / ``except ImportError:
    def op`` -- which binds a different def per machine under one checksum,
    and its class analogue, a factory fed same-file methods
    (``make_dataclass(name, cfg.fields, namespace={"area": _area})``,
    ``type(name, bases, {...})``) whose fields or bases come from config; that
    is the conditional-bind KNOWN GAP recorded in ``_is_risky_drop``.
    """
    if not user_stack or getattr(value, "__qualname__", None) != global_name:
        return False
    here = _norm(user_stack[0].filename)
    if isinstance(value, type):
        attrs = vars(value).values()
        members = [getattr(m, "__func__", getattr(m, "fget", m)) for m in attrs]
        codes = [getattr(m, "__code__", None) for m in members]
        files = [c.co_filename for c in codes if isinstance(c, types.CodeType)]
        return any(_norm(f) == here for f in files)
    file = getattr(getattr(value, "__code__", None), "co_filename", None)
    return isinstance(file, str) and _norm(file) == here


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


# Dynamo's own handle on the builtins dict, minted by
# OutputGraph.install_builtins_dict_in_fglobals.
_BUILTINS_DICT_PREFIX = "__builtins_dict__"


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
