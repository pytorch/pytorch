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

If serialization drops a guard that looks configuration-dependent, the artifact
is refused by default rather than written with variants whose dispatch would be
ambiguous after load. Two tests feed that decision, and their union is
``PrecompileSummary.risky_dropped_guards``. One is the risky-drop lint over the
dropped guard's binding site (``_is_risky_drop``); it waives a builtin read, a
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
import os
import site
import sys
import sysconfig
from typing import TYPE_CHECKING

from .guards import CheckFunctionManager


if TYPE_CHECKING:
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

    This mirrors the type tests of the three branches of the serializer's
    pre-check, the if/elif chain at the top of ``serialize_guards``, and
    nothing past it: not the raise inside the first branch, which is the
    local-scope refusal below. A guard of a refused type is dropped, and so is
    a guard of another type that DERIVES one (a CONSTANT_MATCH on a code object
    runs through ID_MATCH), except that TYPE_MATCH and BUILTIN_MATCH are kept
    whatever they derive, as the chain takes their branch first: BUILTIN_MATCH
    is an ``id_match_unchecked`` that records ID_MATCH, but the builtin pickles
    by name, the artifact carries only save-time copies of the builtins dict
    (the guards' copy ``serialize_guards`` prunes to the names guards read, and
    the pickle-filtered copy ``get_runtime_env`` records for the bytecode), and
    both loaders bind the LIVE ``builtins.__dict__`` under the guard's key
    instead (``CompilePackage.install``,
    ``AOTCompiledFunction._seed_guard_scope``), so the guard rebuilt at load
    catches a builtin swapped afterwards (``test_aot_compile.py``
    ``test_kept_builtin_match_guard_reads_the_seeded_builtins_dict`` pins the
    live dict). DICT_KEYS_MATCH is kept by type for another reason. The derived
    types an entry carries are the UNSAVED build's: ``CheckFunctionManager``
    runs the pre-filter build with ``save_guards=False``, and a DICT_KEYS_MATCH
    on ``torch.utils._pytree.SUPPORTED_NODES`` promotes itself to DICT_VERSION
    in exactly that build, whereas the save build pins it to a keys-match
    (``guard._force_dict_keys_match``) that the pre-check accepts; dropping it
    on the unsaved build's DICT_VERSION would discard the one guard that
    notices a pytree node registered between capture and load. Check any new
    refused derived type against the save build before dropping on it. Past
    the chain the filter mirrors nothing, and the refusal that matters there
    is of local-scope types, which cannot be pickled by name. It has two paths:
    the chain's TYPE_MATCH/BUILTIN_MATCH branch raises when
    ``guard._unserializable`` is set, and ``GuardsStatePickler.reducer_override``
    refuses a plain instance of such a type wherever it sits in the guard tree,
    so a kept guard whose source walks through one fails there.
    (FAKE_SCRIPT_TYPE_MATCH sets the same flag, but nothing outside that branch
    reads it, so a local-scope script-object type is the known hole: kept here
    and not refused by the pre-check.) The pickler's refusal is not universal:
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
    ``CheckFunctionManager.__init__`` applies a different policy inline under
    ``torch._dynamo.config.caching_precompile`` (drop ID_MATCH, CLOSURE_MATCH,
    WEAKREF_ALIVE, DICT_VERSION and anything deriving ID_MATCH or
    DICT_VERSION). The two agree on every refused type, since NN_MODULE,
    FUNCTION_MATCH, CLASS_MATCH and MODULE_MATCH all derive ID_MATCH (the first
    two through ID_MATCH, the other two through ``id_match_unchecked`` directly,
    which records the name ID_MATCH); they differ on BUILTIN_MATCH, which that
    policy drops through its derived ID_MATCH and this keeps, on
    DICT_KEYS_MATCH over SUPPORTED_NODES, which that policy drops through the
    unsaved build's DICT_VERSION and this keeps, and on one of those four
    rooted at a TypeSource, which ``id_match_unchecked`` turns into a
    TYPE_MATCH on a fresh guard so the original derives nothing: dropped here
    by type, kept there.
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


def _norm(path: str) -> str:
    """
    realpath then normcase. A relative path resolves against the process cwd,
    so a recorded ``__file__`` is gated with isabs before it gets here.
    """
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
