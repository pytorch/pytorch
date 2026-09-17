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
