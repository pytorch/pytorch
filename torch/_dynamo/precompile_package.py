"""
Ahead-of-time precompilation of a callable into MANY graphs: the multi-graph
counterpart of ``torch.compile(fn, fullgraph=True).aot_compile(...)``. Every
frame Dynamo produces while the caller's calls run -- the entry
frame, each ``torch_dynamo_resume_in_*`` continuation created by a graph break,
and every recompiled variant of each -- is captured into one serializable
artifact on top of CompilePackage.

Everything here is internal; the capture session and the
``torch.compiler.precompile.capture`` / ``accumulate`` / ``load`` entry points
that build on it follow in later commits. This is distinct from
``torch._dynamo.config.caching_precompile``, which caches ``torch.compile``
artifacts transparently without an explicit capture.

Capture is by execution, and the caller drives it: the session hands back a
callable, the caller invokes it with real inputs inside their own loop, and
every frame Dynamo produces is recorded. Runtime guards stay intact during
capture, so later calls trigger the same recompilations as ordinary
``torch.compile``; ``guard_filter_fn`` applies only to the serialized copy, and
every dropped guard is reported in ``PrecompileSummary.dropped_guards``.

    with torch.compiler.precompile.capture(
        step, artifact_path="m.py", cache_path="m.cache", backend="inductor"
    ) as cap:
        y1 = cap(model, x1)  # runs step(model, x1), returns its result
        y2 = cap(model, x2)  # exercises another variant

    # later, in a fresh process
    compiled = torch.compiler.precompile.load("m.py", "m.cache")
    with compiled, torch.no_grad():
        compiled(model, x1)

``precompile.accumulate`` is the same model, rewriting the artifact on every
call instead of once at block exit.

If serialization drops a configuration-dependent guard, the artifact is refused
by default rather than written with variants whose dispatch would be ambiguous
after load. ``invariants`` writes a readable report
that separates, per frame, the guards holding in EVERY variant from the ones
that differed: the first are preconditions the artifact is only valid under,
the second are what tell its graphs apart. Guards from different frames are not
comparable -- an entry frame guards its arguments, a resume frame guards
whatever crossed the break -- so the intersection is per frame.

Because capture is by execution, a resume function only exists once the frame
ahead of it has actually run, so every variant must be exercised. Whatever you
do not run is not in the artifact, and ``summary().complete`` means complete
only for the observed capture, not for every possible input to the callable. A
captured call that raises marks the session incomplete even if caller code
catches it.

Know these before relying on an artifact in production:

* Calls run with the grad mode the caller sets; capture forces neither
  ``no_grad()`` nor ``enable_grad()``. For an inference artifact run the calls
  under ``torch.no_grad()``. For a training artifact pass ``training=True``,
  which lowers the backward eagerly so the artifact carries one and a served
  output can be backpropagated -- without it, AOTAutograd defers the backward
  to the first ``.backward()`` call, so a grad-enabled capture that never makes
  one records no backends and cannot be written. No loss is needed: the joint
  trace synthesizes tangents from the forward outputs' own metadata.
* A non-tensor argument, and any value that crosses a graph break, is guarded
  by equality, so an int/bool/str argument or a break coming from ``.item()``
  yields an artifact that only serves calls reproducing those exact values.
  ``summary().wont_generalize`` lists them; exercise every value you need to
  serve with a ``cap(...)`` call, or expect poor coverage on new data.
  ``dynamic=True`` helps with shapes but not with pinned values.
* Identity guards cannot be serialized, so precompiling gives up on noticing
  that a guarded object was rebound. ``summary().dropped_guards`` is the
  authoritative list. ``summary().risky_dropped_guards`` includes every drop
  observed to distinguish captured variants plus a lint for configuration-like
  sources; it is still not a proof for unobserved deployments. See
  ``_is_risky_drop``. The public ``torch.compiler.precompile`` facade rejects
  the RISKY subset by default. Refusing every drop is opt-in: every model drops
  the identity guards precompile cannot serialize, so
  ``require_no_dropped_guards=True`` refuses essentially every real artifact.
  Some models trip the lint on library internals. Measured on stock models
  when this was written, torchvision resnet18 and
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

This wraps CompilePackage, which is the low-level component and is not meant to
be used directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .guards import CheckFunctionManager


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .types import GuardFilterEntry


def _compose_with_default(
    user: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]],
) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
    """AND a caller's filter with the default rather than replacing it.

    ``default_guard_filter_fn`` is not a default in the "sensible starting point"
    sense -- it is what drops the identity guards that CANNOT be serialized at
    all. Replacing it means a caller who wanted to drop three of their own guards
    silently re-admits every unserializable one, and the failure surfaces as
    "ID_MATCH guard cannot be serialized" in frames that have nothing to do with
    their filter. A custom filter can only ever want to drop MORE, so composing
    is the only reading that makes sense.
    """

    def composed(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
        chosen = user(entries)
        if len(chosen) != len(entries):
            raise ValueError(
                f"guard_filter_fn returned {len(chosen)} decisions for "
                f"{len(entries)} guards; it must return one per entry."
            )
        base = default_guard_filter_fn(entries)
        return [bool(a) and bool(b) for a, b in zip(base, chosen)]

    return composed


def default_guard_filter_fn(
    guard_entries: Sequence[GuardFilterEntry],
) -> Sequence[bool]:
    """
    Drop every guard whose type, or whose derived type, is one the serializer
    refuses, and keep everything else.

    Read this before trusting an artifact. The refused set is the IDENTITY
    guards -- ID_MATCH, FUNCTION_MATCH, CLOSURE_MATCH, MODULE_MATCH, NN_MODULE,
    CLASS_MATCH, DICT_VERSION, WEAKREF_ALIVE -- so precompiling inherently gives
    up on noticing that a guarded object was REBOUND to a different object of
    the same shape. Most such guards are on modules and builtins and are stable
    in practice, but one on a global holding a function is not: rebind it
    between capture and load and the artifact serves the graph traced against
    the old one, with no error.

    This is not exactly ``CheckFunctionManager.serialize_guards``'s test. It is
    stricter in one direction: the serializer accepts a TYPE_MATCH or
    BUILTIN_MATCH whatever its derived types, so the BUILTIN_MATCH whose
    derived ID_MATCH this drops would have serialized. It is looser in the
    other: a TYPE_MATCH on a local-scope class passes here and the serializer
    still refuses it, because the type cannot be pickled.

    Keeping the identity guards makes serialization raise for essentially every
    function, so every drop is recorded with its source name in
    ``PrecompileSummary.dropped_guards``. A caller is not refused for having
    them, because requiring none would refuse essentially every model. The rail
    that is on is the risky-drop lint, and a lint is not a proof. See
    ``_is_risky_drop``.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]
