"""Plain-data types the multi-graph precompile capture reports through.

A leaf module on purpose. ``import torch`` loads ``torch.compiler``, and through
it ``torch._precompile`` (and, once ``torch.compiler.precompile`` is a module
later in this stack, this module too), without loading ``torch._dynamo``, so a
type that public surface exports cannot live in the Dynamo-side internals,
``torch/_dynamo/precompile_package.py`` (which imports this module at load time
later in this stack), without an import cycle. Import-wise the types could live
in ``torch/_precompile.py``; keeping them out of it is layering: the Dynamo
internals must not depend on the make_fx capture module, which the follow-up
capture session makes an importer of those internals. The types are frozen
dataclasses of immutable fields, so they pickle, compare by value and hash.
"""

import collections
import dataclasses


@dataclasses.dataclass(frozen=True, kw_only=True)
class GuardFact:
    """One guard observed while compiling a frame variant.

    Attributes:
        guard_type: The Dynamo guard type, e.g. ``"TENSOR_MATCH"``.
        source: The guarded source, spelled as ``GuardFilterEntry.name``, i.e.
            the ``Guard.name`` with local scope stripped (``L['x']`` -> ``x``),
            the same spelling as the ``(guard_type, source)`` slots of
            ``PrecompileSummary``: ``"x"``, ``"self.eps"``, ``"G['CFG'].width"``.
            Empty for a guard checked against no source.
        code: The rendered check parts, with the addresses Dynamo interpolates
            scrubbed by the producer, e.g.
            ``("___check_type_id(L['x'], <id>), type=<class 'int'>",)``; empty
            when the guard renders none.
        value: A rendered fragment for what the check compares that its code does
            not show: a tensor's dtype and shape line, or ``"is <callable>"`` for
            an identity guard. Empty when the code says it all.
        enforced: Whether the artifact still checks this guard (it was serialized).
    """

    guard_type: str
    source: str
    code: tuple[str, ...]
    value: str
    enforced: bool


@dataclasses.dataclass(frozen=True, kw_only=True)
class PrecompileSummary:
    """Coverage and guard information from an observed precompile capture.

    ``str(summary)`` renders a one-line digest. Everything here describes the
    calls that ran, not every possible input or unexecuted branch.

    The guard fields hold ``(guard_type, source)`` slots, the source spelled as
    ``GuardFilterEntry.name``, i.e. the ``Guard.name`` with local scope stripped
    (``L['self'].act`` -> ``self.act``; ``G['CFG'].width`` unchanged).
    ``dropped_guards`` is every
    slot the serialized copy's guard filter rejected: the guards the serializer
    refuses (the identity guards, plus ``DICT_VERSION`` and ``WEAKREF_ALIVE``),
    plus whatever a caller-supplied filter dropped.
    ``risky_dropped_guards`` is the subset of it that was observed to tell
    captured variants apart or that the risky-drop lint flags as a
    configuration-chosen binding. ``policy_dropped_guards`` is disjoint from
    both: slots that could have been serialized and were dropped because they
    held identically across every captured variant. Construction raises
    ``ValueError`` when either relation fails: the digest and
    ``dropped_guard_types()`` read ``dropped_guards`` alone, so a risky slot
    listed nowhere else would be reported as no drop at all. The reason and the
    remedy differ, so they are reported apart, but reported, because a capture
    that discards a precondition should not look like one that had none.
    ``kept_guards`` is what the artifact still checks.

    The frame lists (``bypassed``, ``truncated``, ``uncovered_frames``) hold
    bare ``co_name``s, which are not unique: every ``nn.Module`` has a
    ``forward``, so ``['forward', 'forward']`` is two frames, not a repeat, and
    the lists can neither identify a frame nor be checked disjoint.

    Attributes:
        frames: Captured frames.
        resume_functions: Of those, the graph-break continuations.
        guarded_codes: Guarded code objects across all frames.
        backend_graphs: Compiled backend graphs.
        bypassed: ``co_name``s of frames the package refused to record: every compile
            of the frame was bypassed because its guards could not be
            serialized, or a backend artifact was missing when the package was
            saved. Not an eager fallback: the frame ran compiled during capture
            (Dynamo installed its guarded code, it just went unrecorded), the
            artifact carries no variant of it, and an install re-traces it
            rather than skipping it as trivial.
        truncated: ``co_name``s of frames that hit the recompile limit. A lower bound,
            which is why the digest prints it as ``>=``: Dynamo runs a frame
            that hit the limit, and every frame called beneath it, without
            tracing (its ``FrameExecStrategy`` is ``RUN_ONLY`` for the frame and
            its callees alike), so a limit hit below the first one is never seen.
        uncovered_frames: ``co_name``s of frames that ended with no guarded code and
            were not bypassed, so the artifact cannot serve them: a thin wrapper
            whose graphs all landed in an inner frame, or a frame Dynamo gave up
            on. The remainder after ``bypassed``, which also holds no guarded
            code but has a different cause and remedy.
        wont_generalize: Guard *sources* (not frame names) that every captured
            variant pins to one value, so no variant will serve another value.
        dropped_guards: Slots the serialized copy's guard filter rejected: the
            guards the serializer refuses (the identity guards, plus
            ``DICT_VERSION`` and ``WEAKREF_ALIVE``), plus whatever a
            caller-supplied filter dropped.
        kept_guards: Slots the artifact still checks.
        risky_dropped_guards: The subset of ``dropped_guards`` observed to tell
            captured variants apart, or flagged by the risky-drop lint as a
            configuration-chosen binding.
        policy_dropped_guards: Serializable slots dropped because they held
            identically across every variant.
        dropped_guard_code: ``(guard_type, source, rendered_check)`` for each slot
            of ``dropped_guards`` or ``policy_dropped_guards`` that renders a
            check. Every guard type dropped as unserializable or by the
            invariance policy renders one; a slot is missing here only when a
            caller-supplied filter dropped a guard whose check lives in a C++
            leaf and renders no code (``GLOBAL_STATE``, ``TORCH_FUNCTION_STATE``).
            A slot alone can be ambiguous: a dropped ``('HASATTR', "counts['pixel']")``
            is either the benign companion of a kept ``TENSOR_MATCH`` on the same
            source or the only guard on an optional attribute, and only the
            rendered check tells them apart. Kept beside the slot lists rather
            than folded into them, so the slots stay the identity the policy
            compares on; for programmatic consumers, not the digest.
        capture_errors: Messages from capture calls that raised.
    """

    frames: int
    resume_functions: int
    guarded_codes: int
    backend_graphs: int
    bypassed: tuple[str, ...] = ()
    truncated: tuple[str, ...] = ()
    uncovered_frames: tuple[str, ...] = ()
    wont_generalize: tuple[str, ...] = ()
    dropped_guards: tuple[tuple[str, str], ...] = ()
    kept_guards: tuple[tuple[str, str], ...] = ()
    risky_dropped_guards: tuple[tuple[str, str], ...] = ()
    policy_dropped_guards: tuple[tuple[str, str], ...] = ()
    dropped_guard_code: tuple[tuple[str, str, str], ...] = ()
    capture_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dropped = set(self.dropped_guards)
        stray = set(self.risky_dropped_guards) - dropped
        if stray:
            raise ValueError(
                f"risky_dropped_guards must be a subset of dropped_guards; not "
                f"dropped: {sorted(stray)}"
            )
        both = set(self.policy_dropped_guards) & dropped
        if both:
            raise ValueError(
                f"policy_dropped_guards must be disjoint from dropped_guards; in "
                f"both: {sorted(both)}"
            )

    @property
    def complete(self) -> bool:
        """Whether the capture covers everything it exercised.

        Coverage only: False if a frame was ``bypassed``, hit the recompile
        limit (``truncated``) or ended with no guarded code
        (``uncovered_frames``), if a capture call raised (``capture_errors``),
        or if nothing was compiled. Both ``guarded_codes`` and ``backend_graphs``
        must be non-zero for the last check, because ``allow_empty_graphs`` lets
        a frame that compiled nothing still count as one guarded code, so
        ``guarded_codes`` alone cannot tell a real capture from an empty one.
        The guard fields never enter: ``risky_dropped_guards`` is a lint
        finding, not a proof, and ``wont_generalize`` follows from any value the
        capture pinned, so both describe how far a complete capture generalizes
        and the digest reports them on their own.
        """
        return (
            not self.bypassed
            and not self.truncated
            and not self.uncovered_frames
            and not self.capture_errors
            and self.guarded_codes > 0
            and self.backend_graphs > 0
        )

    def dropped_guard_types(self) -> dict[str, int]:
        """Count omitted guards by guard type."""
        return dict(collections.Counter(t for t, _ in self.dropped_guards))

    def kept_guard_types(self) -> dict[str, int]:
        """Count serialized guards by guard type."""
        return dict(collections.Counter(t for t, _ in self.kept_guards))

    def __str__(self) -> str:
        base = (
            f"{self.frames} frames ({self.resume_functions} from graph breaks), "
            f"{self.guarded_codes} guarded codes, "
            f"{self.backend_graphs} backend graphs"
        )
        if self.dropped_guards:
            kept = len(self.kept_guards)
            base += f", dropped guards {self.dropped_guard_types()} ({kept} kept)"
        if self.risky_dropped_guards:
            base += f", RISKY drops {[src for _, src in self.risky_dropped_guards]}"
        if self.policy_dropped_guards:
            base += f", {len(self.policy_dropped_guards)} policy-dropped guards"
        if self.wont_generalize:
            base += f", {len(self.wont_generalize)} value-pinned sources"
        if self.uncovered_frames:
            base += (
                f", {len(self.uncovered_frames)} UNCOVERED: "
                f"{list(self.uncovered_frames)}"
            )
        if self.truncated:
            base += f", >={len(self.truncated)} TRUNCATED: {list(self.truncated)}"
        if self.bypassed:
            base += f", {len(self.bypassed)} BYPASSED: {list(self.bypassed)}"
        if self.capture_errors:
            base += f", {len(self.capture_errors)} CAPTURE ERROR(S)"
        return base
