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

    ``str(summary)`` renders a one-line digest: the counts, the dropped guards
    tallied by type, the risky slots and frame lists cut to their first five
    entries (``+N more``), and the first non-empty line of the first capture
    error.
    Everything here describes the calls that ran, not every possible input or
    unexecuted branch, and once ``truncated`` is non-empty every count and
    frame list is a lower bound: from a limit hit on, that frame and everything
    it called ran untraced, so a frame first reached there is in no list here.

    The guard fields hold ``(guard_type, source)`` slots, the source spelled
    with the local scope stripped (``L['self'].act`` -> ``self.act``;
    ``G['CFG'].width`` unchanged). Each list holds a slot once, however many
    frames or variants carried it, so ``dropped_guard_types`` counts distinct
    slots, not occurrences. The relations between the lists hold within one
    frame variant, where the producer applies them, and are stated here rather
    than checked:

    * ``kept_guards`` and ``dropped_guards`` are disjoint: the filter gives a
      slot one verdict, keep or reject.
    * ``risky_dropped_guards`` is drawn from ``dropped_guards``.
    * ``policy_dropped_guards`` is disjoint from both: a policy drop is taken
      out of the serialized copy the filter kept, once the slot held
      identically across every captured variant, and is not checked either.
    * ``dropped_guard_code`` draws its slots from ``dropped_guards`` and
      ``policy_dropped_guards``.
    * every ``wont_generalize`` source is the source half of a ``kept_guards``
      slot, the value-equality guard that pins it.

    The lists aggregate every captured frame and a slot names no frame, so two
    frames' ``self.act`` are one slot: a slot one frame kept and another dropped
    is in both ``kept_guards`` and ``dropped_guards`` (the digest's ``(N kept)``
    is the kept count beside the drops, not the other half of a total), and a
    slot one frame's filter rejected and another frame's invariance policy
    dropped is in both drop lists. Nothing is enforced, as for the frame lists
    below: a report that raised on its own bookkeeping would lose the coverage
    it exists to describe.

    The frame lists differ in what an entry is. ``bypassed`` and
    ``uncovered_frames`` hold one bare ``co_name`` per frame, read off the
    package's entries, so a name can repeat (every ``nn.Module`` has a
    ``forward``: ``['forward', 'forward']`` is two frames, not a repeat) and
    their lengths count frames. ``truncated`` holds ``co_name
    (filename:firstlineno)``, recorded once per code object that hit the limit,
    so its entries do identify a frame. A bare name identifies none, so the two
    bare lists cannot be checked disjoint.

    Attributes:
        frames: Captured frames: every frame the package holds an entry for, the
            ``bypassed``, ``truncated`` and ``uncovered_frames`` ones included,
            so the digest's frame clauses cut into this count rather than add
            to it.
        resume_functions: Of those, the graph-break continuations.
        guarded_codes: Guarded code objects across all frames.
        backend_graphs: Compiled backend graphs.
        bypassed: ``co_name`` of the frames the package holds nothing installable
            for: no compile of the frame recorded a guarded code and one was
            bypassed (its guards could not be serialized, or its graph held
            parameters by static address), or a
            backend artifact was missing when the package was saved. Not an
            eager fallback: the frame ran compiled during capture, the package
            kept no variant of it a load can serve, and an install re-traces it
            rather than skipping it as trivial.
        truncated: ``co_name (filename:firstlineno)`` of each frame that hit the
            recompile limit. A lower bound, which is why the digest prints it
            as ``>=``: from a limit hit on, that frame and the frames it calls
            run without tracing, so a limit hit that would follow it there is
            never recorded.
        uncovered_frames: ``co_name`` of the frames the capture ran that ended with
            no guarded code and were not bypassed, so the artifact cannot serve
            them: a thin wrapper whose graphs all landed in an inner frame, a
            frame Dynamo gave up on, or a frame whose compile raised (its
            message is in ``capture_errors``, so one failure shows in both
            digest clauses). A different cause and remedy from ``bypassed``
            (an install leaves an uncovered frame to run eagerly, so only a
            re-capture recovers it, where it re-traces a bypassed one), never the same
            frame; a frame that hit the recompile limit before it recorded a
            guarded code is in ``truncated`` too. Not a remainder: which frames
            count as a gap is the producer's decision, and a frame the package
            holds an entry for but never ran is not one, so this is not
            ``frames`` minus ``bypassed`` minus the frames that hold guarded
            code.
        wont_generalize: Guard *sources* (not frame names) a kept value-equality
            guard on a bare argument name pins in some variant (``self.eps`` is
            not a bare name) while no other variant of the same frame guards the
            source without pinning it, so as captured no variant served another
            value. Observed, not proven: a variant that never guarded the source
            does not count as serving other values of it.
        dropped_guards: Slots the serialized copy's guard filter rejected; a
            variant that dropped a slot does not check it, so a load through that
            variant cannot notice whatever it checked. Which guards a filter
            rejects is that filter's own contract and not repeated here; a
            caller-supplied filter decides its own set. A slot is listed under
            the guard's own type
            whatever the reason for the drop, so a ``TENSOR_MATCH`` rejected for
            what its check derives is a dropped ``TENSOR_MATCH``.
        kept_guards: Slots the serialized copy's guard filter kept and the
            invariance policy left in place.
        risky_dropped_guards: The subset of ``dropped_guards`` observed to tell
            captured variants apart, or flagged by the risky-drop lint as a
            configuration-chosen binding.
        policy_dropped_guards: Slots the filter kept that the invariance policy
            then dropped because they held identically across every captured
            variant (only guard types from a fixed set are eligible).
            Reported apart from ``dropped_guards`` because the remedy differs,
            and reported at all because a capture that discards a precondition
            should not look like one that had none.
        dropped_guard_code: ``(guard_type, source, rendered_check)``, one per slot
            of ``dropped_guards`` or ``policy_dropped_guards`` whose guard
            rendered a check (how the check is installed does not predict that,
            and a slot whose guard rendered none has no entry). One rendering
            however many
            variants dropped the slot: where the check embeds the guarded value
            (``EQUALS_MATCH`` renders ``L['n'] == 3``) it is one variant's, the
            producer's pick rather than a merge, so it tells the form of the
            check and not the value; that the slot varied at all is what
            ``risky_dropped_guards`` records. Carried because a slot alone can
            be ambiguous: a dropped ``('HASATTR', "counts['pixel']")`` is either
            the benign companion of a kept ``TENSOR_MATCH`` on the same source
            or the only guard on an optional attribute, and only the rendered
            check tells them apart. Kept beside the slot lists so the slots stay
            the identity the policy compares on; for programmatic consumers,
            not the digest.
        capture_errors: One message per distinct exception a capture call raised
            (repeats of the same type and message collapse), the exception type
            first (``"RuntimeError: boom"``), so the digest's first line is
            never empty however the exception was raised.
    """

    # The producer's rules live in torch._dynamo.precompile_package:
    # default_guard_filter_fn (dropped_guards), _pins_a_value (wont_generalize),
    # _INVARIANT_DROPPABLE_GUARD_TYPES (policy_dropped_guards); a bypassed frame is
    # one OutputGraph.bypass_package flagged; dropped_guard_code holds the
    # Guard.code_list a GuardBuilder method rendered.
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

    @property
    def complete(self) -> bool:
        """Whether the capture covers everything it exercised.

        Coverage only: False if a frame was ``bypassed``, hit the recompile
        limit (``truncated``) or ended with no guarded code
        (``uncovered_frames``), if a capture call raised (``capture_errors``),
        or if nothing was compiled. Both ``guarded_codes`` and ``backend_graphs``
        must be non-zero for the last check, because a frame that compiled
        nothing can still count as one guarded code, so ``guarded_codes`` alone
        cannot tell a real capture from an empty one.
        The guard fields never enter: ``risky_dropped_guards`` is a lint
        finding, not a proof, and ``wont_generalize`` follows from any value the
        capture pinned, so both describe how far a complete capture generalizes
        and the digest reports them on their own.
        """
        # torch._dynamo.config.allow_empty_graphs is what lets an empty frame count.
        return (
            not self.bypassed
            and not self.truncated
            and not self.uncovered_frames
            and not self.capture_errors
            and self.guarded_codes > 0
            and self.backend_graphs > 0
        )

    @property
    def dropped_guard_types(self) -> dict[str, int]:
        """The distinct dropped slots counted by guard type."""
        return dict(collections.Counter(t for t, _ in self.dropped_guards))

    @property
    def kept_guard_types(self) -> dict[str, int]:
        """The distinct kept slots counted by guard type."""
        return dict(collections.Counter(t for t, _ in self.kept_guards))

    def __str__(self) -> str:
        def count(k: int, noun: str) -> str:
            return f"{k} {noun}" if k == 1 else f"{k} {noun}s"

        def some(items: tuple[str, ...]) -> str:
            more = len(items) - 5
            return f"{list(items[:5])}" + (f" +{more} more" if more > 0 else "")

        base = (
            f"{count(self.frames, 'frame')} ({self.resume_functions} from graph breaks), "
            f"{count(self.guarded_codes, 'guarded code')}, "
            f"{count(self.backend_graphs, 'backend graph')}"
        )
        if self.dropped_guards:
            kept = len(self.kept_guards)
            base += f", dropped guards {self.dropped_guard_types} ({kept} kept)"
        if self.risky_dropped_guards:
            # Both halves of the slot: two risky slots can share a source (a
            # dropped ID_MATCH on self.act and a HASATTR checked on it).
            risky = tuple(f"{t} {src}" for t, src in self.risky_dropped_guards)
            base += f", RISKY drops {some(risky)}"
        if self.policy_dropped_guards:
            policy = len(self.policy_dropped_guards)
            base += f", {count(policy, 'policy-dropped guard')}"
        if self.wont_generalize:
            base += f", {count(len(self.wont_generalize), 'value-pinned source')}"
        if self.uncovered_frames:
            base += f", {len(self.uncovered_frames)} UNCOVERED: {some(self.uncovered_frames)}"
        if self.truncated:
            base += f", >={len(self.truncated)} TRUNCATED: {some(self.truncated)}"
        if self.bypassed:
            base += f", {len(self.bypassed)} BYPASSED: {some(self.bypassed)}"
        if self.capture_errors:
            errors = len(self.capture_errors)
            lines = self.capture_errors[0].splitlines()
            first = next((line for line in lines if line.strip()), "")
            base += f", {errors} CAPTURE ERROR{'' if errors == 1 else 'S'}: {first!r}"
            if errors > 1:
                base += f" +{errors - 1} more"
        return base
