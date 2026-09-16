import collections
import dataclasses


@dataclasses.dataclass(frozen=True)
class GuardFact:
    """One guard observed while compiling a frame variant.

    Attributes:
        guard_type: The Dynamo guard type, e.g. ``"TENSOR_MATCH"``.
        source: The guarded source expression, e.g. ``"L['x']"``; empty for a
            guard checked against no source.
        code: The rendered check parts; empty when the guard renders none.
        value: The rendered value the check compares against; empty when it has none.
        enforced: Whether the artifact still checks this guard (it was serialized).
    """

    guard_type: str
    source: str
    code: tuple[str, ...]
    value: str
    enforced: bool

    def render(self) -> str:
        """Render the guard as one stable, human-readable line."""
        body = " ; ".join(self.code) if self.code else f"<{self.guard_type}>"
        if self.value:
            body = f"{body} {self.value}"
        where = f" on {self.source}" if self.source else ""
        label = "enforced" if self.enforced else "dropped"
        return f"[{label:<8}] {body}{where}"


@dataclasses.dataclass(frozen=True)
class FrameInvariants:
    """Guards that held, varied, or were undetermined across one frame's variants.

    Guards from different frames are not comparable (an entry frame guards its
    arguments, a resume frame whatever crossed the break), so the report is per frame.

    Attributes:
        frame: The frame's code name.
        filename: The file its code lives in.
        lineno: Its first line.
        variants: How many guarded variants of the frame were captured.
        invariant: Guards that held identically in every variant: preconditions
            the artifact is only valid under.
        varying: Guards that differed between variants: what tells its graphs apart.
        undetermined: Guards a single variant could not classify either way.
    """

    frame: str
    filename: str
    lineno: int
    variants: int
    invariant: tuple[GuardFact, ...]
    varying: tuple[GuardFact, ...]
    undetermined: tuple[GuardFact, ...]


@dataclasses.dataclass(frozen=True)
class PrecompileSummary:
    """Coverage and guard information from an observed precompile capture.

    ``str(summary)`` renders a one-line digest. Everything here describes the
    calls that ran, not every possible input or unexecuted branch.

    Attributes:
        frames: Captured frames.
        resume_functions: Of those, the graph-break continuations.
        guarded_codes: Guarded code objects across all frames.
        backend_graphs: Compiled backend graphs.
        bypassed: Names of frames Dynamo bypassed (they fell back to eager).
        truncated: Names of frames that hit the recompile limit.
        uncovered_frames: Names of frames that ended with no guarded code, so the
            artifact cannot serve them.
        wont_generalize: Guard *sources* (not frame names) that every captured
            variant pins to one value, so no variant will serve another value.
        dropped_guards: ``(guard_type, source)`` slots that could not be serialized.
        kept_guards: ``(guard_type, source)`` slots the artifact still checks.
        risky_dropped_guards: The subset of ``dropped_guards`` observed to tell
            captured variants apart, or flagged by the configuration-source lint.
        policy_dropped_guards: Serializable slots dropped because they held
            identically across every variant.
        dropped_guard_code: ``(guard_type, source, rendered_check)`` for each
            dropped slot that renders a check; the check disambiguates a slot the
            type and source alone cannot.
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
    # The guard lists hold (guard_type, source) slots. dropped_guards is every
    # slot the artifact omitted because it could not be serialized;
    # risky_dropped_guards is the subset of it that varied between captured
    # variants or that the configuration-slot lint flags (see
    # torch._dynamo.precompile_package._is_risky_drop). policy_dropped_guards
    # is disjoint from both: slots that COULD have been serialized and were
    # dropped because they held identically across every captured variant. The
    # reason and the remedy differ, so it is reported apart -- but reported,
    # because a capture that silently discards a precondition should not look
    # like one that had none. kept_guards is what the artifact still checks.
    dropped_guards: tuple[tuple[str, str], ...] = ()
    kept_guards: tuple[tuple[str, str], ...] = ()
    risky_dropped_guards: tuple[tuple[str, str], ...] = ()
    policy_dropped_guards: tuple[tuple[str, str], ...] = ()
    # (guard_type, source, rendered check) for each slot of dropped_guards or
    # policy_dropped_guards that HAS a rendered check. Some do not:
    # EMPTY_NN_MODULE_HOOKS_DICT installs nothing under the default
    # skip_nnmodule_hook_guards, and the global-state guards are checked in C++
    # against no source, so those appear in the drop lists with no entry here
    # rather than with an empty one.
    #
    # A slot is identified by its type and its SOURCE, which
    # for some types is not enough to judge the drop: a dropped
    # ``('HASATTR', "counts['pixel']")`` may be the benign companion of a kept
    # TENSOR_MATCH on the same source, or the only thing standing between the
    # artifact and an optional attribute going missing, and those want very
    # different reactions. The rendered check names the attribute and so tells
    # them apart. Reported alongside the slot lists rather than folded into
    # them, so the slot tuples stay the identity the policy compares on.
    dropped_guard_code: tuple[tuple[str, str, str], ...] = ()
    capture_errors: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether the capture covers everything it exercised.

        False if a frame was ``bypassed``, hit the recompile limit
        (``truncated``) or was never reached (``uncovered_frames``), if a
        capture call raised (``capture_errors``), or if nothing was compiled.
        Both ``guarded_codes`` and ``backend_graphs`` must be non-zero for the
        last check, because ``allow_empty_graphs`` lets a frame that compiled
        nothing still count as one guarded code, so ``guarded_codes`` alone
        cannot tell a real capture from an empty one.
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
            base += f", dropped guards {self.dropped_guard_types()}"
        if self.risky_dropped_guards:
            base += f", RISKY drops {[src for _, src in self.risky_dropped_guards]}"
        if self.uncovered_frames:
            base += (
                f", {len(self.uncovered_frames)} UNCOVERED: "
                f"{list(self.uncovered_frames)}"
            )
        if self.wont_generalize:
            base += f", {len(self.wont_generalize)} value-pinned guards"
        if self.truncated:
            base += f", >={len(self.truncated)} TRUNCATED: {list(self.truncated)}"
        if self.bypassed:
            base += f", {len(self.bypassed)} BYPASSED: {list(self.bypassed)}"
        if self.capture_errors:
            base += f", {len(self.capture_errors)} CAPTURE ERROR(S)"
        return base
