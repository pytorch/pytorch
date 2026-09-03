import collections
import dataclasses
from collections.abc import Sequence


def _count_types(pairs: Sequence[tuple[str, str]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for guard_type, _ in pairs:
        counts[guard_type] += 1
    return dict(counts)


@dataclasses.dataclass(frozen=True)
class ExampleInput:
    """One call to make during capture, when args alone are not enough."""

    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class GuardFact:
    """One guard observed while compiling a frame variant."""

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
        return f"[{'enforced' if self.enforced else 'dropped '}] {body}{where}"


@dataclasses.dataclass(frozen=True)
class FrameInvariants:
    """Guards that held, varied, or were undetermined across frame variants."""

    frame: str
    filename: str
    lineno: int
    variants: int
    invariant: tuple[GuardFact, ...]
    varying: tuple[GuardFact, ...]
    undetermined: tuple[GuardFact, ...]


@dataclasses.dataclass(frozen=True)
class PrecompileSummary:
    """Coverage and guard information from an observed precompile capture."""

    frames: int
    resume_functions: int
    guarded_codes: int
    backend_graphs: int
    bypassed: tuple[str, ...]
    truncated: tuple[str, ...] = ()
    uncovered_frames: tuple[str, ...] = ()
    wont_generalize: tuple[str, ...] = ()
    dropped_guards: tuple[tuple[str, str], ...] = ()
    kept_guards: tuple[tuple[str, str], ...] = ()
    risky_dropped_guards: tuple[tuple[str, str], ...] = ()
    # Guards that COULD have been serialized and were not, because they held
    # identically across every captured variant. Reported apart from
    # dropped_guards, which is "could not be serialized", because the reason and
    # the remedy differ -- but reported, because a capture that silently
    # discards a precondition should not look like one that had none.
    policy_dropped_guards: tuple[tuple[str, str], ...] = ()
    # (guard_type, source, rendered check) for each dropped slot that HAS a
    # rendered check. Some do not: EMPTY_NN_MODULE_HOOKS_DICT installs nothing
    # under the default skip_nnmodule_hook_guards, and the global-state guards
    # are checked in C++ against no source, so those appear in the drop lists
    # with no entry here rather than with an empty one.
    #
    # A slot is identified by its type and its SOURCE (for HASATTR and the
    # other sibling types, with the member spelled ``source{'member'}``), which
    # is not always enough to judge the drop: a dropped
    # ``('HASATTR', "counts['pixel']{'grad'}")`` may be the benign companion of
    # a kept TENSOR_MATCH on ``counts['pixel']``, or the only thing standing
    # between the artifact and an optional attribute going missing, and those
    # want very different reactions. The rendered check shows what was
    # compared and so tells them apart. Reported alongside the three lists
    # rather than folded into them, so the slot tuples stay the identity the
    # policy compares on.
    dropped_guard_code: tuple[tuple[str, str, str], ...] = ()
    capture_errors: tuple[str, ...] = ()
    # Variants whose serialized guards, after every drop, keep NO guard whose
    # source is rooted at a local of the frame (a call argument or a local it
    # was traced with). Only global-state and unmodelled leaves survived, so
    # the variant is served to any call that reaches its frame.
    #
    # Detects exactly that and no more: a variant counts as input-guarded if
    # ANY kept, modelled guard is rooted at a local, whatever it checks. A
    # frame whose only local-rooted guard is a TYPE_MATCH on the model, or an
    # ID_MATCH on a callable argument, does NOT count here even though no
    # tensor shape or value of its inputs is checked. Not a gate, because a
    # frame with no tensor arguments legitimately looks like this; reported so
    # a capture that lost its input checks is not mistaken for one that had
    # none.
    variants_without_input_guards: int = 0
    # (leaf class, rendered check) for every guard that rebuilt from its own
    # pickle into a check the live capture never made; see
    # PrecompileSession._report_guard_drift. Each will miss at serve time.
    # Describes the render this summary belongs to, like policy_dropped_guards:
    # an accumulating capture re-renders after every call and this is the
    # drift in the artifact being written now, not the union over all of them.
    drifted_guards: tuple[tuple[str, str], ...] = ()
    # (backend id, reason) for each compiled subgraph that could not be
    # composed to readable source and so ships only in the pickled bundle. The
    # reason is the exception class and message from the compose attempt. A
    # subgraph is meant to render -- it is Inductor output, which has a source
    # form -- so an entry here is a readable artifact that silently became an
    # opaque one; empty for backend="eager", which has nothing to render.
    # Populated by rendering, so it is filled in on the summary written into
    # an artifact header and empty on a summary() taken before any render.
    unrendered_backends: tuple[tuple[str, str], ...] = ()
    # Variants Dynamo compiled after inductor's CPU codegen configuration
    # (the ISA and vector width its kernels are built for) moved away from
    # the target the capture recorded on its first graph, and so were left
    # out of the artifact: one artifact carries one codegen target, and a
    # kernel built for another would fail or miscompute on a host matching the
    # recorded one. Each was a call the capture ran and did not keep, so a
    # nonzero count is coverage the summary otherwise does not show as missing.
    variants_dropped_for_codegen_target: int = 0

    @property
    def complete(self) -> bool:
        """Whether the capture covers everything it exercised.

        False if any frame produced NO guarded code at all, if any frame hit the
        recompile limit, if any was bypassed, or if a capture call raised.

        ``backend_graphs`` is checked too, because ``guarded_codes`` alone cannot
        tell a real capture from an empty one: ``allow_empty_graphs`` lets a frame
        that compiled nothing still count as one guarded code, so a model whose
        every graph sits behind a recursive ``torch._dynamo.disable`` reported
        complete while carrying no compiled compute at all.
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
        return _count_types(self.dropped_guards)

    def kept_guard_types(self) -> dict[str, int]:
        """Count serialized guards by guard type."""
        return _count_types(self.kept_guards)

    def __str__(self) -> str:
        base = (
            f"{self.frames} frames ({self.resume_functions} from graph breaks), "
            f"{self.guarded_codes} guarded codes, "
            f"{self.backend_graphs} backend graphs"
        )
        if self.dropped_guards:
            base += f", dropped guards {self.dropped_guard_types()}"
        if self.risky_dropped_guards:
            base += f", RISKY drops {[n for _, n in self.risky_dropped_guards]}"
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
        if self.variants_without_input_guards:
            base += f", {self.variants_without_input_guards} variant(s) with NO input guards"
        if self.drifted_guards:
            base += f", {len(self.drifted_guards)} DRIFTED guard(s)"
        if self.unrendered_backends:
            base += f", {len(self.unrendered_backends)} UNRENDERED subgraph(s)"
        if self.variants_dropped_for_codegen_target:
            base += (
                f", {self.variants_dropped_for_codegen_target} variant(s) DROPPED "
                f"for a changed CPU codegen target"
            )
        return base


# Exported from torch.compiler, so pickling and Sphinx anchor them there.
ExampleInput.__module__ = "torch.compiler"
GuardFact.__module__ = "torch.compiler"
FrameInvariants.__module__ = "torch.compiler"
PrecompileSummary.__module__ = "torch.compiler"
