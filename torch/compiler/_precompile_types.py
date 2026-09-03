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
    capture_errors: tuple[str, ...] = ()

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
        return base


# GuardFact, FrameInvariants and PrecompileSummary keep their truthful
# __module__: torch.compiler does not export them, so pointing __module__ at it
# would make every instance unpicklable (pickle resolves the class through
# __module__). ExampleInput is different: torch.compiler.precompile exports it,
# so torch/_precompile.py re-homes it there with the singleton's other members
# (pickle resolves it through the singleton).
