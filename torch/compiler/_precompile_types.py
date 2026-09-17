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
