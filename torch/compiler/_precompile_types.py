import collections
import dataclasses
from collections.abc import Sequence
from typing import Any


def _count_types(pairs: Sequence[tuple[str, str]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for guard_type, _ in pairs:
        counts[guard_type] += 1
    return dict(counts)


@dataclasses.dataclass(frozen=True)
class ExampleInput:
    r"""ExampleInput(args=(), kwargs={})

    Describes one call used by ``torch.compiler.precompile`` with the Dynamo tracer.

    Args:
        args (tuple, optional): Positional arguments for the example call.
            Default: ``()``.
        kwargs (dict, optional): Keyword arguments for the example call.
            Default: ``{}``.

    Examples::

        >>> example = torch.compiler.ExampleInput(
        ...     args=(torch.randn(4),), kwargs={"scale": 2.0}
        ... )
        >>> code, cache = torch.compiler.precompile(
        ...     lambda x, *, scale: x * scale,
        ...     example_inputs=[example],
        ...     tracer="dynamo",
        ... )
    """

    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class GuardFact:
    """One guard observed while compiling a Dynamo frame variant."""

    guard_type: str
    source: str
    code: tuple[str, ...]
    value: str
    enforced: bool

    def render(self) -> str:
        """Render this guard as one stable, human-readable line."""
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
    variant_examples: tuple[int, ...]
    invariant: tuple[GuardFact, ...]
    varying: tuple[GuardFact, ...]
    undetermined: tuple[GuardFact, ...]


@dataclasses.dataclass(frozen=True)
class PrecompileSummary:
    """Coverage and guard information from a Dynamo precompile capture."""

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
    policy_dropped_guards: tuple[tuple[str, str], ...] = ()
    capture_errors: tuple[str, ...] = ()
    variant_examples: tuple[tuple[int, ...], ...] = ()

    @property
    def complete(self) -> bool:
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
        result = (
            f"{self.frames} frames ({self.resume_functions} from graph breaks), "
            f"{self.guarded_codes} guarded codes, "
            f"{self.backend_graphs} backend graphs"
        )
        if self.dropped_guards:
            result += f", dropped guards {self.dropped_guard_types()}"
        if self.risky_dropped_guards:
            result += f", RISKY drops {[n for _, n in self.risky_dropped_guards]}"
        if self.uncovered_frames:
            result += (
                f", {len(self.uncovered_frames)} UNCOVERED: "
                f"{list(self.uncovered_frames)}"
            )
        if self.wont_generalize:
            result += f", {len(self.wont_generalize)} value-pinned guards"
        if self.truncated:
            result += f", {len(self.truncated)} TRUNCATED: {list(self.truncated)}"
        if self.bypassed:
            result += f", {len(self.bypassed)} BYPASSED: {list(self.bypassed)}"
        if self.capture_errors:
            result += f", {len(self.capture_errors)} CAPTURE ERROR(S)"
        return result


@dataclasses.dataclass(frozen=True)
class _DynamoGuardedVariant:
    guards_state: bytes
    dynamo_code: Any


@dataclasses.dataclass(frozen=True)
class _DynamoCodeState:
    code: Any
    python_module: str
    function_names: tuple[str, ...]
    install_to_global: bool
    code_source: str | None
    global_bindings: dict[str, tuple[str, tuple[str, ...]]]
    value_globals: dict[str, object]
    import_sources: dict[str, str]
    defaults: tuple[object, ...] | None
    kwdefaults: dict[str, object] | None
    variants: tuple[_DynamoGuardedVariant, ...]


@dataclasses.dataclass(frozen=True)
class _DynamoDisabledFunction:
    code: Any
    name: str
    defaults: tuple[object, ...] | None
    kwdefaults: dict[str, object] | None
    module_globals: dict[str, str]
    value_globals: dict[str, object]


@dataclasses.dataclass(frozen=True)
class _DynamoInputContractVariant:
    spec: str
    leaves: tuple[dict[str, object] | None, ...]


@dataclasses.dataclass(frozen=True)
class _DynamoInputContract:
    variants: tuple[_DynamoInputContractVariant, ...]


@dataclasses.dataclass(frozen=True)
class _DynamoArtifactState:
    codes: tuple[_DynamoCodeState, ...]
    disabled_functions: dict[str, _DynamoDisabledFunction]
    input_contract: _DynamoInputContract | None
    serving_mode: str
    entry_module: str
    entry_qualname: str
    entry_name: str
    entry_firstlineno: int
    device_type: str = "cpu"
    system_info: Any | None = None
    mutates_input_grads: bool = False
    recompile_limit: int = 256
    dynamic: bool | None = None
    summary: PrecompileSummary | None = None
    package: Any | None = None


ExampleInput.__module__ = "torch.compiler"
GuardFact.__module__ = "torch.compiler"
FrameInvariants.__module__ = "torch.compiler"
PrecompileSummary.__module__ = "torch.compiler"
