import dataclasses
from typing import Any


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
    package: Any | None = None


ExampleInput.__module__ = "torch.compiler"
