from __future__ import annotations

import contextlib
import functools
from typing import Any, TYPE_CHECKING

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from .base import VariableTracker


class _GeneratorReconstructionMutationMode(TorchDispatchMode):
    """Reject writes to input tensors before fake operator execution."""

    def __init__(
        self,
        tx: InstructionTranslatorBase,
        tracked_tensors: list[tuple[torch.Tensor, VariableTracker]],
    ) -> None:
        super().__init__()
        self.tx = tx
        self.tracked_tensors = tracked_tensors

    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        schema = func._schema
        schema_info = torch._C._SchemaInfo(schema)
        argument_values: list[Any] = []
        for idx, argument in enumerate(schema.arguments):
            if idx < len(args):
                value = args[idx]
            elif argument.name in kwargs:
                value = kwargs[argument.name]
            elif argument.has_default_value():
                value = argument.default_value
            else:
                value = None
            argument_values.append(value)
            if isinstance(argument.type, torch.BoolType) and isinstance(value, bool):
                schema_info.add_argument_value(argument.name, value)

        for idx, value in enumerate(argument_values):
            schema_arg = torch._C._SchemaArgument(
                torch._C._SchemaArgType.input,
                idx,
            )
            if not schema_info.is_mutable(schema_arg):
                continue
            for mutated_tensor in pytree.tree_leaves(value):
                if not isinstance(mutated_tensor, torch.Tensor):
                    continue
                for tracked_tensor, variable in self.tracked_tensors:
                    try:
                        overlaps = (
                            mutated_tensor is tracked_tensor
                            or torch._C._overlaps(mutated_tensor, tracked_tensor)
                        )
                    except RuntimeError:
                        overlaps = mutated_tensor is tracked_tensor
                    if overlaps:
                        self.tx.output.side_effects.check_allowed_side_effect(variable)

        return func(*args, **kwargs)


@contextlib.contextmanager
def detect_generator_reconstruction_tensor_mutations(
    tx: InstructionTranslatorBase,
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> Generator[None, None, None]:
    """Observe the concrete operators run while reconstructing a generator."""

    if not tx.output.side_effects.is_reconstructing_generator():
        yield
        return

    from .base import VariableTracker

    tracked_tensors: list[tuple[torch.Tensor, VariableTracker]] = []

    def collect(variable: VariableTracker) -> None:
        if not variable.is_tensor():
            return
        proxy = variable.as_proxy()
        if not isinstance(proxy, torch.fx.Proxy):
            return
        fake = proxy.node.meta.get("example_value")
        if isinstance(fake, torch.Tensor):
            tracked_tensors.append((fake, variable))

    VariableTracker.visit(
        collect,
        (args, kwargs),
        side_effects=tx.output.side_effects,
    )
    if not tracked_tensors:
        yield
        return

    with _GeneratorReconstructionMutationMode(tx, tracked_tensors):
        yield


@functools.lru_cache(None)
def torch_op_mutates_first_arg(name: str) -> bool:
    if "." in name:
        name = name.split(".", 1)[0]
    if not name.startswith("aten::"):
        name = f"aten::{name}"
    return any(
        schema.arguments
        and isinstance(schema.arguments[0].type, torch.TensorType)
        and schema.arguments[0].alias_info
        and schema.arguments[0].alias_info.is_write
        for schema in torch._C._jit_get_schemas_for_operator(name)
    )
