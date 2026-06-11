#  Copyright (c) Meta Platforms, Inc. and affiliates
"""Schema-discovered registration of single-dim strategy variants.

Manual single-dim registrations target the semantic base op.
``auto_register_op_variants`` mechanically covers the related ATen overloads
(in-place, out, functional, and foreach) by reusing the base op's strategy.
"""

from collections.abc import Sequence
from typing import Any

from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    _get_overload_packet,
    _get_packet_overload,
    _normalize_schema_type,
    _op_namespace_and_base_name,
    _resolve_foreach_elementwise_overload,
    _schema_args_match,
    _ShardingPlaceholder,
    _SingleDimStrategyFunc,
    _SingleDimStrategyInfo,
)
from torch.distributed.tensor.placement_types import Placement


def _tensor_schema_count(schema_values: Sequence[Any]) -> int:
    """Count Tensor-valued entries in an ATen schema sequence.

    Variant discovery uses this to reject overloads with incompatible arity.
    """
    return sum(1 for value in schema_values if "Tensor" in str(value.type))


def _schema_tensor_output_count(op: OpOverload) -> int:
    """Count Tensor-valued returns for an ATen overload."""
    return _tensor_schema_count(op._schema.returns)


def _is_write_arg(arg: Any) -> bool:
    """Return whether a schema argument is mutated by the op.

    Schema alias info is the reliable way to identify explicit out args and
    mutable inputs whose placements must be preserved.
    """
    return arg.alias_info is not None and arg.alias_info.is_write


def _is_foreach_like_op_name(op_name: str) -> bool:
    """Return whether the op name uses foreach/fused naming conventions.

    These ops are already aggregate/list variants, so they should not be used
    as scalar bases for another round of automatic variant registration.
    """
    return op_name.startswith(
        ("aten::_foreach_", "aten::_amp_foreach_", "aten::_fused_")
    )


def _schema_args_are_same(
    base_args: Sequence[Any], candidate_args: Sequence[Any]
) -> bool:
    """Return whether two schema arg lists are identical for strategy reuse.

    Out variants require this stricter check so only the written output args are
    allowed to differ from the base op.
    """
    if len(base_args) != len(candidate_args):
        return False
    return all(
        base_arg.name == candidate_arg.name
        and base_arg.kwarg_only == candidate_arg.kwarg_only
        and _normalize_schema_type(base_arg) == _normalize_schema_type(candidate_arg)
        for base_arg, candidate_arg in zip(base_args, candidate_args)
    )


def _is_explicit_out_arg(arg: Any) -> bool:
    """Return whether a schema argument is an explicit output argument."""
    return bool(getattr(arg, "is_out", False))


def _schema_non_alias_tensor_output_indices(op: OpOverload) -> list[int]:
    """Return tensor return indices that are not aliases of mutable inputs."""
    indices: list[int] = []
    for idx, ret in enumerate(op._schema.returns):
        if "Tensor" in str(ret.type) and ret.alias_info is None:
            indices.append(idx)
    return indices


def _schema_written_tensor_arg_count(op: OpOverload) -> int:
    """Count Tensor-valued arguments written by an op."""
    return sum(
        1
        for arg in op._schema.arguments
        if _is_write_arg(arg) and "Tensor" in str(arg.type)
    )


def _functional_variant_tensor_output_count(base_op: OpOverload) -> int:
    """Return expected tensor outputs for a functionalization variant.

    Mutable ops can have real tensor returns and/or alias returns for mutated
    inputs. Functionalization returns the real tensor returns plus updated
    copies of written tensor inputs.
    """
    return len(_schema_non_alias_tensor_output_indices(base_op)) + (
        _schema_written_tensor_arg_count(base_op)
    )


def _iter_packet_overloads(packet: Any | None) -> list[OpOverload]:
    """Return overloads from an overload packet, skipping lookup misses."""
    if packet is None:
        return []
    overloads: list[OpOverload] = []
    for overload_name in packet.overloads():
        overload = _get_packet_overload(packet, overload_name)
        if overload is not None:
            overloads.append(overload)
    return overloads


def _count_tensor_meta_values(value: object) -> int:
    """Count TensorMeta leaves in a value that may contain nested tuples/lists.

    Strategy rule indices are based on tensor leaves, not raw schema argument
    count.
    """
    if isinstance(value, TensorMeta):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(_count_tensor_meta_values(item) for item in value)
    return 0


def _clone_schema_info(
    schema_info: RuntimeSchemaInfo | None, *, needs_pytree: bool | None = None
) -> RuntimeSchemaInfo | None:
    """Copy schema cache metadata, optionally overriding pytree flattening.

    Auto-registered variants should inherit the base op cache behavior unless
    their argument structure changes, as foreach does with Tensor lists.
    """
    if schema_info is None:
        if needs_pytree is None:
            return None
        return RuntimeSchemaInfo(needs_pytree=needs_pytree)

    return RuntimeSchemaInfo(
        static_argnum=schema_info.static_argnum,
        static_kwargkey=(
            list(schema_info.static_kwargkey)
            if schema_info.static_kwargkey is not None
            else None
        ),
        needs_pytree=schema_info.needs_pytree if needs_pytree is None else needs_pytree,
    )


def _clone_strategy_info(
    info: _SingleDimStrategyInfo, func: _SingleDimStrategyFunc
) -> _SingleDimStrategyInfo:
    """Copy strategy options while replacing the underlying strategy function.

    Variant wrappers must preserve base-op flags such as uneven sharding support.
    """
    return _SingleDimStrategyInfo(
        func=func,
        allow_unbacked_sharding=info.allow_unbacked_sharding,
        allow_uneven_sharding=info.allow_uneven_sharding,
        different_mesh_args=(
            list(info.different_mesh_args)
            if info.different_mesh_args is not None
            else None
        ),
    )


def _canonical_variant_base_name(op: OpOverload) -> tuple[str, str]:
    """Return the namespace and base name shared by related ATen variants."""
    namespace, base_name = _op_namespace_and_base_name(op)
    if base_name.endswith("_functional"):
        return namespace, base_name.removesuffix("_functional")
    if op._schema.is_mutable and base_name.endswith("_"):
        return namespace, base_name.removesuffix("_")
    return namespace, base_name


def _strip_output_args_for_base_call(
    out_op: OpOverload,
    output_arg_names: set[str],
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> tuple[ArgsType, KwargsType]:
    """Drop explicit out args before invoking the functional base strategy.

    Base strategy functions do not expect the extra writable tensors from the
    out overload schema.
    """
    base_args: list[object] = []
    positional_idx = 0
    for arg in out_op._schema.arguments:
        if arg.kwarg_only:
            continue
        if positional_idx >= len(args_schema):
            break
        value = args_schema[positional_idx]
        positional_idx += 1
        if arg.name not in output_arg_names:
            base_args.append(value)

    base_kwargs = {
        name: value
        for name, value in kwargs_schema.items()
        if name not in output_arg_names
    }
    return tuple(base_args), base_kwargs


def _find_inplace_variant_overloads(base_op: OpOverload) -> list[OpOverload]:
    """Find an inplace overload that has the same tensor schema as base_op.

    Inplace variants can reuse the base strategy only when inputs and outputs
    have the same tensor structure.
    """
    namespace, base_name = _canonical_variant_base_name(base_op)
    if base_op._schema.is_mutable:
        return []
    if _is_foreach_like_op_name(base_op.name()):
        return []
    if any(_is_explicit_out_arg(arg) for arg in base_op._schema.arguments):
        return []

    packet = _get_overload_packet(namespace, f"{base_name}_")
    if packet is None:
        return []

    variants: list[OpOverload] = []
    for candidate in _iter_packet_overloads(packet):
        if not candidate._schema.is_mutable:
            continue
        if _schema_tensor_output_count(candidate) != _schema_tensor_output_count(
            base_op
        ):
            continue
        if _schema_args_match(base_op._schema.arguments, candidate._schema.arguments):
            variants.append(candidate)
    return variants


def _find_out_variant_overloads(
    base_op: OpOverload,
) -> list[tuple[OpOverload, tuple[str, ...]]]:
    """Find an out overload and return its written output arg names.

    Out variants need extra strategy inputs for the provided output tensors.
    """
    namespace, base_name = _canonical_variant_base_name(base_op)
    if _is_foreach_like_op_name(base_op.name()):
        return []
    if base_name.endswith("_functional"):
        return []
    if any(_is_explicit_out_arg(arg) for arg in base_op._schema.arguments):
        return []

    base_num_outputs = _schema_tensor_output_count(base_op)
    if base_num_outputs == 0:
        return []

    packet = _get_overload_packet(namespace, base_name)
    if packet is None:
        return []

    variants: list[tuple[OpOverload, tuple[str, ...]]] = []
    for candidate in _iter_packet_overloads(packet):
        if candidate is base_op:
            continue
        if not candidate._schema.is_mutable:
            continue
        if _schema_tensor_output_count(candidate) != base_num_outputs:
            continue

        output_args = [
            arg for arg in candidate._schema.arguments if _is_explicit_out_arg(arg)
        ]
        if len(output_args) != base_num_outputs:
            continue
        if not all("Tensor" in str(arg.type) for arg in output_args):
            continue

        non_output_args = [
            arg for arg in candidate._schema.arguments if not _is_explicit_out_arg(arg)
        ]
        if _schema_args_are_same(base_op._schema.arguments, non_output_args):
            variants.append((candidate, tuple(arg.name for arg in output_args)))

    return variants


def _find_functional_variant_overloads(base_op: OpOverload) -> list[OpOverload]:
    """Find the functional variant generated from a mutable base overload.

    Functional variants return updated copies of mutable inputs, so their extra
    outputs must follow the mutated input placements.
    """
    if not any(_is_write_arg(arg) for arg in base_op._schema.arguments):
        return []
    namespace, base_name = _canonical_variant_base_name(base_op)
    if _is_foreach_like_op_name(base_op.name()):
        return []
    if any(_is_explicit_out_arg(arg) for arg in base_op._schema.arguments):
        return []

    packet = _get_overload_packet(namespace, f"{base_name}_functional")
    if packet is None:
        return []

    variants: list[OpOverload] = []
    expected_outputs = _functional_variant_tensor_output_count(base_op)
    for candidate in _iter_packet_overloads(packet):
        if candidate._schema.is_mutable:
            continue
        if not _schema_args_match(
            base_op._schema.arguments, candidate._schema.arguments
        ):
            continue
        if _schema_tensor_output_count(candidate) != expected_outputs:
            continue
        variants.append(candidate)
    return variants


def _find_foreach_variants(base_op: OpOverload) -> list[OpOverload]:
    """Find elementwise foreach overloads that can reuse base_op's strategy.

    Foreach variants are registered in one pass because newly registered
    variants are not revisited by auto_register_op_variants.
    """
    namespace, base_name = _op_namespace_and_base_name(base_op)
    if namespace != "aten":
        return []
    if base_op._schema.is_mutable or _is_foreach_like_op_name(base_op.name()):
        return []
    if _schema_tensor_output_count(base_op) == 0:
        return []

    variants: list[OpOverload] = []
    for foreach_name in (f"_foreach_{base_name}", f"_foreach_{base_name}_"):
        packet = _get_overload_packet(namespace, foreach_name)
        if packet is None:
            continue
        for overload_name in packet.overloads():
            if "out" in overload_name:
                continue
            candidate = _get_packet_overload(packet, overload_name)
            if candidate is None:
                continue
            if _resolve_foreach_elementwise_overload(candidate) == base_op:
                variants.append(candidate)
    return variants


def _make_same_schema_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc, base_op: OpOverload
) -> _SingleDimStrategyFunc:
    """Wrap base_fn so a same-schema variant uses base-op strategy logic.

    Passing base_op keeps op-sensitive strategy functions on their original
    lookup paths.
    """

    def strategy(
        op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        return base_fn(base_op, args_schema, kwargs_schema)

    return strategy


def _make_out_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc,
    base_op: OpOverload,
    out_op: OpOverload,
    output_arg_names: tuple[str, ...],
) -> _SingleDimStrategyFunc:
    """Wrap base_fn so out tensors appear as extra strategy inputs.

    DTensor dispatch cannot redistribute out kwargs, so generated rules must
    make each output placement agree with its corresponding out tensor.
    """
    output_arg_name_set = set(output_arg_names)
    output_arg_to_index = {
        name: output_idx for output_idx, name in enumerate(output_arg_names)
    }
    base_num_outputs = _schema_tensor_output_count(base_op)

    def _output_arg_placements(
        rule: list[Placement | _ShardingPlaceholder],
        arg_name: str,
        value: object,
    ) -> list[Placement | _ShardingPlaceholder]:
        """Return placements to append for one explicit out argument."""
        tensor_count = _count_tensor_meta_values(value)
        if tensor_count == 0:
            return []
        if tensor_count > 1 and len(output_arg_names) == 1:
            return [rule[i] for i in range(base_num_outputs)]
        return [rule[output_arg_to_index[arg_name]]]

    def strategy(
        op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        base_args, base_kwargs = _strip_output_args_for_base_call(
            out_op, output_arg_name_set, args_schema, kwargs_schema
        )
        base_rules = base_fn(base_op, base_args, base_kwargs)
        rules: list[list[Placement | _ShardingPlaceholder]] = []

        for rule in base_rules:
            output_arg_placements: list[Placement | _ShardingPlaceholder] = []
            positional_idx = 0
            for arg in out_op._schema.arguments:
                if arg.kwarg_only:
                    continue
                if positional_idx >= len(args_schema):
                    break
                value = args_schema[positional_idx]
                positional_idx += 1
                if arg.name in output_arg_to_index:
                    output_arg_placements.extend(
                        _output_arg_placements(rule, arg.name, value)
                    )

            for name, value in kwargs_schema.items():
                if name in output_arg_to_index:
                    output_arg_placements.extend(
                        _output_arg_placements(rule, name, value)
                    )

            rules.append([*rule, *output_arg_placements])

        return rules

    return strategy


def _make_functional_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc, base_op: OpOverload
) -> _SingleDimStrategyFunc:
    """Wrap mutable base_fn for functional variants with extra mutated outputs.

    The added functional outputs represent updated mutable inputs and therefore
    inherit those input placements.
    """
    mutable_arg_names = {
        arg.name for arg in base_op._schema.arguments if _is_write_arg(arg)
    }
    base_num_outputs = _schema_tensor_output_count(base_op)
    non_alias_output_indices = _schema_non_alias_tensor_output_indices(base_op)

    def _mutable_input_rule_indices(
        args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> list[int]:
        """Map mutable schema args to their positions in a base strategy rule."""
        mutable_indices: list[int] = []
        tensor_input_idx = 0
        positional_idx = 0
        for arg in base_op._schema.arguments:
            value = None
            if arg.kwarg_only:
                value = kwargs_schema.get(arg.name)
            elif positional_idx < len(args_schema):
                value = args_schema[positional_idx]
                positional_idx += 1
            else:
                value = kwargs_schema.get(arg.name)

            tensor_count = _count_tensor_meta_values(value)
            if tensor_count == 0:
                continue
            if arg.name in mutable_arg_names:
                mutable_indices.append(base_num_outputs + tensor_input_idx)
            tensor_input_idx += tensor_count
        return mutable_indices

    def strategy(
        op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        mutable_input_indices = _mutable_input_rule_indices(args_schema, kwargs_schema)
        base_rules = base_fn(base_op, args_schema, kwargs_schema)
        rules: list[list[Placement | _ShardingPlaceholder]] = []
        for rule in base_rules:
            output_placements = [
                rule[output_idx] for output_idx in non_alias_output_indices
            ]
            output_placements.extend(
                rule[input_idx] for input_idx in mutable_input_indices
            )
            rules.append([*output_placements, *rule[base_num_outputs:]])
        return rules

    return strategy


def auto_register_op_variants() -> None:
    """Register schema-discovered variants for existing single-dim strategies.

    This keeps manual registrations focused on the semantic base op while
    covering mechanically related ATen overloads (in-place, out, functional,
    and foreach).
    """
    from torch.distributed.tensor._api import DTensor

    propagator = DTensor._op_dispatcher.sharding_propagator
    registry = propagator.op_single_dim_strategy_funcs
    already_registered = set(registry)

    for base_op, info in list(registry.items()):
        base_schema_info = propagator.op_to_schema_info_for_single_dim_strategy.get(
            base_op
        )

        for inplace_op in _find_inplace_variant_overloads(base_op):
            if inplace_op in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                inplace_op,
                _clone_strategy_info(
                    info, _make_same_schema_variant_strategy_fn(info.func, base_op)
                ),
                _clone_schema_info(base_schema_info),
            )
            already_registered.add(inplace_op)

        for out_variant in _find_out_variant_overloads(base_op):
            out_op, output_arg_names = out_variant
            if out_op in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                out_op,
                _clone_strategy_info(
                    info,
                    _make_out_variant_strategy_fn(
                        info.func, base_op, out_op, output_arg_names
                    ),
                ),
                _clone_schema_info(base_schema_info),
            )
            already_registered.add(out_op)

        for functional_op in _find_functional_variant_overloads(base_op):
            if functional_op in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                functional_op,
                _clone_strategy_info(
                    info, _make_functional_variant_strategy_fn(info.func, base_op)
                ),
                _clone_schema_info(base_schema_info),
            )
            already_registered.add(functional_op)

        for foreach_op in _find_foreach_variants(base_op):
            if foreach_op in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                foreach_op,
                info,
                RuntimeSchemaInfo(needs_pytree=True),
            )
            already_registered.add(foreach_op)
