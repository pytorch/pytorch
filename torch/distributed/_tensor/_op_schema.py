# mypy: allow-untyped-defs
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh


try:
    from torch.utils._cxx_pytree import tree_leaves, tree_map_only, TreeSpec
except ImportError:
    from torch.utils._pytree import (  # type: ignore[no-redef, assignment]
        tree_leaves,
        tree_map_only,
        TreeSpec,
    )


# Common type aliases
ArgsType = Tuple[object, ...]
KwargsType = Dict[str, object]
# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type sould
# be the same set of possibilities.
OutputSpecType = Optional[Union[DTensorSpec, Sequence[Optional[DTensorSpec]]]]


def _rebuild_tensor_from_dtensor_meta(arg) -> object:
    """
    This is used to propagate tensor metadata, must be under fake mode
    """
    assert arg.tensor_meta is not None, "DTensorSpec does not contain tensor_meta."
    return torch.empty_strided(
        arg.tensor_meta.shape,
        arg.tensor_meta.stride,
        dtype=arg.tensor_meta.dtype,
    )


def _is_inplace_op(op: OpOverload):
    # simple analysis of function schema to determine
    # if this is an inplace variant, it might not
    # be entirely correct, but it's good enough for now.
    return op._schema.name[-1] == "_"


def _is_out_variant_op(op: OpOverload):
    # simple analysis of function schema to determine
    # if this is an out variant, it might not
    # be entirely correct, but it's good enough for now.
    return "out" in op._schema.overload_name


def _pretty_print_spec(spec: object) -> str:
    if spec is None:
        return "None"
    elif isinstance(spec, DTensorSpec):
        return "".join([str(p) for p in spec.placements])
    elif isinstance(spec, Sequence):
        return "(" + ", ".join([_pretty_print_spec(s) for s in spec]) + ")"
    else:
        raise RuntimeError(f"Unknown spec type to print: spec={spec}")


@dataclass
class PlacementStrategy:
    """
    A placement strategy describes acceptable sharding placements of the output
    and the tensor arguments of an operation.

    note: when the op return value is a single DTensor object, output_specs is
    DTensorSpec; when the return value is a tuple of Optional[DTensor],
    output_specs is a tuple of Optional[DTensorSpec].
    """

    output_specs: Union[DTensorSpec, Tuple[Optional[DTensorSpec], ...]]
    input_specs: Optional[Sequence[DTensorSpec]] = None

    # redistribute costs for this op placement strategy
    # we need a nested list to record the cost for each
    # operand of this operator, and for each operand of
    # this operator it might have multiple placement strategies
    redistribute_cost: Optional[List[List[float]]] = None

    @cached_property
    def output_spec(self) -> DTensorSpec:
        """
        This function requires that the strategy have exactly one DTensorSpec as the
        output spec. If the output_specs is a tuple, we throw an exception.
        """
        if isinstance(self.output_specs, DTensorSpec):
            return self.output_specs
        else:
            raise ValueError(
                f"function output_spec expects a single DTensorSpec but got: {self.output_specs}"
            )

    def input_spec(self, index: int = 0) -> DTensorSpec:
        assert self.input_specs is not None, "input_specs of PlacementStrategy is None!"
        assert len(self.input_specs) > index, (
            f"Invalid index {index} for input_specs of length "
            f"{len(self.input_specs)}: {self.input_specs}"
        )
        return self.input_specs[index]

    def __str__(self) -> str:
        if self.input_specs is not None:
            input_specs_str = f"{_pretty_print_spec(self.input_specs)} -> "
        else:
            input_specs_str = ""
        output_spec_str = _pretty_print_spec(self.output_specs)
        return f"{input_specs_str}{output_spec_str}"


class StrategyType:
    """
    Base class type for op strategy, We have two StrategyType:
        OpStrategy and TupleStrategy
    """

    pass


class OpStrategy(StrategyType):
    """
    OpStrategy that consists of a list of placement strategies associated with the op
    """

    def __init__(self, strategies: List[PlacementStrategy]) -> None:
        super().__init__()
        self.strategies: List[PlacementStrategy] = strategies

    def __str__(self) -> str:
        strategy_list_str = ", ".join([str(strategy) for strategy in self.strategies])
        mesh_shape = self.mesh_shape
        return f"[{strategy_list_str}] @ mesh: {mesh_shape}"

    def max_num_shards(self) -> int:
        """
        Returns the max number of shards across all placement strategies
        """
        return max(strategy.output_spec.num_shards for strategy in self.strategies)

    @property
    def mesh_shape(self):
        output_spec = self.strategies[0].output_specs
        if isinstance(output_spec, DTensorSpec):
            return output_spec.mesh.shape
        else:
            assert isinstance(
                output_spec, tuple
            ), "found no DTensorSpec in the OpStrategy!"
            assert output_spec[0] is not None
            return output_spec[0].mesh.shape

    @property
    def ndim(self):
        return self.strategies[0].output_spec.ndim

    @property
    def shape(self):
        return self.strategies[0].output_spec.shape


class TupleStrategy(StrategyType):
    """
    TupleStrategy represents the output strategy of this op is a tuple
    of strategy, i.e. If the output of this op is a tuple of tensors or list of tensors
    with possibly different placement strategies, we should return a TupleStrategy that
    contains a tuple of OpStrategy, where each child represents the sharding strategy
    of "each element" of the tuple/list of tensors the op returns.

    NOTE: if the output of the op is a List[Tensor] and they share the same placement
    strategy, then we should return a single OpStrategy instead of a TupleStrategy
    """

    def __init__(self, childs: Sequence[StrategyType]) -> None:
        super().__init__()
        self.childs: Sequence[StrategyType] = childs

    def __str__(self) -> str:
        child_strategies_str = ", ".join(
            [f"{str(strat)}" for idx, strat in enumerate(self.childs)]
        )
        return f"TupleStrategy({child_strategies_str})"


@dataclass
class RuntimeSchemaInfo:
    """
    RuntimeSchemaInfo stores the operator schema related information for runtime (eager)
    execution. This is mainly used for two ways: 1. to generate hash for args to determine
    whether to re-run sharding prop or not 2. to determine if we need pytree
    """

    # This static_argnum records static arg "starting index" for ops that have non-tensor
    # args/kwargs which would affect sharding propagation results. All args starting from
    # this index would be hashed to our sharding cache.
    # Note that only a few ops need this information, e.g. view, transpose, var.dim, etc.
    static_argnum: int = 100
    # This static_kwargkey records static kwarg names which would affect sharding prop
    static_kwargkey: Optional[List[str]] = None
    # each op can decide if it wants to use pytree flatten/unflatten during operator
    # eager execution, by default we don't need to do flatten/unflatten, only if the
    # op indicate it needs to, this is to accelerate eager performance.
    needs_pytree: bool = False


@dataclass
class OpSchema:
    """
    OpSchema is a data class that describes an operator input schemas, it
    includes DTensor DTensorSpecs and non-tensor args/kwargs (positional order
    preserved). It is mainly used by the dispatching logic below to run things like
    sharding propagation.

    NOTE: this should be used as a read only data class
    TODO: make this a frozen dataclass

    Args:
        op: the operator overload we are intercepting
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec
    """

    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType

    schema_info: Optional[RuntimeSchemaInfo] = None

    @property
    def args_spec(self) -> Tuple[DTensorSpec, ...]:
        """
        args_spec: Tuple[DTensorSpec, ...]: contains a clean list of args spec list
            with NO non-DTensor positional arguments (i.e. int/float/tuple, etc)
            mainly used by sharding propagation to propagate the output spec
        """
        args = (
            tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.args_schema
        )
        return tuple(item for item in args if isinstance(item, DTensorSpec))

    @property
    def args_strategy(self) -> Tuple[OpStrategy, ...]:
        # filter out non-relevant values from args schema to get a clean OpStrategy list
        # separate with args_spec for the ease of type annotation
        # TODO: see if we should merge this with args_spec
        args = (
            tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else self.args_schema
        )
        return tuple(item for item in args if isinstance(item, OpStrategy))

    def __repr__(self) -> str:
        args_schema = ", ".join([str(arg_schema) for arg_schema in self.args_schema])
        return (
            f"OpSchema(op={self.op},"
            f" args_schema=({args_schema}),"
            f" kwargs_schema={self.kwargs_schema})"
        )

    def __str__(self) -> str:
        args_sharding: List[str] = []
        mesh_shape = None
        for arg in self.args_schema:
            if isinstance(arg, DTensorSpec):
                args_sharding.append(str(arg))
                mesh_shape = arg.mesh.shape
            elif isinstance(arg, OpStrategy):
                assert len(arg.strategies) == 1
                args_sharding.append(_pretty_print_spec(arg.strategies[0].output_specs))
                mesh_shape = arg.mesh_shape
            elif isinstance(arg, TupleStrategy):
                first_op_strtgy = arg.childs[0]
                assert isinstance(first_op_strtgy, OpStrategy)
                mesh_shape = first_op_strtgy.mesh_shape
                args_sharding.append(str(arg))
            else:
                args_sharding.append(str(arg))
        return f"Op(op={self.op}, args_sharding={', '.join(args_sharding)} @ mesh: {mesh_shape})"

    def __post_init__(self) -> None:
        has_symints = False
        for a in self.args_schema:
            if isinstance(a, DTensorSpec) and a.tensor_meta is not None:
                if any(isinstance(s, torch.SymInt) for s in a.tensor_meta.shape):
                    has_symints = True
                    break
        self.has_symints = has_symints

    def arg_type_tensor_or_tensor_list_like(self, arg_idx: int) -> bool:
        arg = self.args_schema[arg_idx]
        is_tensor = isinstance(arg, DTensorSpec)
        if is_tensor:
            return True

        if not isinstance(arg, list):
            return False

        return all(isinstance(e, DTensorSpec) or e is None for e in arg)

    def return_type_tuple_tensor_like(self) -> bool:
        # all dispatch ops could only return Tuple[Tensor] or have None/ints/floats
        # in the tuple, but the first element must be a Tensor, so this check is enough
        return_types = self.op._schema.returns
        return len(return_types) > 1 and isinstance(
            return_types[0].type, torch.TensorType
        )

    def return_type_tensor(self) -> bool:
        return_types = self.op._schema.returns
        # all dispatch ops only return Tensor or Tuple[Tensor] for tensor like
        # return types, so this check is enough for tensor like types
        return isinstance(return_types[0].type, torch.TensorType)

    def __hash__(self) -> int:
        # Only hash args and kwargs that op indicates to hash
        if not self.schema_info:
            static_argnum = len(self.args_schema)
            static_kwargkey = None
        else:
            static_argnum = self.schema_info.static_argnum
            static_kwargkey = self.schema_info.static_kwargkey

        args_to_hash = tuple(
            tuple(e) if isinstance(e, list) else e
            for i, e in enumerate(self.args_schema)
            if self.arg_type_tensor_or_tensor_list_like(i) or i >= static_argnum
        )
        if static_kwargkey is not None:
            kwargs_to_hash = tuple(
                self.kwargs_schema.get(k, None) for k in static_kwargkey
            )
            return hash((self.op, args_to_hash, kwargs_to_hash))
        else:
            return hash((self.op, args_to_hash))

    def __eq__(self, other: object) -> bool:
        # early return checks
        if not isinstance(other, OpSchema):
            return False

        if self.op != other.op:
            return False

        if len(self.args_schema) != len(other.args_schema):
            return False

        # compare each element and early return if any of them is different
        if not self.schema_info:
            static_argnum = len(self.args_schema)
            static_kwargkey = None
        else:
            static_argnum = self.schema_info.static_argnum
            static_kwargkey = self.schema_info.static_kwargkey

        for i, (self_arg, other_arg) in enumerate(
            zip(self.args_schema, other.args_schema)
        ):
            if isinstance(self_arg, DTensorSpec) and self_arg != other_arg:
                return False
            elif i >= static_argnum and self_arg != other_arg:
                return False

        # check kwarg equality when there's a static kwarg key
        if static_kwargkey:
            for key in static_kwargkey:
                if self.kwargs_schema.get(key, None) != other.kwargs_schema.get(
                    key, None
                ):
                    return False

        return True

    def gen_fake_args(self) -> ArgsType:
        """
        gen_fake_args: generate fake args for the operator, this is mainly used
            by sharding propagation rules to generate fake args for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(
            DTensorSpec, _rebuild_tensor_from_dtensor_meta, self.args_schema
        )

    def gen_fake_kwargs(self) -> KwargsType:
        """
        gen_fake_kwargs: generate fake kwargs for the operator, this is mainly used
            by sharding propagation rules to generate fake kwargs for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(
            DTensorSpec, _rebuild_tensor_from_dtensor_meta, self.kwargs_schema
        )

    def _inplace_rewrap_schema_suggestion(self, origin_schema: "OpSchema") -> None:
        suggestion_args_spec = self.args_spec
        new_arg_schema: List[object] = []
        idx_of_args_spec = 0
        if (
            origin_schema.schema_info is not None
            and origin_schema.schema_info.needs_pytree
        ):
            args_schema: Sequence[Any] = tree_leaves(origin_schema.args_schema)
        else:
            args_schema = origin_schema.args_schema
        for arg in args_schema:
            if isinstance(arg, DTensorSpec):
                new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
                idx_of_args_spec += 1
            else:
                new_arg_schema.append(arg)
        self.args_schema = tuple(new_arg_schema)
        self.kwargs_schema = origin_schema.kwargs_schema


@dataclass
class OutputSharding:
    """
    OutputSharding is a data class that is used by the sharding propagation
    rules, it could set the output_spec upon successful propagation, and if
    it failed, output_spec would become None and sharding propagation rules
    could give a list of suggestions for inputs to reshard.

    NOTE: the schema_suggestion generated by sharding propagation should be
    exactly the same as the operator OpSchema, except the DTensor DTensorSpecs
    """

    output_spec: OutputSpecType
    redistribute_schema: Optional[OpSchema] = None
    needs_redistribute: bool = False


@dataclass
class OpInfo:
    """
    All Runtime Op execution info are packed here
    """

    mesh: DeviceMesh
    schema: OpSchema
    flat_args_schema: List[object]
    local_args: Sequence[object]
    local_kwargs: Dict[str, object]
    args_tree_spec: Optional[TreeSpec] = None

    # the output sharding info
    output_sharding: Optional[OutputSharding] = None
