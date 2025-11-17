# mypy: allow-untyped-defs
"""
DTensor operator schema definitions and utilities.

This module defines the core data structures and utilities for describing and managing
distributed tensor operations in PyTorch's DTensor system. It provides the foundational
schema types used for sharding propagation, operator strategy selection, and distributed
execution planning.

Key components:
- OpSpec: Describes acceptable sharding placements for operations
- OpStrategy: Represents the possible sharding strategies for an operator
- TupleStrategy: Container for multiple strategies when ops have tuple/list of tensors input
- OpSchema: Describes operator input/output schemas with DTensorSpecs
- OutputSharding: Manages output sharding specifications and redistribution
- RuntimeSchemaInfo: Runtime execution metadata for operators
- OpInfo: Complete runtime operator execution information

These schema definitions enable the DTensor system to:
1. Propagate tensor sharding information to the operator outputs
2. Greedily select sharding strategies for distributed operations
3. Plan and execute tensor redistributions when needed
4. Cache sharding decisions for performance optimization
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional, Union
from typing_extensions import deprecated

import torch
from torch._C import (
    _DTensor_OpSchema_post_init,
    _DTensor_OpSchema_recompute_comparison_key,
)
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Placement


try:
    from torch.utils._cxx_pytree import (
        register_pytree_node,
        tree_leaves,
        tree_map_only,
        TreeSpec,
    )
except ImportError:
    from torch.utils._pytree import (  # type: ignore[no-redef, assignment]
        register_pytree_node,
        tree_leaves,
        tree_map_only,
        TreeSpec,
    )


# Common type aliases
ArgsType = tuple[object, ...]
KwargsType = dict[str, object]

PlacementList = list[Optional[Placement]]

# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type should
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
class OpSpec:
    """
    An OpSpec describes an acceptable sharding placements of an operation, with the
    specified DTensorSpecs for both the output and the inputs.

    note: when the op return value is a single DTensor object, output_specs is
    DTensorSpec; when the return value is a tuple of Optional[DTensor],
    output_specs is a tuple of Optional[DTensorSpec].

    note: we MUST produce an DTensorSpec for every output that is a Tensor.  None
    entries only occur for non-Tensor outputs (e.g., operators that return Optional[Tensor],
    or non-Tensor outputs.)

    invariant: the DeviceMesh on all DTensorSpec must be the same
    """

    # output_specs and input_specs are related: for this op, given these input_specs,
    # this is the way the output would look
    output_specs: Union[DTensorSpec, tuple[Optional[DTensorSpec], ...]]
    input_specs: Optional[Sequence[DTensorSpec]] = None

    """
    redistribute_cost tells how expensive it is to redistribute a given input into the
    placement specified in this OpSpec.

    outer list: one entry (list) per (tensor) input in the op's arg schema
    inner list: one entry (cost value) per possible sharding spec for that input

    Example:
    -------
    another_op() -> tensor_a   # another_op produces the output that becomes our first input
    my_op(tensor_a)

    Let's assume this OpSpec's input_specs are [Replicate()],
    but another_op() supports 2 strategies (OpSpecs) which produce outputs of
       Replicate()
       Shard(0)

    In this example, redistribute_costs would look like this
    [
        # one row representing "my_op's first input" (tensor_a)
        [
            # two entries, one for each strategies supported by another_op
            0.0,  # cost of redistributing tensor_a from 'Replicate()'
            K,    # cost of redistributing tensor_a from 'Shard(0)'
        ],
    """
    redistribute_cost: Optional[list[list[float]]] = None

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

    @cached_property
    def mesh(self):
        if isinstance(self.output_specs, DTensorSpec):
            return self.output_specs.mesh
        elif isinstance(self.output_specs, tuple):
            out_spec = self.output_specs[0]
            assert isinstance(out_spec, DTensorSpec)
            return out_spec.mesh
        else:
            raise ValueError(
                f"function output_spec expects a single DTensorSpec or a tuple of DTensorSpec but got: {self.output_specs}"
            )

    def input_spec(self, index: int = 0) -> DTensorSpec:
        assert self.input_specs is not None, "input_specs of OpSpec is None!"
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


class OpStrategy(StrategyType):
    """
    OpStrategy that consists of a list of sharding strategies associated with the op,
    where each strategy is an OpSpec that describes the acceptable input/output sharding.

    invariant: the DeviceMesh on all OpSpec must be the same
    """

    def __init__(self, strategies: list[OpSpec]) -> None:
        super().__init__()
        self.strategies: list[OpSpec] = strategies

    def __str__(self) -> str:
        strategy_list_str = ", ".join([str(strategy) for strategy in self.strategies])
        mesh_shape = self.mesh_shape
        return f"[{strategy_list_str}] @ mesh: {mesh_shape}"

    def max_num_shards(self) -> int:
        """
        Returns the max number of shards across all OpSpecs
        """
        return max(strategy.output_spec.num_shards for strategy in self.strategies)

    @property
    def mesh(self):
        return self.strategies[0].mesh

    @property
    def mesh_shape(self):
        return self.strategies[0].mesh.shape

    @property
    def ndim(self):
        return self.strategies[0].output_spec.ndim

    @property
    def shape(self):
        return self.strategies[0].output_spec.shape


class TupleStrategy(StrategyType):
    """
    TupleStrategy is a special case for operators that are fundamentally compound or batched such that some subset
    of the inputs and outputs are completely unrelated to some other subset.

    Generally, foreach_* ops are the most common use-case for TupleStrategy, because they accept lists of inputs,
    but operate independently on each input or tuple of zipped inputs.

    For example, [out_a, out_b] = torch.foreach_add([a,  b], scalar): input a's sharding only affects out_a's sharding,
    independent of b and out_b.

    An example of an operator that should NOT use TupleStrategy is torch.split.  It produces a List[Tensor]
    as its output, but the sharding decision of one output is bound together with the decision
    of each other output and the common input.
    """

    def __init__(
        self,
        children: Sequence[StrategyType],
    ) -> None:
        super().__init__()
        self.children: Sequence[StrategyType] = children

    @property
    @deprecated(
        "TupleStrategy.childs is deprecated, use TupleStrategy.children instead.",  # codespell:ignore childs
        category=FutureWarning,
    )
    def childs(self) -> Sequence[StrategyType]:  # codespell:ignore childs
        """
        Alias for children, to maintain backward compatibility.
        """
        return self.children

    def child_mesh(self, index: int) -> DeviceMesh:
        op_strategy = self.children[index]
        assert isinstance(op_strategy, OpStrategy)
        return op_strategy.mesh

    def __str__(self) -> str:
        child_strategies_str = ", ".join(
            [f"{str(strat)}" for idx, strat in enumerate(self.children)]
        )
        return f"TupleStrategy({child_strategies_str})"


try:
    register_pytree_node(
        TupleStrategy,
        lambda node: (node.children, None),
        lambda children, _: TupleStrategy(tuple(children)),
    )
except ValueError:
    # already registered TupleStrategy, skip
    pass


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
    static_kwargkey: Optional[list[str]] = None
    # each op can decide if it wants to use pytree flatten/unflatten during operator
    # eager execution, by default we don't need to do flatten/unflatten, only if the
    # op indicate it needs to, this is to accelerate eager performance.
    needs_pytree: bool = False


@dataclass
class OpSchema:
    """
    OpSchema is a data class that describes an operator input schemas, it includes
    DTensorSpecs/OpStrategies (instead of DTensor) and non-tensor args/kwargs (positional
    order preserved). It is mainly used by the DTensor's dispatching logic to perform various
    actions (i.e. sharding propagation, caching sharding decisions, redistribute, etc.)

    NOTE: this must be used as a read only data class
    TODO: make this a frozen dataclass

    Args:
        op: the operator overload we are intercepting
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec or OpStrategy
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec or OpStrategy
    """

    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType

    schema_info: Optional[RuntimeSchemaInfo] = None

    _comparison_key: Optional[tuple[object, ...]] = None

    has_symints: bool = field(init=False)

    @property
    def args_spec(self) -> tuple[DTensorSpec, ...]:
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
    def args_strategy(self) -> tuple[OpStrategy, ...]:
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
        args_schema: list[str] = []
        mesh_shape = None
        for arg in self.args_schema:
            if isinstance(arg, DTensorSpec):
                args_schema.append(str(arg))
                mesh_shape = arg.mesh.shape
            elif isinstance(arg, OpStrategy):
                assert len(arg.strategies) == 1
                args_schema.append(_pretty_print_spec(arg.strategies[0].output_specs))
                mesh_shape = arg.mesh_shape
            elif isinstance(arg, TupleStrategy):
                first_op_strategy = arg.children[0]
                assert isinstance(first_op_strategy, OpStrategy)
                mesh_shape = first_op_strategy.mesh_shape
                args_schema.append(str(arg))
            else:
                args_schema.append(str(arg))
        return f"Op(op={self.op}, args_schema={', '.join(args_schema)} @ mesh: {mesh_shape})"

    def __post_init__(self) -> None:
        _DTensor_OpSchema_post_init(self)

    def arg_type_tensor_or_tensor_list_like(self, arg: object) -> bool:
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

    def return_type_list_tensor_like(self) -> bool:
        # returns True if the return type is a List
        return_types = self.op._schema.returns
        return len(return_types) == 1 and isinstance(
            return_types[0].type, torch.ListType
        )

    def return_type_tensor(self) -> bool:
        return_types = self.op._schema.returns
        # all dispatch ops only return Tensor or Tuple[Tensor] for tensor like
        # return types, so this check is enough for tensor like types
        return isinstance(return_types[0].type, torch.TensorType)

    def get_mesh_from_args(self, validate: bool = True) -> DeviceMesh:
        """
        This util can be used to get a mesh from the OpSchema that contains multiple
        DTensors as arguments. When `validate` is True, it will try to validate that all the
        arguments have the same mesh to avoid unexpected cross mesh errors.

        NOTE: this util currently does not handle TupleStrategy when `validate=True`,
        this is because for TupleStrategy there could be different types of checks, i.e.:
            - for stack and cat like op, we need to check within a TupleStrategy is every
              input is on the same mesh
            - for foreach like ops we need to check "zipped" inputs are on the same mesh
              for each index.
        """
        first_arg = self.args_schema[0]
        if isinstance(first_arg, (DTensorSpec, OpStrategy)):
            mesh = first_arg.mesh
        elif isinstance(first_arg, (list, tuple, TupleStrategy)):
            first_elem = (
                first_arg.children[0]
                if isinstance(first_arg, TupleStrategy)
                else first_arg[0]
            )
            assert isinstance(first_elem, (DTensorSpec, OpStrategy))
            mesh = first_elem.mesh
        else:
            raise ValueError(f"Cannot find device mesh from args for op : {self.op}.")

        if validate:
            for arg in self.args_schema[1:]:
                if isinstance(arg, (DTensorSpec, OpStrategy)) and arg.mesh != mesh:
                    raise RuntimeError(
                        f"DTensor does not support cross-mesh operation on {self.op}! "
                        f"Got meshes: {mesh} {arg.mesh}. "
                        f"Please make sure all the arguments have the same DeviceMesh."
                    )

        return mesh

    def is_inplace_op(self) -> bool:
        # simple analysis of function schema to determine
        # if this is an inplace variant, it might not
        # be entirely correct, but it's good enough for now.
        return self.op._schema.name[-1] == "_"

    def is_out_variant_op(self) -> bool:
        # simple analysis of function schema to determine
        # if this is an out variant, it might not
        # be entirely correct, but it's good enough for now.
        return "out" in self.op._schema.overload_name

    def is_view_op(self) -> bool:
        return self.op._schema._is_view_op()

    def _recompute_comparison_key(self) -> None:
        _DTensor_OpSchema_recompute_comparison_key(self)

    def __hash__(self) -> int:
        return hash(self._comparison_key)

    def __eq__(self, other: object) -> bool:
        # early return checks
        if not isinstance(other, OpSchema):
            return False

        if self.op != other.op:
            return False

        if len(self.args_schema) != len(other.args_schema):
            return False

        return self._comparison_key == other._comparison_key

    def gen_fake_args(self) -> ArgsType:
        """
        gen_fake_args: generate fake args for the operator, this is mainly used
            by sharding propagation rules to generate fake args for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(
            DTensorSpec,
            _rebuild_tensor_from_dtensor_meta,
            self.args_schema,
            is_leaf=lambda x: isinstance(x, DTensorSpec),
        )

    def gen_fake_kwargs(self) -> KwargsType:
        """
        gen_fake_kwargs: generate fake kwargs for the operator, this is mainly used
            by sharding propagation rules to generate fake kwargs for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(
            DTensorSpec,
            _rebuild_tensor_from_dtensor_meta,
            self.kwargs_schema,
            is_leaf=lambda x: isinstance(x, DTensorSpec),
        )

    def _inplace_rewrap_schema_suggestion(self, origin_schema: "OpSchema") -> None:
        suggestion_args_spec = self.args_spec
        new_arg_schema: list[object] = []
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
        self._recompute_comparison_key()


@dataclass
class OutputSharding:
    """
    OutputSharding is a data class that is used by the sharding propagation,
    it could set the output_spec upon successful propagation. If needs_redistribute
    is set to True, a redistribute_schema would be returned together to indicate
    the input arguments needs to be redistributed before the op execution.

    NOTE: the redistribute_schema generated by sharding propagation should be
    exactly the same as the operator OpSchema, except the DTensorSpecs
    """

    # specifies the output sharding pattern
    output_spec: OutputSpecType
    # schema for redistribution if needed
    redistribute_schema: Optional[OpSchema] = None
    # flag indicating if inputs need redistribution
    needs_redistribute: bool = False
    # flag to use values from `redistribute_schema`
    use_val_from_redistribute_schema: bool = False

    @cached_property
    def mesh(self):
        if isinstance(self.output_spec, DTensorSpec):
            return self.output_spec.mesh
        elif isinstance(self.output_spec, tuple):
            out_spec = self.output_spec[0]
            if isinstance(out_spec, DTensorSpec):
                return out_spec.mesh
            else:
                raise ValueError(f"Unknown output spec type: {type(out_spec)}")
        else:
            raise ValueError(f"Unknown output spec type: {type(self.output_spec)}")


@dataclass
class OpInfo:
    """
    All Runtime Op execution info are packed here
    """

    # The first compute device mesh recorded from args
    # NOTE: one op could have multiple meshes from its args. We just record the first
    # mesh here to check if current rank should participate in computation or not.
    compute_mesh: DeviceMesh

    # compete runtime operator infos
    schema: OpSchema
    flat_args_schema: list[object]
    local_args: Sequence[object]
    local_kwargs: dict[str, object]
    args_tree_spec: Optional[TreeSpec] = None

    # the output sharding info
    output_sharding: Optional[OutputSharding] = None
