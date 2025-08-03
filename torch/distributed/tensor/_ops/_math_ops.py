# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast, Optional, Union

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import (
    as_list,
    expand_to_full_mesh_op_strategy,
    generate_redistribute_costs,
    is_tensor_evenly_shardable,
    normalize_dim,
    normalize_dims,
    register_op_strategy,
)
from torch.distributed.tensor._utils import normalize_to_torch_size
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


@dataclass(frozen=True)
class NormReduction:
    norm_type: Union[int, float, str]


ReductionOpType = Union[NormReduction, str]


@dataclass(frozen=True)
class _NormPartial(Partial):
    """
    This placement is used for partial vector norm.

    For p-norms (where p not inf or -inf), the p-norm over n elements computes
        (sum_i x_i^p)^(1/p)
    where the sum is from i=1 to n. The reduction op is the p-norm itself.
    For example, consider 2 ranks, a (4,) tensor sharded on dim-0, and 2-norm:
        Rank 0: [t1, t2] | Rank 1: [t3, t4]
    After computing 2-norm per gradient (partial placement):
        Rank 0: [sqrt(t1^2 + t2^2)] | Rank 1: [sqrt(t3^2 + t4^2)]
    Converting from partial to replicate wants to ultimately get:
        Rank 0/1: [sqrt(t1^2 + t2^2 + t3^2 + t4^2)]
    This can be achieved by computing 2-norm on each rank's result. This holds
    similarly for inf and -inf norm. For 0-norm, the reduction op is sum.
    """

    norm_type: Union[int, float, str] = 2

    def __post_init__(self):
        """Set the appropriate reduce op based on the norm type."""
        # Use `object.__setattr__` to bypass frozen checks
        if self.norm_type in (float("inf"), "inf"):
            object.__setattr__(self, "reduce_op", "max")
        elif self.norm_type in (float("-inf"), "-inf"):
            object.__setattr__(self, "reduce_op", "min")
        elif isinstance(self.norm_type, (int, float)):
            object.__setattr__(self, "reduce_op", "sum")
        else:
            raise NotImplementedError(f"Unsupported norm type: {self.norm_type}")

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        For example, consider 4 ranks, a (3,) replicated tensor, and 2-norm:
            Ranks 0 and 1: sqrt(t1^2 + t2^2 + t3^3)
        To convert from replicated to partial, we want f(x) such that
            sqrt(t1^2 + t2^2 + t3^3) = sqrt(4f(t1)^2 + 4f(t2)^2 + 4f(t3)^2)
                                     = sqrt(4) sqrt(f(t1)^2 + f(t2)^2 + f(t3)^2).
        One such f(x) is f(x) = x / sqrt(4). This generalizes to d ranks and
        p-norm as f(x) = x / d^(1/p).
        """
        if self.reduce_op in ("max", "min"):
            return tensor
        elif self.reduce_op == "sum":
            if self.norm_type == 0:
                raise NotImplementedError(f"Unsupported norm type:: {self.norm_type}")
            elif self.norm_type == 1:
                return tensor / mesh.size(mesh_dim)
            assert isinstance(self.norm_type, (int, float))
            return tensor / math.pow(mesh.size(mesh_dim), 1 / self.norm_type)
        raise NotImplementedError(self.reduce_op)

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        assert isinstance(shard_spec, Shard), f"{shard_spec}"
        tensor = self._pre_reduce_transform(tensor)
        reduced_tensor = super()._reduce_shard_value(tensor, mesh, mesh_dim, shard_spec)
        return self._post_reduce_transform(reduced_tensor)

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        tensor = self._pre_reduce_transform(tensor)
        reduced_tensor = super()._reduce_value(tensor, mesh, mesh_dim)
        return self._post_reduce_transform(reduced_tensor)

    def _pre_reduce_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduce_op == "sum":
            assert isinstance(self.norm_type, (int, float)), f"{self.norm_type}"
            if self.norm_type != 0 and self.norm_type != 1:
                return tensor**self.norm_type
        return tensor

    def _post_reduce_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduce_op == "sum":
            assert isinstance(self.norm_type, (int, float)), f"{self.norm_type}"
            if self.norm_type != 0 and self.norm_type != 1:
                return tensor ** (1.0 / self.norm_type)
        return tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _NormPartial):
            return False
        return self.norm_type == other.norm_type

    def __hash__(self) -> int:
        return 1 + hash(self.norm_type)


def _infer_reduction_dims(dims_arg: object, ndim: int) -> Optional[list[int]]:
    if dims_arg is None:
        return None
    dims = cast(list[int], as_list(dims_arg))
    dims = cast(list[int], normalize_dims(dims, ndim))
    empty_dims = [[0], [-1], []]
    if ndim == 0 and dims_arg in empty_dims:
        return None
    return dims


def _infer_reduce_dims_map(
    reduction_dims: list[int], input_ndim: int, keep_dim=False
) -> list[int]:
    reduction_dims_map = []
    new_dim_count = 0
    for input_dim in range(input_ndim):
        if input_dim in reduction_dims and not keep_dim:
            # if input dim in reduction dims, mark it as -1
            reduction_dims_map.append(-1)
        else:
            # otherwise mark it as the new dim
            reduction_dims_map.append(new_dim_count)
            new_dim_count += 1

    return reduction_dims_map


def _replicate_dims_start_at(
    placements: Sequence[Placement], start_dim: int = 0
) -> tuple[Placement, ...]:
    new_placements: list[Placement] = []
    for p in placements:
        if p.is_partial() or (isinstance(p, Shard) and p.dim >= start_dim):
            new_placements.append(Replicate())  # make it replicate
        else:
            new_placements.append(p)  # keep the placement
    return tuple(new_placements)


# return new_placements which align with placements but skip the skipped_dim
def _skip_dim(
    placements: tuple[Placement, ...], skipped_dim: int
) -> tuple[Placement, ...]:
    new_placements: list[Placement] = []
    for p in placements:
        if isinstance(p, Shard) and p.dim >= skipped_dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)
    return tuple(new_placements)


def replicate_reduction_dims(
    placements: tuple[Placement, ...], reduction_dims: list[int]
) -> tuple[Placement, ...]:
    # replicate the reduction dims if not reduction_linear
    new_placements: list[Placement] = []

    for p in placements:
        if p.is_partial():
            new_placements.append(Replicate())
        elif isinstance(p, Shard) and p.dim in reduction_dims:
            new_placements.append(Replicate())
        else:
            new_placements.append(p)

    return tuple(new_placements)


def map_placements_after_reduction(
    placements: tuple[Placement, ...],
    reduction_dims: list[int],
    reduction_dims_map: list[int],
    reduction_op: ReductionOpType,
) -> tuple[Placement, ...]:
    """
    Map each placement based on the output shape after reduction.
    """
    new_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            new_placements.append(placement)
        else:
            assert isinstance(placement, Shard)
            shard_dim = placement.dim
            new_shard_dim = reduction_dims_map[shard_dim]
            if new_shard_dim == -1 or shard_dim in reduction_dims:
                # if new_shard_dim collapsed or its in the reduction dims
                # (i.e. for the case where keepdims=True), we generate partial
                new_placements.append(get_placement_from_reduction_op(reduction_op))
            else:
                new_placements.append(Shard(new_shard_dim))
    return tuple(new_placements)


def get_placement_from_reduction_op(reduction_op: ReductionOpType) -> Placement:
    if isinstance(reduction_op, NormReduction):
        return _NormPartial(norm_type=reduction_op.norm_type)
    return Partial(reduction_op)


def common_reduction_strategy(
    input_strategy: OpStrategy,
    reduce_dims: list[int],
    keep_dim: bool = False,
    reduction_linear: bool = True,
    reduction_op: ReductionOpType = "sum",
) -> OpStrategy:
    """
    reduction_linear means that the reduction `f` follows this rule:
        f([f(a), f(b)]) = f([a, b])

    reduction linear should be super set of linearity.
    """
    # by default follow reduction input strategy
    reduction_strategy = OpStrategy([])

    for op_spec in input_strategy.strategies:
        if not reduction_linear:
            # input placements for this strategy should clear out pending sum and sharding
            # on the reduction dimension
            input_placements = replicate_reduction_dims(
                op_spec.output_spec.placements, reduce_dims
            )
        else:
            input_placements = op_spec.output_spec.placements

        input_spec = DTensorSpec(
            mesh=input_strategy.mesh,
            placements=input_placements,
            tensor_meta=op_spec.output_spec.tensor_meta,
        )

        reduce_dims_map = _infer_reduce_dims_map(reduce_dims, input_spec.ndim, keep_dim)
        out_placements = map_placements_after_reduction(
            input_spec.placements, reduce_dims, reduce_dims_map, reduction_op
        )
        redistribute_cost = [generate_redistribute_costs(input_strategy, input_spec)]
        reduction_strategy.strategies.append(
            OpSpec(
                output_specs=DTensorSpec(
                    mesh=input_strategy.mesh,
                    placements=out_placements,
                ),
                input_specs=(input_spec,),
                redistribute_cost=redistribute_cost,
            )
        )

    return reduction_strategy


LINEAR_REDUCTION_OP_MAP = {
    aten.all.default: "sum",
    aten.all.dim: "sum",
    aten.sum.default: "sum",
    aten.sum.dim_IntList: "sum",
    aten.prod.default: "product",
    aten.prod.dim_int: "product",
    aten.prod.int_out: "product",
    aten.mean.default: "avg",
    aten.mean.dim: "avg",
    aten.mean.out: "avg",
    aten.max.default: "max",
    aten.max.dim: "max",
    aten.max.out: "max",
    aten.min.default: "min",
    aten.min.dim: "min",
    aten.min.out: "min",
    aten.any.default: "sum",
    aten.any.dim: "sum",
    aten.any.out: "sum",
    aten.amax.default: "max",
    aten.amax.out: "max",
    aten.amin.default: "min",
    aten.amin.out: "min",
}


@register_op_strategy(
    list(LINEAR_REDUCTION_OP_MAP.keys()), schema_info=RuntimeSchemaInfo(1)
)
def linear_reduction_strategy(op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)

    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.ndim)

    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims

    keep_dim = len(op_schema.args_schema) > 2 and bool(op_schema.args_schema[2])
    reduction_op = LINEAR_REDUCTION_OP_MAP[op_schema.op]
    return common_reduction_strategy(
        input_strategy,
        reduce_dims,
        keep_dim=keep_dim,
        reduction_linear=True,
        reduction_op=reduction_op,
    )


@register_op_strategy(aten.cumsum.default, schema_info=RuntimeSchemaInfo(1))
def cumsum_strategy(op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dim = args_schema[1]
    assert isinstance(dim, int), f"{dim}"

    return common_reduction_strategy(
        input_strategy, [dim], keep_dim=True, reduction_linear=False
    )


@register_op_strategy(
    [aten.var.correction, aten.var.correction_out],
    schema_info=RuntimeSchemaInfo(1, ["keepdim"]),
)
def var_reduction_strategy(op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.ndim)

    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims

    keep_dim = cast(bool, op_schema.kwargs_schema.get("keepdim", False))
    return common_reduction_strategy(
        input_strategy, reduce_dims, keep_dim=keep_dim, reduction_linear=False
    )


@register_op_strategy(
    [aten.linalg_vector_norm.default], schema_info=RuntimeSchemaInfo(1)
)
def vector_norm_strategy(op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)

    norm_type = args_schema[1] if len(args_schema) > 1 else 2
    assert isinstance(norm_type, (int, float, str)), f"{norm_type}"
    dim = args_schema[2] if len(args_schema) > 2 else None
    keepdim = args_schema[3] if len(args_schema) > 3 else False
    dims = _infer_reduction_dims(dim, input_strategy.ndim)
    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims
    return common_reduction_strategy(
        input_strategy,
        reduce_dims,
        keep_dim=cast(bool, keepdim),
        reduction_linear=True,
        reduction_op=NormReduction(norm_type),
    )


@register_op_strategy(
    [aten._foreach_norm.Scalar], schema_info=RuntimeSchemaInfo(1, needs_pytree=True)
)
def foreach_norm_strategy(op_schema: OpSchema) -> TupleStrategy:
    args_schema = op_schema.args_schema
    input_tuple_strategy = args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy)
    norm_type = args_schema[1] if len(args_schema) > 1 else 2
    assert isinstance(norm_type, (int, float, str)), f"{norm_type}"
    output_tuple_strategy_children: list[OpStrategy] = []
    for op_strategy in input_tuple_strategy.children:
        assert isinstance(op_strategy, OpStrategy), f"{op_strategy}"
        reduce_dims = list(range(op_strategy.ndim))
        output_strategy = common_reduction_strategy(
            op_strategy,
            reduce_dims,
            reduction_linear=True,
            reduction_op=NormReduction(norm_type),
        )
        output_tuple_strategy_children.append(output_strategy)
    return TupleStrategy(output_tuple_strategy_children)


@register_op_strategy(
    [
        aten._linalg_svd.default,
        aten.linalg_qr.default,
        # TODO: The diagonal ops can have an improved sharding strategy for
        # shard placements that does not require redistributing to replicate.
        aten.diagonal_copy.default,
        aten.diag_embed.default,
        aten.diag.default,
        aten.diagonal.default,
        aten.tril.default,
        aten.triu.default,
        aten._linalg_eigh.default,
        aten.upsample_bicubic2d.default,
        aten.upsample_bilinear2d.default,
        aten.upsample_linear1d.default,
        aten.upsample_nearest2d.default,
        aten.upsample_trilinear3d.default,
        # TODO: support the full F.interpolate set of options.
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def linalg_replicate_strategy(op_schema: OpSchema) -> OpStrategy:
    """
    Since we do not have a simple way to compute some linear algebra operations
    like SVD or QR decomposition, always fall back to replicate.
    """
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy), f"{input_strategy}"
    mesh = input_strategy.mesh

    output_strategies: list[OpSpec] = []
    for placement_strategy in input_strategy.strategies:
        replicate_placements = tuple(Replicate() for _ in range(mesh.ndim))
        replicate_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_placements,
            tensor_meta=placement_strategy.output_spec.tensor_meta,
        )
        redistribute_cost = [
            generate_redistribute_costs(input_strategy, replicate_spec)
        ]
        replicate_strategy = OpSpec(
            output_specs=replicate_spec,
            input_specs=(replicate_spec,),
            redistribute_cost=redistribute_cost,
        )
        output_strategies.append(replicate_strategy)
    return OpStrategy(output_strategies)


@register_op_strategy(
    [aten._log_softmax.default, aten._softmax.default, aten._safe_softmax.default],
    schema_info=RuntimeSchemaInfo(1),
)
def softmax_strategy(op_schema: OpSchema) -> OpStrategy:
    input_strategy, softmax_dim, *_ = op_schema.args_schema
    input_strategy = cast(OpStrategy, input_strategy)

    softmax_dim = cast(int, softmax_dim)
    softmax_dim = normalize_dim(softmax_dim, input_strategy.ndim)

    output_strategy = OpStrategy([])
    for input_placement_strategy in input_strategy.strategies:
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec

        # make sure input is replicated along the softmax dim
        input_target_spec = DTensorSpec(
            mesh=input_strategy.mesh,
            placements=replicate_reduction_dims(
                input_src_spec.placements, [softmax_dim]
            ),
            tensor_meta=input_src_spec.tensor_meta,
        )
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )
        output_target_spec = input_target_spec
        output_strategy.strategies.append(
            OpSpec(
                output_specs=output_target_spec,
                input_specs=[input_target_spec],
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_op_strategy(
    [
        aten._log_softmax_backward_data.default,
        aten._softmax_backward_data.default,
    ],
    schema_info=RuntimeSchemaInfo(2),
)
def softmax_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    grad_out_strategy, out_strategy, softmax_dim, _ = op_schema.args_schema
    grad_out_strategy = cast(OpStrategy, grad_out_strategy)
    out_strategy = cast(OpStrategy, out_strategy)
    softmax_dim = cast(int, softmax_dim)
    softmax_dim = normalize_dim(softmax_dim, grad_out_strategy.ndim)

    grad_in_strategy = OpStrategy([])
    for grad_out_placement_strat, out_placement_strat in zip(
        grad_out_strategy.strategies, out_strategy.strategies
    ):
        # follow the sharding of the grad_out or out depending on which has more shards
        grad_out_src_spec = grad_out_placement_strat.output_spec
        out_src_spec = out_placement_strat.output_spec
        src_spec = (
            grad_out_src_spec
            if grad_out_src_spec.num_shards >= out_src_spec.num_shards
            else out_src_spec
        )

        # make sure inputs are replicated along the softmax dim
        tgt_spec = DTensorSpec(
            mesh=grad_out_strategy.mesh,
            placements=replicate_reduction_dims(src_spec.placements, [softmax_dim]),
        )
        new_grad_out_spec = DTensorSpec(
            mesh=tgt_spec.mesh,
            placements=tgt_spec.placements,
            tensor_meta=grad_out_src_spec.tensor_meta,
        )
        new_out_spec = DTensorSpec(
            mesh=tgt_spec.mesh,
            placements=tgt_spec.placements,
            tensor_meta=out_src_spec.tensor_meta,
        )
        redist_grad_out_cost = generate_redistribute_costs(grad_out_strategy, tgt_spec)
        redist_out_cost = generate_redistribute_costs(out_strategy, tgt_spec)
        grad_in_strategy.strategies.append(
            OpSpec(
                output_specs=tgt_spec,
                input_specs=(new_grad_out_spec, new_out_spec),
                redistribute_cost=[redist_grad_out_cost, redist_out_cost],
            )
        )

    return grad_in_strategy


@register_op_strategy(
    [aten.nll_loss_forward.default, aten.nll_loss2d_forward.default],
    schema_info=RuntimeSchemaInfo(3),
)
def nll_loss_forward_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()

    assert len(op_schema.args_schema) == 5

    (
        input_strategy,
        target_strategy,
        weight_strategy,
        reduction,
        _,
    ) = op_schema.args_schema
    input_strategy = cast(OpStrategy, input_strategy)
    target_strategy = cast(OpStrategy, target_strategy)
    reduction = cast(int, reduction)

    input_shape = input_strategy.shape
    channel_dim = 1 if len(input_shape) >= 2 else 0

    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []

        # make sure input is replicated along the channel dim
        input_src_spec = input_placement_strategy.output_spec
        input_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(
                input_src_spec.placements, [channel_dim]
            ),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_expected_spec)
        )

        # target doesn't have channel dim, and it follows input on other dims
        target_src_spec = target_strategy.strategies[idx].output_spec
        target_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=_skip_dim(input_expected_spec.placements, channel_dim),
            tensor_meta=target_src_spec.tensor_meta,
        )
        op_args_target_specs.append(target_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(target_strategy, target_expected_spec)
        )

        # weight tensor, if given, has to be a Tensor of size input_shape[channel_dim]
        # make sure it is replicated
        if weight_strategy is not None:
            assert isinstance(weight_strategy, OpStrategy)
            weight_src_spec = weight_strategy.strategies[idx].output_spec
            weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_expected_spec)
            redistribute_costs.append(
                generate_redistribute_costs(weight_strategy, weight_expected_spec)
            )

        if reduction == Reduction.NONE.value:
            output_expected_spec = target_expected_spec
            total_weight_expected_spec = DTensorSpec(
                mesh=mesh, placements=tuple([Replicate()] * mesh.ndim)
            )
        else:
            if reduction == Reduction.MEAN.value:
                reduction_op = "avg"
                if not is_tensor_evenly_shardable(
                    target_expected_spec.shape, target_expected_spec
                ):
                    raise ValueError(
                        "The intermediate results of nll_loss cannot be evenly sharded, \
                        resulting in biased mean result."
                    )
            else:  # reduction == Reduction.SUM.value:
                reduction_op = "sum"
            reduce_dims = list(range(target_expected_spec.ndim))
            reduce_dims_map = _infer_reduce_dims_map(
                reduce_dims, target_expected_spec.ndim, keep_dim=False
            )
            out_placements = map_placements_after_reduction(
                target_expected_spec.placements,
                reduce_dims,
                reduce_dims_map,
                reduction_op,
            )
            output_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=out_placements,
            )

            # whether reduction is sum or mean, the total weight has to be summed up if not replicated
            total_weight_placements = map_placements_after_reduction(
                target_expected_spec.placements,
                reduce_dims,
                reduce_dims_map,
                "sum",
            )
            total_weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=total_weight_placements,
            )

        output_strategy.strategies.append(
            OpSpec(
                output_specs=(output_expected_spec, total_weight_expected_spec),
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_op_strategy(
    [aten.nll_loss_backward.default, aten.nll_loss2d_backward.default],
    schema_info=RuntimeSchemaInfo(4),
)
def nll_loss_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)

    assert len(op_schema.args_schema) == 7
    (
        grad_out_strategy,
        input_strategy,
        target_strategy,
        weight_strategy,
        reduction,
        _,
        total_weight_strategy,
    ) = op_schema.args_schema
    grad_out_strategy = cast(OpStrategy, grad_out_strategy)
    input_strategy = cast(OpStrategy, input_strategy)
    target_strategy = cast(OpStrategy, target_strategy)
    reduction = cast(int, reduction)
    total_weight_strategy = cast(OpStrategy, total_weight_strategy)

    input_shape = input_strategy.shape
    channel_dim = 1 if len(input_shape) >= 2 else 0

    grad_in_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []

        # make sure input is replicated along the channel dim
        input_src_spec = input_placement_strategy.output_spec
        input_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(
                input_src_spec.placements, [channel_dim]
            ),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_expected_spec)
        )

        # target doesn't have channel dim, and it follows input on other dims
        target_src_spec = target_strategy.strategies[idx].output_spec
        target_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=_skip_dim(input_expected_spec.placements, channel_dim),
            tensor_meta=target_src_spec.tensor_meta,
        )
        op_args_target_specs.append(target_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(target_strategy, target_expected_spec)
        )

        # grad_out follows target if there is no reduction;
        # otherwise, it should be a replicated scalar.
        grad_out_src_spec = grad_out_strategy.strategies[idx].output_spec
        if reduction == Reduction.NONE.value:
            grad_out_expected_spec = target_expected_spec
        else:
            grad_out_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(grad_out_src_spec.placements),
                tensor_meta=grad_out_src_spec.tensor_meta,
            )
        op_args_target_specs.insert(0, grad_out_expected_spec)
        redistribute_costs.insert(
            0, generate_redistribute_costs(grad_out_strategy, grad_out_expected_spec)
        )

        # weight tensor, if given, has to be a Tensor of size input_shape[channel_dim]
        # make sure it is replicated
        if weight_strategy is not None:
            assert isinstance(weight_strategy, OpStrategy)
            weight_src_spec = weight_strategy.strategies[idx].output_spec
            weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_expected_spec)
            redistribute_costs.append(
                generate_redistribute_costs(weight_strategy, weight_expected_spec)
            )

        # total_weight should always be replicated
        total_weight_src_spec = total_weight_strategy.strategies[idx].output_spec
        total_weight_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(total_weight_src_spec.placements),
            tensor_meta=total_weight_src_spec.tensor_meta,
        )
        op_args_target_specs.append(total_weight_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(
                total_weight_strategy, total_weight_expected_spec
            )
        )

        grad_in_expected_spec = input_expected_spec
        grad_in_strategy.strategies.append(
            OpSpec(
                output_specs=grad_in_expected_spec,
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return grad_in_strategy


@register_op_strategy(
    [aten.native_layer_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
def layer_norm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()

    # args must be: input, normalized_shape, weight, bias, eps
    # for None weight and bias, their corresponding objects will
    # be None as well. layer_norm_strategy returns one OpStrategy
    # for the triple return values (out, mean, rstd).
    assert len(op_schema.args_schema) == 5
    (
        input_strategy,
        normalized_shape,
        weight_strategy,
        bias_strategy,
        _,
    ) = op_schema.args_schema

    # the current layer norm implementation requires that all
    # input DTensor's sharding must be in form of OpStrategy
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)

    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)

    # we use OpStrategy because the output (out, mean, rstd)
    # should have the same placements
    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec

        # for the input tensor, we replicate it on the inner dims if necessary
        # TODO: we can avoid forcing the redistribution once we figure out
        # how to decompose layer norm
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )

        if weight_strategy is not None:
            assert isinstance(weight_strategy, OpStrategy)
            weight_src_spec = weight_strategy.strategies[idx].output_spec

            # for the weight tensor, we replicate it on all dims if necessary
            # TODO: we can avoid forcing the redistribution once we figure out
            # how to decompose layer norm
            weight_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(weight_strategy, weight_target_spec)
            )

        if bias_strategy is not None:
            assert isinstance(bias_strategy, OpStrategy)
            bias_src_spec = bias_strategy.strategies[idx].output_spec

            # for the bias tensor, we replicate it on all dims if necessary
            # TODO: we can avoid forcing the redistribution once we figure out
            # how to decompose layer norm
            bias_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(bias_src_spec.placements),
                tensor_meta=bias_src_spec.tensor_meta,
            )
            op_args_target_specs.append(bias_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(bias_strategy, bias_target_spec)
            )

        # the output spec is the same as input spec
        output_target_spec = input_target_spec
        output_strategy.strategies.append(
            OpSpec(
                output_specs=output_target_spec,
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


def _common_norm_backward_strategy(
    op_schema: OpSchema,
    rms_norm: bool = False,
) -> OpStrategy:
    """Common backward strategy logic for layer_norm and rms_norm."""
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)

    if not rms_norm:
        # layer_norm args: grad_out, input, normalized_shape, mean, rstd,
        # weight, bias, output_mask. For None weight and bias, their
        # corresponding objects will be None as well.
        assert len(op_schema.args_schema) == 8
        (
            grad_out_strategy,
            input_strategy,
            normalized_shape,
            mean_strategy,
            rstd_strategy,
            weight_strategy,
            bias_strategy,
            output_mask,
        ) = op_schema.args_schema
    else:
        # rms_norm args: grad_out, input, normalized_shape, rstd,
        assert len(op_schema.args_schema) == 6
        (
            grad_out_strategy,
            input_strategy,
            normalized_shape,
            rstd_strategy,
            weight_strategy,
            output_mask,
        ) = op_schema.args_schema
        mean_strategy = None
        bias_strategy = None

    assert isinstance(grad_out_strategy, OpStrategy)
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(rstd_strategy, OpStrategy)
    if mean_strategy is not None:
        assert isinstance(mean_strategy, OpStrategy)

    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)
    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)
    outer_dims = list(range(axis))

    if not rms_norm:
        assert isinstance(output_mask, list) and len(output_mask) == 3
    else:
        assert isinstance(output_mask, list) and len(output_mask) == 2

    # output tuple: (d_input, d_weight[, d_bias])
    out_tuple_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        # args for OpSpec
        output_specs_list: list[Optional[DTensorSpec]] = []
        input_specs_list: list[DTensorSpec] = []
        redistribute_costs = []

        input_src_spec = input_placement_strategy.output_spec
        # arg: grad_out
        # TODO: change the strategy to the following rule.
        # d_input is basically a product of element-wise mul of
        # grad_out, rstd, and normalized input, among which rstd
        # and normalized input (x_hat) should have the same sharding
        # placements, and grad_out's sharding is determined by the
        # pointwise result of x_hat and weight/bias.
        # TODO: now grad_out spec follows input spec. we may need
        # to change it to apply a pointwise rule over grad_out,
        # input, and weight.
        grad_out_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        input_specs_list.append(grad_out_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(grad_out_strategy, grad_out_target_spec)
        )
        output_specs_list.append(grad_out_target_spec if output_mask[0] else None)

        # arg: input
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        input_specs_list.append(input_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )

        # arg: mean
        if not rms_norm:
            assert mean_strategy is not None  # mypy fix
            mean_src_spec = mean_strategy.strategies[idx].output_spec
            input_specs_list.append(mean_src_spec)
            redistribute_costs.append([0.0 for _ in mean_strategy.strategies])

        # arg: rstd
        rstd_src_spec = rstd_strategy.strategies[idx].output_spec
        input_specs_list.append(rstd_src_spec)
        redistribute_costs.append([0.0 for _ in rstd_strategy.strategies])

        def _add_target_input_spec(strategy) -> DTensorSpec:
            # shared logic for setting the weight and bias target input specs
            assert isinstance(strategy, OpStrategy)
            src_spec = strategy.strategies[idx].output_spec
            # no need to redistribute since they should be replicated in forward pass
            input_specs_list.append(src_spec)
            redistribute_costs.append([0.0 for _ in strategy.strategies])
            return src_spec

        # arg: weight
        # d_weight = sum(grad_out * (input - mean) / rstd, outer_dim, keepdim=False)
        # For RMS norm, mean is 0, so it's just: sum(grad_out * input / rstd, outer_dim, keepdim=False)
        if weight_strategy is not None:
            weight_src_spec = _add_target_input_spec(weight_strategy)
            # TODO: now d_weight spec follows input spec w/ a reduction.
            # we may need to change to a pointwise rule over grad_out and
            # input, then apply a reduction.
            inp_placements = _replicate_dims_start_at(input_src_spec.placements, axis)
            reduce_dims_map = _infer_reduce_dims_map(
                outer_dims, input_src_spec.ndim, False
            )
            out_placements = map_placements_after_reduction(
                inp_placements, outer_dims, reduce_dims_map, "sum"
            )
            weight_out_spec = DTensorSpec(
                mesh=mesh,
                placements=out_placements,
                tensor_meta=weight_src_spec.tensor_meta,
            )
            output_specs_list.append(weight_out_spec if output_mask[1] else None)
        else:
            if not rms_norm:
                error_msg = "output_mask[1] should not be `True` while weight argument is `None` in native_layer_norm_backward."
            else:
                error_msg = "output_mask[1] should not be `True` while weight argument is `None` in _fused_rms_norm_backward."
            assert output_mask[1] is False, error_msg
            output_specs_list.append(None)

        # arg: bias
        # d_bias = sum(grad_out, outer_dim, keepdim=False)
        if not rms_norm:
            if bias_strategy is not None:
                bias_src_spec = _add_target_input_spec(bias_strategy)
                # d_bias spec follows a reduction over grad_out
                inp_placements = _replicate_dims_start_at(
                    grad_out_target_spec.placements, axis
                )
                reduce_dims_map = _infer_reduce_dims_map(
                    outer_dims, grad_out_target_spec.ndim, False
                )
                out_placements = map_placements_after_reduction(
                    inp_placements, outer_dims, reduce_dims_map, "sum"
                )
                bias_out_spec = DTensorSpec(
                    mesh=mesh,
                    placements=out_placements,
                    tensor_meta=bias_src_spec.tensor_meta,
                )
                output_specs_list.append(bias_out_spec if output_mask[2] else None)
            else:
                assert output_mask[2] is False, (
                    "output_mask[2] should not be `True` while bias argument is `None` in native_layer_norm_backward."
                )
                output_specs_list.append(None)

        out_tuple_strategy.strategies.append(
            OpSpec(
                output_specs=tuple(output_specs_list),
                input_specs=input_specs_list,
                redistribute_cost=redistribute_costs,
            )
        )

    return out_tuple_strategy


@register_op_strategy(
    [aten.native_layer_norm_backward.default],
    schema_info=RuntimeSchemaInfo(2),
)
def layer_norm_bwd_strategy(op_schema: OpSchema) -> OpStrategy:
    return _common_norm_backward_strategy(op_schema)


@register_op_strategy(
    [aten._fused_rms_norm_backward.default],
    schema_info=RuntimeSchemaInfo(2),
)
def fused_rms_norm_bwd_strategy(op_schema: OpSchema) -> OpStrategy:
    return _common_norm_backward_strategy(op_schema, rms_norm=True)


@register_op_strategy(
    [aten.topk.default],
    schema_info=RuntimeSchemaInfo(2),
)
def topk_strategy(op_schema: OpSchema) -> OpStrategy:
    input_strategy = cast(OpStrategy, op_schema.args_schema[0])
    topk_dim = (
        cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else -1
    )
    topk_dim = normalize_dim(topk_dim, input_strategy.ndim)

    single_mesh_dim_strategies = []

    # two outputs (values, indices), 1 input
    # replicate always works
    all_replicate: PlacementList = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # every dim except topk dim should work
    for dim in range(input_strategy.ndim):
        if dim != topk_dim:
            dim_shardings: PlacementList = [Shard(dim)] * 3
            single_mesh_dim_strategies.append(dim_shardings)
    # TODO: topk on sharded dim requires non-trival reduction, address it later

    return expand_to_full_mesh_op_strategy(
        input_strategy.mesh, op_schema, single_mesh_dim_strategies, input_index=2
    )


@register_op_strategy(
    [aten.histc.default],
    # strategy choice depends on the value of 'min' and 'max' kwargs, which are position 2 and 3
    schema_info=RuntimeSchemaInfo(2),
)
def histc_strategy(op_schema: OpSchema) -> OpStrategy:
    input_strategy = cast(OpStrategy, op_schema.args_schema[0])
    single_mesh_dim_strategies: list[PlacementList] = []
    single_mesh_dim_strategies.append([Replicate(), Replicate()])

    # histc can support sharded input and partial output on any input dim, provided the min and max
    # values are user-specified.  If not user-specified, the true min and max of the data in each local
    # tensor will be used to compute bin boundaries, which will not be the same across ranks, leading to
    # an incorrect final result
    if len(op_schema.args_schema) == 4:
        for dim in range(input_strategy.ndim):
            dim_shardings: PlacementList = [Partial(), Shard(dim)]
            single_mesh_dim_strategies.append(dim_shardings)

    return expand_to_full_mesh_op_strategy(
        input_strategy.mesh, op_schema, single_mesh_dim_strategies
    )
