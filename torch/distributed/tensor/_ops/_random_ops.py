# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.distributed.tensor._random as random
from torch._library.utils import fill_defaults
from torch._ops import OpOverload
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._random import is_rng_supported_mesh
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Placement


aten = torch.ops.aten


def _contiguous_stride(shape: torch.Size) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _flat_start_for_dtensor(tensor: DTensor) -> int:
    local_shape, global_offset = compute_local_shape_and_global_offset(
        tensor.shape,
        tensor.device_mesh,
        tensor.placements,
    )
    if tuple(local_shape) != tuple(tensor._local_tensor.shape):
        raise RuntimeError(
            f"Local shape mismatch for {tensor.shape}: metadata={local_shape}, "
            f"actual={tuple(tensor._local_tensor.shape)}"
        )
    if len(tensor.shape) > 1 and (
        any(global_offset[dim] != 0 for dim in range(1, len(tensor.shape)))
        or any(
            local_shape[dim] != tensor.shape[dim] for dim in range(1, len(tensor.shape))
        )
    ):
        raise NotImplementedError(
            "DTensor dense-equivalent RNG initialization only supports local shards "
            "that are contiguous in the dense flattened tensor"
        )
    return sum(
        offset * stride
        for offset, stride in zip(global_offset, _contiguous_stride(tensor.shape))
    )


def _dense_distribution_offset_increment(
    numel: int, dtype: torch.dtype, device: torch.device
) -> int:
    if numel == 0:
        return 0
    block_size = 256
    unroll_factor = 2 if dtype == torch.float64 else 4
    props = torch.cuda.get_device_properties(device)
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    return ((numel - 1) // (block_size * grid * unroll_factor) + 1) * 4


def _rng_key_from_state(
    state: random._PhiloxState, device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [state.seed.item(), state.offset.item()],
        dtype=torch.uint64,
        device=device,
    )


def _sync_rng_state_from_mesh_root(tensor: DTensor, state: torch.Tensor) -> None:
    if tensor.device_mesh.ndim != 1:
        return
    src_rank = int(tensor.device_mesh.mesh.flatten()[0].item())
    torch.distributed.broadcast(
        state, src=src_rank, group=tensor.device_mesh.get_group()
    )


def _run_dtensor_local_rng_op(
    tensor: DTensor,
    generator: torch.Generator | None,
    dense_equivalent_op_call: torch._ops.OpOverload,
    fallback_op_call: torch._ops.OpOverload,
    *op_args: object,
) -> DTensor:
    if tensor.device_mesh.get_coordinate() is None:
        return tensor

    local_tensor = tensor._local_tensor
    if local_tensor.is_meta:
        return tensor
    if local_tensor.device.type != "cuda":
        fallback_op_call(local_tensor, *op_args, generator=generator)
        return tensor
    if not local_tensor.is_contiguous():
        raise RuntimeError(
            f"{dense_equivalent_op_call}: expected a contiguous local shard, "
            f"got stride {local_tensor.stride()}"
        )

    if generator is None:
        if not random._rng_tracker and is_rng_supported_mesh(tensor.device_mesh):
            random._rng_tracker = random.OffsetBasedRNGTracker(tensor.device_mesh)
        tracker = random._rng_tracker
        if not isinstance(tracker, random.OffsetBasedRNGTracker):
            raise AssertionError
        device_state = tracker._get_device_state()
        _sync_rng_state_from_mesh_root(tensor, device_state)
        state = random._PhiloxState(device_state)
    else:
        state = random._PhiloxState(generator.get_state())

    key = _rng_key_from_state(state, local_tensor.device)
    global_numel = tensor.numel()
    shard_offset = _flat_start_for_dtensor(tensor)
    dense_equivalent_op_call(local_tensor, key, global_numel, shard_offset, *op_args)
    state.offset = state.offset + _dense_distribution_offset_increment(
        global_numel, local_tensor.dtype, local_tensor.device
    )

    if generator is None:
        tracker = random._rng_tracker
        if not isinstance(tracker, random.OffsetBasedRNGTracker):
            raise AssertionError
        tracker._set_device_state(state.state)
    else:
        generator.set_state(state.state)
    return tensor


def _normal_dtensor_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> DTensor:
    args, kwargs = fill_defaults(op_call._schema, args, kwargs)
    tensor, mean, std = args
    generator = kwargs.pop("generator")
    if kwargs:
        raise AssertionError
    if not isinstance(tensor, DTensor):
        raise AssertionError
    if not (generator is None or isinstance(generator, torch.Generator)):
        raise AssertionError
    return _run_dtensor_local_rng_op(
        tensor,
        generator,
        aten._dtensor_local_normal_.default,
        op_call,
        mean,
        std,
    )


def _uniform_dtensor_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> DTensor:
    args, kwargs = fill_defaults(op_call._schema, args, kwargs)
    tensor, low, high = args
    generator = kwargs.pop("generator")
    if kwargs:
        raise AssertionError
    if not isinstance(tensor, DTensor):
        raise AssertionError
    if not (generator is None or isinstance(generator, torch.Generator)):
        raise AssertionError
    return _run_dtensor_local_rng_op(
        tensor,
        generator,
        aten._dtensor_local_uniform_.default,
        op_call,
        low,
        high,
    )


DTensor._op_dispatcher._custom_op_handlers[aten.normal_.default] = (
    _normal_dtensor_handler
)
DTensor._op_dispatcher._custom_op_handlers[aten.uniform_.default] = (
    _uniform_dtensor_handler
)


def _random_inplace_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim strategy for in-place random ops (single tensor input, output follows input).

    No Partial inputs: random sampling on partial tensors is undefined.
    """
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError
    num_outputs = sum(1 for r in op._schema.returns if "Tensor" in str(r.type))
    placements: list[list[Placement | _ShardingPlaceholder]] = []
    for i in range(len(self_meta.shape)):
        rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(i)] * (
            num_outputs + 1
        )
        placements.append(rule)
    return placements


# In-place random sampling ops: output follows input sharding exactly.
_inplace_random_ops = [
    aten.normal_.default,
    aten.uniform_.default,
    aten.native_dropout.default,
    aten.bernoulli_.float,
    aten.bernoulli.default,
    aten.log_normal_.default,
    aten.exponential_.default,
    aten.geometric_.default,
]

for _op in _inplace_random_ops:
    register_single_dim_strategy(_op, allow_uneven_sharding=True)(
        _random_inplace_single_dim_strategy
    )


@register_single_dim_strategy(aten.multinomial.default)
def multinomial_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim strategy for multinomial.

    multinomial(self, num_samples, ...) -> Tensor
    Input: [*, n_categories], Output: [*, num_samples] (dtype=long)

    Only batch dims (all except the last) can be sharded — the last dim
    (categories) is consumed by the sampling and maps to a different
    semantic dim (num_samples) in the output.
    """
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError
    placements: list[list[Placement | _ShardingPlaceholder]] = []
    for i in range(len(self_meta.shape) - 1):
        rule: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(i),
            _ShardingPlaceholder(i),
        ]
        placements.append(rule)
    return placements
