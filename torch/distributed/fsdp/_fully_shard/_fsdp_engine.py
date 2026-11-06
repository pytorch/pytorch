# mypy: allow-untyped-defs
"""The FlexShard engine bucket behind each ``fully_shard`` parameter group.

A group keeps its parameters' padded shards in one engine bucket of bytes
(``torch.distributed.fsdp._flex_shard``) and all-gathers through the bucket's
placement, FlexShard's ``Shard``, whose collective copies are FSDP's native
copy ops. The engine's FSDP-owned persistent buffers are the parameters'
all-gather outputs, and the reduce-scatter copy-in is the placement's.

``FSDPParamGroup`` keeps FSDP2's scheduling (training-state machine, prefetch,
post-backward bookkeeping, gradient reduction and accumulation) and its DTensor
parameters: ``FSDPParam`` is the representation adapter over the engine
storage. Groups the bucket can't express yet (all-gather extensions, tensor
subclass parameters, custom all-gather layouts) and ``TORCH_FSDP_ENGINE=legacy``
use FSDP2's per-parameter storage and multi-input all-gather instead.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from .._flex_shard.bucket_storage import ParamInfo
from .._flex_shard.placements.shard import Shard as EngineShard
from ._all_gather_layout import DefaultAllGatherLayout
from ._fsdp_collectives import _wait_all_gather, AllGatherResult, DefaultAllGather
from ._fsdp_param import FSDPParam, ShardedState


if TYPE_CHECKING:
    from ._fsdp_param_group import FSDPParamGroup


__all__: list[str] = []

logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

ENGINE_ENV_VAR = "TORCH_FSDP_ENGINE"


def engine_enabled() -> bool:
    """Whether ``fully_shard`` groups use the FlexShard engine bucket, the
    default. ``TORCH_FSDP_ENGINE=legacy`` restores FSDP2's per-parameter
    storage and collectives."""
    return os.environ.get(ENGINE_ENV_VAR, "flex").lower() != "legacy"


@functools.cache
def _warn_legacy_once() -> None:
    logger.warning(
        "%s=legacy: fully_shard uses its pre-FlexShard-engine storage and "
        "collectives. This escape hatch is deprecated.",
        ENGINE_ENV_VAR,
    )


class _GroupMesh:
    """The 1-D view of a process group that engine placements use as a mesh."""

    ndim = 1

    def __init__(self, group: dist.ProcessGroup, device_type: str) -> None:
        self._group = group
        self.device_type = device_type

    def get_group(self, mesh_dim: Any = None) -> dist.ProcessGroup:
        return self._group

    def size(self, mesh_dim: Any = None) -> int:
        return self._group.size()

    def get_local_rank(self, mesh_dim: Any = None) -> int:
        return self._group.rank()


def static_general_path_reason(
    group: FSDPParamGroup, engine_requested: bool
) -> str | None:
    """Why ``group``'s parameters can't live in an engine bucket, else None.
    ``engine_requested`` is :func:`engine_enabled` when the group was built."""
    if not engine_requested:
        _warn_legacy_once()
        return f"{ENGINE_ENV_VAR}=legacy"
    if not group.fsdp_params:
        return "no parameters"
    for fsdp_param in group.fsdp_params:
        if hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather"):
            return "all-gather extensions"
        if type(fsdp_param._sharded_param_data) is not torch.Tensor:
            return "tensor subclass parameters"
    return None


@functools.cache
def _native_copy_output_fn() -> Any:
    from ..experimental import all_gather_output_fn_with_native_copy

    return all_gather_output_fn_with_native_copy


def dynamic_general_path_reason(group: FSDPParamGroup) -> str | None:
    """Why ``group`` can't all-gather through its bucket right now, else None."""
    layout = group._all_gather_layout
    if type(layout) is not DefaultAllGatherLayout or layout.output_fn not in (
        None,
        _native_copy_output_fn(),
    ):
        # The bucket placement's copy-out is the native one
        return "custom all-gather layouts"
    return None


def _aligned_offsets(tensors: list[torch.Tensor]) -> list[int]:
    """Byte offsets of ``tensors`` packed in one buffer, each aligned to its
    element size so that it can be viewed in its dtype."""
    offsets = []
    offset = 0
    for tensor in tensors:
        itemsize = tensor.element_size()
        offset = -(-offset // itemsize) * itemsize
        offsets.append(offset)
        offset += tensor.numel() * itemsize
    return offsets


class FSDPEngineBucket:
    """One parameter group's engine bucket: its storage, the parameters'
    engine infos, and the all-gather in flight."""

    def __init__(self, fsdp_params: list[FSDPParam]) -> None:
        self.storage: torch.Tensor | None = None
        self.views: list[torch.Tensor] = []
        self.infos: list[ParamInfo] = []
        self.info_by_param: dict[FSDPParam, ParamInfo] = {}
        self.prepared: Any = None
        self.post_forward_infos: list[ParamInfo] = []
        self.home(fsdp_params)

    # Storage #
    def homed(self, fsdp_params: list[FSDPParam]) -> bool:
        # FSDPParam.reset_sharded_param rebinds _sharded_param_data when the
        # sharded parameter changed (to_empty, load_state_dict(assign=True))
        return self.storage is not None and all(
            fsdp_param._sharded_param_data is view
            for fsdp_param, view in zip(fsdp_params, self.views)
        )

    def home(self, fsdp_params: list[FSDPParam]) -> None:
        """Move each parameter's padded sharded data into the bucket and point
        its ``_sharded_param_data`` and sharded parameter at it."""
        datas = [fsdp_param._sharded_param_data for fsdp_param in fsdp_params]
        devices = {data.device for data in datas}
        if len(devices) != 1:
            raise AssertionError(f"Expected one device, got {devices}")
        (device,) = devices
        nbytes = [data.numel() * data.element_size() for data in datas]
        offsets = _aligned_offsets(datas)
        total = offsets[-1] + nbytes[-1]
        storage = self.storage
        if storage is None or storage.numel() != total or storage.device != device:
            pin_memory = device.type == "cpu" and any(
                fsdp_param.pin_memory for fsdp_param in fsdp_params
            )
            storage = torch.empty(
                total, dtype=torch.uint8, device=device, pin_memory=pin_memory
            )
        views = []
        for fsdp_param, data, size, offset in zip(fsdp_params, datas, nbytes, offsets):
            view = storage[offset : offset + size].view(data.dtype)
            if not data.is_meta and (
                data.numel()
                and data.untyped_storage().data_ptr()
                != storage.untyped_storage().data_ptr()
            ):
                view.copy_(data)
            fsdp_param._sharded_param_data = view
            views.append(view)
            shard_dim = fsdp_param.fsdp_placement.dim
            sharded_param = fsdp_param.sharded_param
            local = sharded_param._local_tensor
            length = local.size(shard_dim) if local.numel() > 0 else 0
            sharded_param._local_tensor = view.view(
                fsdp_param.padded_sharded_param_size
            ).narrow(shard_dim, 0, length)
        self.storage = storage
        self.views = views
        self.build_infos(fsdp_params)

    def build_infos(self, fsdp_params: list[FSDPParam]) -> None:
        infos = []
        datas = [fsdp_param._sharded_param_data for fsdp_param in fsdp_params]
        for index, (fsdp_param, data, offset) in enumerate(
            zip(fsdp_params, datas, _aligned_offsets(datas))
        ):
            fqn = (
                fsdp_param._param_fqn or f"{index}:{fsdp_param._module_info.param_name}"
            )
            infos.append(
                ParamInfo(
                    fqn=fqn,
                    global_shape=fsdp_param._orig_size,
                    global_stride=fsdp_param._contiguous_orig_stride,
                    dtype=data.dtype,
                    requires_grad=fsdp_param.sharded_param.requires_grad,
                    placements=(EngineShard(fsdp_param.fsdp_placement.dim),),
                    # Mixed precision dtypes are resolved in lazy init
                    param_dtype=getattr(fsdp_param, "param_dtype", None),
                    reduce_dtype=getattr(fsdp_param, "reduce_dtype", None),
                    local_shape=fsdp_param.sharded_size,
                    local_numel=fsdp_param.sharded_size.numel(),
                    byte_offset=offset,
                    storage_nbytes=data.numel() * data.element_size(),
                    global_numel=fsdp_param._orig_size.numel(),
                )
            )
        self.infos = infos
        self.info_by_param = dict(zip(fsdp_params, infos))

    # Runtime #
    def all_gather(self, group: FSDPParamGroup, async_op: bool) -> AllGatherResult:
        copy_in_stream, all_gather_stream = group.comm_ctx.get_all_gather_streams(
            async_op, group._training_state
        )
        device_handle = group.device_handle
        if group.is_sharded_post_forward:
            tensors = [p._sharded_post_forward_param_data for p in group.fsdp_params]
            infos = self.post_forward_infos
        else:
            if not self.homed(group.fsdp_params):
                # Like FSDP2, the copy-in otherwise relies on the root's
                # pre-forward and pre-backward stream syncs, without waiting
                # on the compute it overlaps
                self.home(group.fsdp_params)
                copy_in_stream.wait_stream(device_handle.current_stream())
            tensors = [p._sharded_param_data for p in group.fsdp_params]
            infos = self.infos
        placement = infos[0].placement
        with device_handle.stream(copy_in_stream):
            comm = group._all_gather_comm
            prepared = placement._prepare_unshard_bucket(
                tensors,
                infos,
                _GroupMesh(group._all_gather_process_group, group.device.type),
                group._module_fqn,
                skip_single_rank_all_gather=True,
                comm=None if type(comm) is DefaultAllGather else comm,
                # CPU offload: the placement copies the bucket into the device's
                # gather buffer
                device=group.device,
            )
            copy_in_event = None
            if all_gather_stream is not copy_in_stream:
                copy_in_event = device_handle.Event()
                copy_in_event.record()
        with device_handle.stream(all_gather_stream):
            if copy_in_event is not None:
                all_gather_stream.wait_event(copy_in_event)
            work = placement.run_prepared_unshard(prepared, async_op=async_op)
            event = None
            if not isinstance(work, dist.distributed_c10d.Work):
                work = None
                event = device_handle.Event()
                event.record()
        self.prepared = prepared
        return AllGatherResult(
            all_gather_output=prepared.buffers[-1],
            all_gather_event=event,
            all_gather_work=work,
            param_all_gather_input_dtypes=[],
            param_all_gather_input_numels=[],
            all_gather_input_split_sizes=[],
            all_gather_input_outer_sizes=[],
        )

    def copy_out(self, group: FSDPParamGroup) -> None:
        """Wait for the pending all-gather and copy it into the parameters'
        all-gather outputs, the engine's persistent buffers."""
        prepared = self.prepared
        if prepared is None or group._all_gather_result is None:
            raise AssertionError("Expected a pending engine all-gather")
        _wait_all_gather(group._all_gather_result)
        world_size = group._all_gather_process_group.size()
        outputs = []
        for fsdp_param, info in zip(group.fsdp_params, self.infos):
            if not fsdp_param.all_gather_outputs:
                fsdp_param.init_all_gather_outputs(
                    [info.storage_nbytes // info.dtype.itemsize],
                    [info.unsharded_dtype],
                    world_size,
                    group.device,
                )
            else:
                fsdp_param.alloc_all_gather_outputs()
            outputs.append(fsdp_param.all_gather_outputs[0])
        prepared.persistent = True
        prepared.persistent_buffers = outputs
        preserve = (
            torch.autograd._unsafe_preserve_version_counter(tuple(outputs))
            if not any(output.is_inference() for output in outputs)
            else contextlib.nullcontext()
        )
        with preserve, torch.no_grad():
            prepared.placement.finish_prepared_unshard(prepared)
        self.prepared = None

    def to_sharded_post_forward(self, group: FSDPParamGroup) -> None:
        """Reshard to the post-forward mesh (``reshard_after_forward=int``):
        each rank keeps its chunk of every all-gather output, in one buffer so
        that the next unshard gathers them through the bucket placement."""
        for fsdp_param in group.fsdp_params:
            fsdp_param.check_post_forward_reshard()
        mesh_info = group.post_forward_mesh_info
        world_size = mesh_info.shard_mesh_size
        rank = mesh_info.shard_mesh_rank
        outputs = [p.all_gather_outputs[0] for p in group.fsdp_params]
        numels = []
        for output in outputs:
            if output.numel() % world_size != 0:
                raise AssertionError(
                    f"All-gather output size ({output.numel()}) must be divisible "
                    f"by the shard world size ({world_size})"
                )
            numels.append(output.numel() // world_size)
        chunks = [
            output.narrow(0, numel * rank, numel)
            for output, numel in zip(outputs, numels)
        ]
        offsets = _aligned_offsets(chunks)
        storage = torch.empty(
            offsets[-1] + chunks[-1].numel() * chunks[-1].element_size(),
            dtype=torch.uint8,
            device=group.device,
        )
        infos = []
        for fsdp_param, output, chunk, offset in zip(
            group.fsdp_params, outputs, chunks, offsets
        ):
            nbytes = chunk.numel() * chunk.element_size()
            data = storage[offset : offset + nbytes].view(chunk.dtype)
            data.copy_(chunk)
            _set_post_forward_param(fsdp_param, data)
            infos.append(
                ParamInfo(
                    fqn=fsdp_param._display_name,
                    global_shape=torch.Size([output.numel()]),
                    global_stride=(1,),
                    dtype=output.dtype,
                    requires_grad=fsdp_param.sharded_param.requires_grad,
                    placements=(EngineShard(0),),
                    local_shape=torch.Size([chunk.numel()]),
                    local_numel=chunk.numel(),
                    byte_offset=offset,
                    storage_nbytes=nbytes,
                    global_numel=output.numel(),
                )
            )
        self.post_forward_infos = infos


def _set_post_forward_param(fsdp_param: FSDPParam, data: torch.Tensor) -> None:
    """``FSDPParam.to_sharded_post_forward`` with the post-forward data in the
    bucket's post-forward buffer rather than its own clone."""
    fsdp_param._assert_in_states(ShardedState.UNSHARDED)
    fsdp_param._sharded_post_forward_param_data = data
    sharded_post_forward_tensor = torch.as_strided(
        data,
        size=fsdp_param.sharded_post_forward_size,
        stride=fsdp_param.contiguous_sharded_post_forward_stride,
        storage_offset=data.storage_offset(),
    )
    fsdp_param._sharded_post_forward_param = nn.Parameter(
        fsdp_param.to_sharded_post_forward_dtensor(sharded_post_forward_tensor),
        requires_grad=fsdp_param.sharded_param.requires_grad,
    )
    fsdp_param._sharded_post_forward_param.grad_dtype = fsdp_param.sharded_grad_dtype
    fsdp_param._setattr_on_modules(fsdp_param._sharded_post_forward_param)
    if not fsdp_param.keep_unsharded_storage:
        fsdp_param.free_unsharded_param()
    fsdp_param.sharded_state = ShardedState.SHARDED_POST_FORWARD
