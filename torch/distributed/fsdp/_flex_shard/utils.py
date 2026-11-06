# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
    _pop_mode,
    _push_mode,
)
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import BucketParamFQNsByIndex, ShardedBucketStorage
    from .placement_contract import Placement


def _with_fqn(label: str, fqn: str | None) -> str:
    """Append a module/bucket FQN to profiler labels, matching FSDP style."""
    if fqn:
        return f"{label} ({fqn})"
    return label


def _record_function_if_eager(
    label: str,
    fqn: str | None,
) -> AbstractContextManager[Any]:
    """Return a profiler range in eager and a no-op context during compile."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return torch.profiler.record_function(_with_fqn(label, fqn))


def _record_copy_in_if_eager() -> AbstractContextManager[Any]:
    """Return an eager profiler range for unshard copy-in."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return torch.profiler.record_function("FlexShard::copy_in")


def _record_copy_out_if_eager() -> AbstractContextManager[Any]:
    """Return an eager profiler range for unshard copy-out."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return torch.profiler.record_function("FlexShard::copy_out")


def _record_view_out_if_eager() -> AbstractContextManager[Any]:
    """Return an eager profiler range for unshard view-out."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return torch.profiler.record_function("FlexShard::view_out")


def _record_comm_if_eager(
    label: str,
    fqn: str | None,
) -> AbstractContextManager[Any]:
    """Return an eager profiler range for communication launch code."""
    return _record_function_if_eager(label, fqn)


_SAC_MODES = (_CachingTorchDispatchMode, _CachedTorchDispatchMode)
_R = TypeVar("_R")


def _outside_selective_checkpoint(fn: Callable[..., _R]) -> Callable[..., _R]:
    """Run eager ``fn`` hidden from selective activation checkpointing.

    SAC requires a checkpointed region's recompute to run the ops its forward
    ran, but whether a bucket hook there unshards, prefetches, or releases a
    prefetch depends on runtime state that differs between the two. Unshards
    run without grad and refill their buffers in place, so SAC has nothing to
    save or replay. Other dispatch modes stay active; during compile, SAC tags
    the traced graph instead, so nothing is hidden.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _R:
        if torch.compiler.is_compiling():
            return fn(*args, **kwargs)
        stack = _get_current_dispatch_mode_stack()
        first_sac = next(
            (i for i, mode in enumerate(stack) if isinstance(mode, _SAC_MODES)), None
        )
        if first_sac is None:
            return fn(*args, **kwargs)
        popped = [_pop_mode() for _ in range(len(stack) - first_sac)]
        kept = [mode for mode in reversed(popped) if not isinstance(mode, _SAC_MODES)]
        for mode in kept:
            _push_mode(mode)
        try:
            return fn(*args, **kwargs)
        finally:
            for _ in kept:
                _pop_mode()
            for mode in reversed(popped):
                _push_mode(mode)

    return wrapper


def _get_single_placement(placements: tuple[Placement, ...]) -> Placement:
    """Return the only placement supported by the minimal eager path."""
    if len(placements) != 1:
        raise ValueError(
            "FlexShard eager mode currently supports exactly one placement "
            f"per parameter, but got {len(placements)} placements."
        )
    return placements[0]


def _module_path_common_prefix(paths: list[str]) -> str:
    """Return the common module path prefix for parameter-owner module paths."""
    if not paths:
        return ""
    common_parts = paths[0].split(".") if paths[0] else []
    for path in paths[1:]:
        parts = path.split(".") if path else []
        limit = min(len(common_parts), len(parts))
        i = 0
        while i < limit and common_parts[i] == parts[i]:
            i += 1
        common_parts = common_parts[:i]
        if not common_parts:
            break
    return ".".join(common_parts)


def _strip_checkpoint_wrapped_module_path(path: str) -> str:
    """Remove CheckpointWrapper internals from a dotted module path."""
    return ".".join(
        part for part in path.split(".") if part != "_checkpoint_wrapped_module"
    )


def _top_level_owner_path(module: nn.Module, owner_path: str) -> str:
    """Return the top-level module path (one level into containers) of an owner path."""
    parts = owner_path.split(".")
    if not parts or not parts[0]:
        return ""
    child = getattr(module, parts[0])
    if (
        isinstance(child, (nn.ModuleDict, nn.ModuleList, nn.Sequential))
        and len(parts) > 1
    ):
        return ".".join(parts[:2])
    return parts[0]


def _get_bucket_storage_debug_fqn(
    bucket_storage: ShardedBucketStorage,
) -> str | None:
    """Return a concise module/bucket FQN for profiler annotations."""
    owner_paths = sorted(
        {
            _strip_checkpoint_wrapped_module_path(".".join(fqn.split(".")[:-1]))
            for fqn in bucket_storage._param_infos
        }
    )
    if not owner_paths:
        return None
    common = _module_path_common_prefix(owner_paths)
    if common:
        return common
    top_level_paths = sorted(
        {
            _top_level_owner_path(bucket_storage._module, owner_path)
            for owner_path in owner_paths
        }
    )
    top_level_paths = [path for path in top_level_paths if path]
    if not top_level_paths:
        return None
    return ", ".join(top_level_paths)


def _set_param_on_module(
    root_module: nn.Module,
    fqn: str,
    param: nn.Parameter,
) -> None:
    """Navigate to submodule by FQN and set parameter."""
    parts = fqn.split(".")
    module = root_module
    for part in parts[:-1]:
        module = getattr(module, part)
    param_name = parts[-1]
    wrapped = module._modules.get("_checkpoint_wrapped_module")
    if wrapped is not None and param_name not in module._parameters:
        module = wrapped
    if param_name in module._parameters:
        module._parameters[param_name] = param
    else:
        setattr(module, param_name, param)


def _get_managed_named_params(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    """
    Collect parameters managed by this root-level flex_shard() call.

    A parameter registered under several names (e.g. tied embedding and output
    weights) is listed once, under its first name; ``_get_shared_param_names``
    returns its other names.
    """
    managed_params: list[tuple[str, nn.Parameter]] = []
    seen_params: set[int] = set()
    for fqn, param in module.named_parameters(remove_duplicate=False):
        if id(param) not in seen_params:
            seen_params.add(id(param))
            managed_params.append((fqn, param))
    return managed_params


def _get_shared_param_names(module: nn.Module) -> dict[str, list[str]]:
    """Map each shared parameter's first name to its other names."""
    first_names: dict[int, str] = {}
    shared: dict[str, list[str]] = {}
    for fqn, param in module.named_parameters(remove_duplicate=False):
        first_name = first_names.setdefault(id(param), fqn)
        if first_name != fqn:
            shared.setdefault(first_name, []).append(fqn)
    return shared


def _validate_flex_shard_mesh(mesh: DeviceMesh) -> None:
    """Validate mesh inputs for FlexShard eager mode."""
    if mesh.ndim != 1:
        raise ValueError(
            f"flex_shard requires a 1D DeviceMesh, but got {mesh.ndim}D mesh"
        )
    if mesh.device_type != "cuda":
        raise NotImplementedError(
            "FlexShard runtime requires a CUDA DeviceMesh. CPU bucket runtime "
            "is not supported; CPU offload will be added separately."
        )


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    """Return the current rank's device for ``mesh``."""
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_module = _get_device_handle(mesh.device_type)
    if device_module is None:
        return torch.device(mesh.device_type)
    return torch.device(mesh.device_type, device_module.current_device())


def _validate_eager_params(
    named_params: list[tuple[str, nn.Parameter]],
    expected_device: torch.device | None = None,
) -> None:
    """Validate parameters supported by the eager-only path."""
    for fqn, param in named_params:
        if isinstance(param, DTensor):
            raise ValueError(
                "FlexShard eager mode expects plain parameters; "
                f"{fqn!r} is a DTensor. Convert DTensor parameters with "
                "flex_shard.layout_adapters.dtensor_to_global_layout(module) first."
            )
        if (
            expected_device is not None
            and param.device.type != "meta"
            and param.device != expected_device
        ):
            raise ValueError(
                f"Parameter {fqn!r} is on {param.device}, but FlexShard expected "
                f"{expected_device}. Move the module to the target mesh device "
                "before calling flex_shard()."
            )


def _validate_placements(
    param_placements: dict[str, tuple[Placement, ...]],
    named_params: list[tuple[str, nn.Parameter]],
) -> None:
    """Validate that placements are compatible with eager FlexShard."""
    param_dict = dict(named_params)
    expected_fqns = set(param_dict)
    actual_fqns = set(param_placements)
    missing_fqns = expected_fqns - actual_fqns
    extra_fqns = actual_fqns - expected_fqns
    if missing_fqns or extra_fqns:
        msg_parts = []
        if missing_fqns:
            msg_parts.append(f"missing placements for {sorted(missing_fqns)}")
        if extra_fqns:
            msg_parts.append(f"unexpected placements for {sorted(extra_fqns)}")
        raise ValueError(
            "BucketSpec.placement_fn must return placements for exactly the "
            f"provided parameters; {', '.join(msg_parts)}."
        )

    from .placement_contract import Placement

    for fqn, placements in param_placements.items():
        placement = _get_single_placement(placements)
        if not isinstance(placement, Placement):
            raise TypeError(
                "BucketSpec.placement_fn must return Placement instances, but "
                f"{fqn!r} uses {type(placement).__name__}."
            )


def _validate_bucket_uniform_dtype_and_placement(
    bucket_assignments: BucketParamFQNsByIndex,
    param_placements: dict[str, tuple[Placement, ...]],
    buckets: list[Any],
    named_params: list[tuple[str, nn.Parameter]],
) -> None:
    """Validate minimal eager bucket constraints."""
    param_dict = dict(named_params)
    for bucket_idx, fqns in enumerate(bucket_assignments):
        if not fqns:
            continue
        reference_dtype = param_dict[fqns[0]].dtype
        reference_placements = param_placements[fqns[0]]
        reference_compatibility_key = _get_single_placement(
            reference_placements
        ).bucket_compatibility_key()
        for fqn in fqns:
            dtype = param_dict[fqn].dtype
            if dtype != reference_dtype:
                raise ValueError(
                    f"Bucket {bucket_idx} "
                    f"{buckets[bucket_idx].patterns} "
                    f"has mixed parameter dtypes: {fqns[0]!r} uses "
                    f"{reference_dtype} but {fqn!r} uses {dtype}. "
                    "All params in a FlexShard bucket storage must share the same "
                    "dtype."
                )
            placements = param_placements[fqn]
            compatibility_key = _get_single_placement(
                placements
            ).bucket_compatibility_key()
            if compatibility_key != reference_compatibility_key:
                raise ValueError(
                    f"Bucket {bucket_idx} "
                    f"{buckets[bucket_idx].patterns} "
                    f"has mixed placements: {fqns[0]!r} uses "
                    f"{reference_placements!r} but {fqn!r} uses "
                    f"{placements!r}. All params in a FlexShard bucket must "
                    "use compatible placements because bucket collectives use "
                    "one placement layout. Split parameters with incompatible "
                    "placements into separate buckets."
                )


def alloc_grads_in_param_layout(named_params: list[tuple[str, nn.Parameter]]) -> None:
    """A ``BucketSpec.pre_backward_hook`` that allocates the missing grads
    zeroed, as views of one buffer per storage the unsharded params view, at
    their offsets, as Megatron-LM allocates ``main_grad``: a placement whose
    gradient reduction layout is its parameter layout reduces that buffer as
    is. The bucket runtime runs it for such a bucket with no hook of its own
    (see ``Placement.reduces_grads_in_param_layout``)."""
    buffers: dict[int, torch.Tensor] = {}
    for _, param in named_params:
        if not param.requires_grad or param.grad is not None:
            continue
        storage = param.untyped_storage()
        buffer = buffers.get(storage._cdata)
        if buffer is None:
            buffer = torch.zeros(
                storage.nbytes() // param.element_size(),
                dtype=param.dtype,
                device=param.device,
            )
            buffers[storage._cdata] = buffer
        offset = param.storage_offset()
        param.grad = buffer[offset : offset + param.numel()].view(param.shape)


def mesh_rank_hosts(
    global_ranks: Sequence[int], ranks_per_host: int | None = None
) -> tuple[int, ...]:
    """Each mesh rank's host, numbered 0, 1, ... in mesh order, with hosts of
    ``ranks_per_host`` consecutive global ranks, as torchrun numbers them.
    ``global_ranks`` are the mesh's ranks in mesh order. ``ranks_per_host``
    defaults to torchrun's LOCAL_WORLD_SIZE; without it, every rank is its own
    host."""
    if ranks_per_host is None:
        ranks_per_host = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if ranks_per_host <= 0:
        raise ValueError(f"ranks_per_host must be positive, got {ranks_per_host}.")
    host_ids: dict[int, int] = {}
    return tuple(
        host_ids.setdefault(rank // ranks_per_host, len(host_ids))
        for rank in global_ranks
    )


def _even_first_positions(count: int) -> list[int]:
    """Each of 0, ..., count - 1's position in the order 0, 2, 4, ..., 1, 3, ..."""
    return [
        index // 2 if index % 2 == 0 else (count + 1) // 2 + index // 2
        for index in range(count)
    ]


def assign_to_hosts_then_ranks(
    loads: Sequence[int],
    hosts: Sequence[int],
    *,
    carried_loads: Sequence[int] | None = None,
    one_per_rank: bool = False,
) -> list[int]:
    """An owner rank for each load, largest first: onto the host with the least
    load placed by this call, then the least carried load, then onto that
    host's rank with the least load placed by this call, then the least
    carried load. Remaining ties take hosts, and ranks within a host, in the
    order 0, 2, 4, ..., 1, 3, ...

    ``hosts`` gives each rank's host (see ``mesh_rank_hosts``) and
    ``carried_loads`` each rank's load from earlier calls. With
    ``one_per_rank``, each rank takes at most one load.

    So the largest loads go to hosts 0, 2, 4, ... and heavy and light hosts
    alternate in rank order. When owner rows lie back to back in rank order
    and are cut into equal chunks, a heavy host's overflow then lands in a
    light neighbor's chunks rather than pushing every later host out of its
    own. A later call with carried loads puts its largest loads on the other
    hosts."""
    num_ranks = len(hosts)
    num_hosts = len(set(hosts))
    if sorted(set(hosts)) != list(range(num_hosts)):
        raise ValueError(f"hosts must be numbered 0, 1, ..., got {tuple(hosts)}.")
    if one_per_rank and len(loads) > num_ranks:
        raise ValueError(
            f"{len(loads)} loads don't fit one per rank on {num_ranks} ranks."
        )
    carried = [0] * num_ranks if carried_loads is None else list(carried_loads)
    members: list[list[int]] = [[] for _ in range(num_hosts)]
    for rank, host in enumerate(hosts):
        members[host].append(rank)
    host_position = _even_first_positions(num_hosts)
    rank_position = [0] * num_ranks
    for ranks in members:
        for rank, position in zip(ranks, _even_first_positions(len(ranks))):
            rank_position[rank] = position
    host_carried = [sum(carried[rank] for rank in ranks) for ranks in members]
    host_load = [0] * num_hosts
    rank_load = [0] * num_ranks
    taken = [False] * num_ranks
    owners = [0] * len(loads)
    for index in sorted(range(len(loads)), key=lambda index: -loads[index]):
        host = min(
            (
                host
                for host in range(num_hosts)
                if not one_per_rank or not all(taken[rank] for rank in members[host])
            ),
            key=lambda host: (host_load[host], host_carried[host], host_position[host]),
        )
        rank = min(
            (rank for rank in members[host] if not (one_per_rank and taken[rank])),
            key=lambda rank: (rank_load[rank], carried[rank], rank_position[rank]),
        )
        owners[index] = rank
        taken[rank] = True
        host_load[host] += loads[index]
        rank_load[rank] += loads[index]
    return owners
