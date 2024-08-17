# mypy: allow-untyped-defs
import copy
import io
import math
import weakref
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor


if dist.is_available() or TYPE_CHECKING:
    from torch.distributed import distributed_c10d
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed.tensor import distribute_tensor, DTensor, Replicate


def _identity_func(
    obj: torch.Tensor,
    pg: Optional[dist.ProcessGroup],
    device: Optional[torch.device],
    companion_obj: Any,
) -> torch.Tensor:
    return obj


def _all_gather_sharded_tensor(
    sharded_tensor: "ShardedTensor",
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if pg is None:
        pg = distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    shards = sharded_tensor.local_shards()
    dim_0_size = sharded_tensor.size()[0]  # type: ignore[index]
    tensor_numel = sharded_tensor.size().numel()  # type: ignore[union-attr]
    chunk_size = math.ceil(dim_0_size / world_size) * tensor_numel // dim_0_size
    pg_device = (
        distributed_c10d._get_pg_default_device(pg) if device is None else device
    )
    if shards:
        local_tensor = shards[0].tensor.flatten()
        if local_tensor.device.type != pg_device.type:
            local_tensor = local_tensor.to(pg_device)
        num_padding = chunk_size - local_tensor.numel()
        if num_padding > 0:
            local_tensor = F.pad(local_tensor, [0, num_padding])
    else:
        local_tensor = torch.zeros(
            chunk_size, dtype=sharded_tensor.dtype, device=pg_device
        )

    tensor = torch.empty(
        chunk_size * world_size,
        dtype=local_tensor.dtype,
        device=pg_device,
    )
    dist.all_gather_into_tensor(tensor, local_tensor, group=pg)

    tensor = tensor.narrow(0, 0, tensor_numel).reshape(sharded_tensor.size())
    return tensor


class CompanionMismatch(Exception):
    ...


def _iterate_state_dict(
    iter_object: Any,
    sharded_tensor_func: Callable,
    dtensor_func: Callable,
    tensor_func: Callable,
    *,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    cpu_offload: bool = False,
    companion_obj: Any = None,
    ranks_only: Tuple[int, ...] = (),
    type_check: bool = True,
    non_blocking: bool = True,
) -> Dict[str, Any]:
    """Iterate through the state dict, applying the given functions to each tensor type.

    Args:
        iter_object (Any): the target state_dict.
        sharded_tensor_func (Callable): the function to apply to ShardedTensor
        dtensor_func (Callable): the function to apply to DTensor
        tensor_func (Callable): the function to apply to Tensor
        pg (Optional[dist.ProcessGroup]): process group passed to tensor functions
        device (Optional[torch.device]): device passed to tensor functions
        cpu_offload (bool): whether to offload the tensors to CPU memory. This option is ignored
            if a companion_obj is supplied.
        companion_obj (Any): A companion object to the state dict. If this object
            is supplied, we attempt to copy the tensor to the companion object.
        ranks_only (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.
        non_blocking (bool): whether to use non-blocking copy when copying to the companion object.
    """
    # TODO: should we use pytree?
    cpu_device = torch.device("cpu")
    if isinstance(iter_object, ShardedTensor):
        ret = sharded_tensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, DTensor):
        ret = dtensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, torch.Tensor):
        ret = tensor_func(iter_object, pg, device, companion_obj)
    elif (
        isinstance(iter_object, (int, float, str, bytes, io.BytesIO))
        or iter_object is None
    ):
        ret = iter_object
    elif isinstance(iter_object, dict):
        if companion_obj is not None and (
            not isinstance(companion_obj, dict)
            or set(companion_obj.keys()) != set(iter_object.keys())
        ):
            msg = (
                ""
                if isinstance(companion_obj, dict)
                else f"{set(companion_obj.keys())=} {set(iter_object.keys())=}"
            )
            raise CompanionMismatch(msg)

        ret = {
            key: _iterate_state_dict(
                value,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[key] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for key, value in iter_object.items()
        }
    elif isinstance(iter_object, (list, tuple)):
        if companion_obj is not None and (
            not isinstance(companion_obj, (list, tuple))
            or len(companion_obj) != len(iter_object)
        ):
            raise CompanionMismatch

        ret = [
            _iterate_state_dict(
                v,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[idx] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for idx, v in enumerate(iter_object)
        ]
        if isinstance(iter_object, tuple):
            ret = tuple(ret)
    elif not type_check:
        ret = copy.deepcopy(iter_object)
    else:
        raise ValueError(f"Unexpected value type {type(iter_object)}")

    if not ranks_only or dist.get_rank(pg) in ranks_only:
        if isinstance(ret, torch.Tensor):
            if cpu_offload and companion_obj is None:
                ret = ret.to(cpu_device)

            if companion_obj is not None:
                # TODO: support DTensor
                companion_obj.copy_(ret, non_blocking=non_blocking)
                ret = companion_obj
    else:
        ret = {} if isinstance(ret, dict) else None

    return ret


def _gather_state_dict(
    state_dict: Dict[str, Any],
    *,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    cpu_offload: bool = False,
    ranks_only: Tuple[int, ...] = (),
    type_check: bool = True,
) -> Dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensors or DTensors in
    the state_dict.


    Args:
        state_dict (Dict[str, Any]): the target sharded state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        device: (Optional[torch.device]): the device that is used to
            perform allgather for ShardedTensor. Note that gathering a DTensor
            will use the DeviceMesh. So this argument will be ignored when
            gathering a DTensor.
        cpu_offload (bool): whether to offload the tensors to CPU memory. The
            default value is False.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    def sharded_tensor_func(value, pg, device, companion_obj):
        # ShardedTensor does not seem to record the original device type.
        # So if the tensor is moved to CPU, we won't know the original type.
        # As a result, we have to rely on the user to tell us the correct one.
        cpu_device = torch.device("cpu")
        output_tensor = _all_gather_sharded_tensor(value, pg, device)
        local_shard_device = (
            value.local_shards()[0].tensor.device
            if value.local_shards()
            else cpu_device
        )
        if output_tensor.device != local_shard_device:
            value = output_tensor.to(local_shard_device)
        else:
            value = output_tensor
        return value

    def dtensor_func(value, pg, device, companion_obj):
        if value.device != value.device_mesh.device_type:
            value = value.to(value.device_mesh.device_type)
        # FSDP all_gather: [Shard(0)] -> [Replicate()]
        # HSDP all_gather: [Replicate(), Shard(0)] -> [Replicate(), Replicate()]
        # 2D FSDP + TP all_gather:
        # - [Shard(0), Shard(n)] -> [Replicate(), Replicate()]
        # - [Shard(0), Replicate()] -> [Replicate(), Replicate()]
        placements = [Replicate() for _ in value.placements]
        value = value.redistribute(
            device_mesh=value.device_mesh,
            placements=placements,
        )
        # Call `wait()` to force the tensor to be synchronous with respect
        # to the main stream.
        # See the discussion in https://github.com/pytorch/pytorch/pull/117799.
        value = value.to_local()
        if isinstance(value, AsyncCollectiveTensor):
            value = value.wait()
        return value

    return _iterate_state_dict(
        state_dict,
        sharded_tensor_func,
        dtensor_func,
        _identity_func,
        pg=pg,
        device=device,
        cpu_offload=cpu_offload,
        ranks_only=ranks_only,
        type_check=type_check,
    )


def _offload_state_dict_to_cpu(
    state_dict: Dict[str, Any],
    *,
    ranks_only: Tuple[int, ...] = (),
    type_check: bool = True,
) -> Dict[str, Any]:
    """
    Given a state_dict, this API offload all the tensors to CPU memory.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=True,
        ranks_only=ranks_only,
        type_check=type_check,
    )
    return ret


def _copy_state_dict(
    state_dict: Dict[str, Any],
    copy_state_dict: Dict[str, Any],
    non_blocking: bool = False,
    type_check: bool = True,
) -> Dict[str, Any]:
    """
    Copies all tensors in a given state dict into a different state_dict with the
    same structure. Additionally, a copied state dict with the same value references
    is returned. Editing the keys on this state dict will not affect the
    passed in copy_state_dict (but the value references are the same).

    .. warning::
        It is expected by this function that state_dict and copy_state_dict share
        the same structure and data types.

    .. warning::
        The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        copy_state_dict (Dict[str, Any]):
            The state dict we are copying into. This state_dict must have exactly
             the same structure as the source `state_dict`.
        non_blocking: (bool): Whether copy ops should be performed asynchronously
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP. The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        State Dict copy
    """

    return _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        companion_obj=copy_state_dict,
        type_check=type_check,
        non_blocking=non_blocking,
    )


def _create_cpu_state_dict(
    state_dict: Dict[str, Any], pin_memory: bool = False, share_memory: bool = False
) -> Dict[str, Any]:
    """
    Given a state_dict, create another state_dict with the same structure and elements.
    However, all tensors in the returned state_dict are new tensors on CPU. These
    tensors can be placed on pin_memory or share_memory based on the provided arguments.

    .. warning::
        Setting both `pin_memory` and `share_memory` to True significantly increases the
        latency of this method because of the nuances which require us to register memory
        as pinned directly as opposed to relying on the pin_memory cache allocator. This
        option should only be used for long lived tensors which are required to be shared.
        This is not the case as long as at least one of `pin_memory` or `share_memory` is
         set to False.

    """

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        _: Any,
    ) -> torch.Tensor:
        if len(obj.size()) == 0:
            return torch.tensor(0, dtype=obj.dtype)

        if share_memory:
            t = torch.empty(*tuple(obj.size()), dtype=obj.dtype)
            t = t.share_memory_()
            if pin_memory:

                def unpin_memory(t):
                    succ = int(torch.cuda.cudart().cudaHostUnregister(t.data_ptr()))
                    assert (
                        succ == 0
                    ), f"Unpinning shared memory failed with error-code: {succ}"

                weakref.finalize(t, unpin_memory, t)
                succ = int(
                    torch.cuda.cudart().cudaHostRegister(
                        t.data_ptr(),
                        t.numel() * t.element_size(),
                        1,  # lines up with 'cudaHostRegisterPortable'
                    )
                )
                assert (
                    succ == 0
                ), f"Pinning shared memory failed with error-code: {succ}"
            return t
        elif pin_memory:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype).pin_memory()
        else:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype)

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        tensor_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        type_check=False,
    )
    return ret


def _check_state_dict_similarity(
    state_dict: Dict[str, Any],
    compared_state_dict: Dict[str, Any],
) -> bool:
    """
    Given two state_dicts, check if the structures are the same. And
    if a [key, tensor] pair exist in one state_dict there must be
    the a corresponding pait, [key, other_tensor], in the other state_dict,
    where tensor and other_tensor have the same size and dtype.

    Return the check result.
    """

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        companion_obj: Any,
    ) -> torch.Tensor:
        if companion_obj.dtype != obj.dtype or companion_obj.size() != obj.size():
            raise CompanionMismatch
        return obj

    try:
        _iterate_state_dict(
            state_dict,
            _identity_func,
            _identity_func,
            tensor_func,
            pg=None,
            device=None,
            cpu_offload=False,
            ranks_only=(),
            companion_obj=compared_state_dict,
            type_check=False,
        )
    except CompanionMismatch:
        return False

    return True


class _TensorInfo(NamedTuple):
    size: torch.Size
    dtype: torch.dtype


def _broadcast_tensors(
    full_state_dict: Dict[str, Any],
    local_state_dict: Dict[str, Any],
    keys: List[str],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    tensors = []
    for key in keys:
        if dist.get_rank() == 0:
            full_state = full_state_dict[key]
            assert isinstance(full_state, torch.Tensor)
            full_tensor = full_state.detach().to(device)
        else:
            tensor_info = full_state_dict[key]
            full_tensor = torch.empty(
                size=tensor_info.size,
                device=device,
                dtype=tensor_info.dtype,
            )

        tensors.append(full_tensor)
        local_state = local_state_dict.get(key, None)
        if local_state is None:
            continue
        elif isinstance(local_state, DTensor):
            local_state_dict[key] = (local_state, full_tensor)
        else:
            local_state_dict[key] = full_tensor

    if pg is None:
        pg = dist.distributed_c10d._get_default_group()

    if len(tensors) > 1:
        dist._broadcast_coalesced(pg, tensors, 500, 0)
    else:
        dist.broadcast(tensors[0], src=0, group=pg)

    _distribute_tensors(local_state_dict, keys, device, pg)


def _distribute_tensors(
    local_state_dict: Dict[str, Any],
    keys: List[str],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    if pg is None:
        pg = dist.distributed_c10d._get_default_group()
    for key in keys:
        _local_state = local_state_dict.get(key, None)
        if _local_state is None or torch.is_tensor(_local_state):
            continue

        local_state = _local_state[0]
        full_tensor = _local_state[1]
        local_state_dict[key] = distribute_tensor(
            full_tensor, local_state.device_mesh, local_state.placements
        )


def _broadcast_state_dict(
    full_state_dict: Dict[str, Any],
    local_state_dict: Dict[str, Any],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
    strict: bool = False,
) -> None:
    # Broadcast from rank0's `full_state_dict` to all ranks' `local_state_dict`.
    # If strict is True, any keys in `local_state_dict` but not in `full_state_dict`
    # will be removed from `local_state_dict`.
    ret = {}
    if dist.get_rank() == 0:
        for key, value in full_state_dict.items():
            if not torch.is_tensor(value):
                ret[key] = value
            elif value.dim() == 0:
                ret[key] = value.cpu()
            else:
                ret[key] = _TensorInfo(value.size(), value.dtype)

    broadcast_list = [ret]
    dist.broadcast_object_list(broadcast_list, src=0, group=pg)
    ret = broadcast_list[0]

    # Gather values
    keys = []
    local_state_dict_keys = set(local_state_dict.keys())
    global_keys = set()
    for key, value in ret.items():
        global_keys.add(key)
        if not isinstance(value, _TensorInfo):
            if key in local_state_dict:
                local_state_dict[key] = value
            continue

        if dist.get_rank() == 0:
            ret[key] = full_state_dict[key]

        keys.append(key)
        # Broadcast every tensor to avoid OOM for now.
        if len(keys) >= 1:
            _broadcast_tensors(ret, local_state_dict, keys, device, pg)
            keys.clear()

    if strict:
        if missing_keys := (local_state_dict_keys - global_keys):
            for key in missing_keys:
                local_state_dict.pop(key)

    if keys:
        _broadcast_tensors(ret, local_state_dict, keys, device, pg)


def _distribute_state_dict(
    full_state_dict: Dict[str, Any],
    local_state_dict: Dict[str, Any],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    # Full_state_dict = True, broadcast_from_rank0 = False here. Each rank has
    # full_state_dict. Skip the broadcast in ``_broadcast_state_dict`` and
    # distribute tensors in each rank
    for key, value in full_state_dict.items():
        if key not in full_state_dict:
            continue
        if not torch.is_tensor(value):
            local_state_dict[key] = value
        elif value.dim() == 0:
            local_state_dict[key] = value.cpu()
        else:
            assert isinstance(value, torch.Tensor)
            full_tensor = value.detach().to(device)
            local_state = local_state_dict.get(key, None)
            if local_state is None:
                continue
            elif isinstance(local_state, DTensor):
                local_state_dict[key] = (local_state, full_tensor)
            else:
                local_state_dict[key] = full_tensor

            _distribute_tensors(local_state_dict, [key], device, pg)


# These APIs are from torch.distributed.checkpoint.
# TODO: We should consolidate the code here as some not all modules can depend on
# DCP.
PATH_ITEM = Union[str, int]
OBJ_PATH = Tuple[PATH_ITEM, ...]
FLATTEN_MAPPING = Dict[str, OBJ_PATH]
STATE_DICT_TYPE = Dict[str, Any]
CONTAINER_TYPE = MutableMapping[PATH_ITEM, Any]


def _traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, Any], None],
) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping, list, and tuple will be flattened and other value types are treated
    as the terminal values and will invoke ``visitor``.
    """

    def _traverse_obj(path: OBJ_PATH, value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)
        else:
            visitor(path, value)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def _flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: Any) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    _traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings


def _set_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: Any) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    cur_container = cast(CONTAINER_TYPE, root_dict)

    def extend_list(lst: List[Any], idx: int) -> None:
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        def_val: Union[CONTAINER_TYPE, List[Any]] = {} if type(key) == str else []

        if isinstance(cur_container, Mapping):
            cur_container = cast(
                CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val)
            )
        else:
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    key = path[-1]
    if type(key) == int:
        extend_list(cast(List[Any], cur_container), key)

    cur_container[key] = value


def _unflatten_state_dict(
    state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING
) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    nested: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        _set_element(nested, mapping[key], value)
    return nested
