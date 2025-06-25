# mypy: allow-untyped-defs
import copy
import io
import logging
import os
import sys
import traceback
import weakref
from logging import getLogger
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist

from .pin_memory_utils import pin_shared_mem, unpin_memory
from .shm_mem_utils import SharedMemoryManager


if dist.is_available() or TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._tensor import DTensor

logger = getLogger()
logger.setLevel(logging.INFO)

# Allows retrying cudaHostRegister if it fails
CKPT_PIN_ALLOW_RETRY = os.environ.get("CKPT_PIN_ALLOW_RETRY", "1") == "1"
# Peeks last cudaError before pinning shared memory
CKPT_PIN_PEEK_CUDA_ERROR = os.environ.get("CKPT_PIN_PEEK_CUDA_ERROR", "0") == "1"
# Pops last cudaError before pinning shared memory
CKPT_PIN_POP_CUDA_ERROR = os.environ.get("CKPT_PIN_POP_CUDA_ERROR", "0") == "1"


def _identity_func(
    obj: torch.Tensor,
    pg: Optional[dist.ProcessGroup],
    device: Optional[torch.device],
    companion_obj: Any,
) -> torch.Tensor:
    return obj


def _pin_shared_mem(t: torch.Tensor) -> None:
    pin_shared_mem(t.data_ptr(), t.numel() * t.element_size())

    def _unpin_memory(t):
        unpin_memory(t.data_ptr())

    weakref.finalize(t, _unpin_memory, t)


class CompanionMismatch(Exception):
    """Exception raised when companion object doesn't match the expected structure."""


def _iterate_state_dict(
    iter_object: Any,
    sharded_tensor_func: Callable,
    dtensor_func: Callable,
    tensor_func: Callable,
    *,
    exclude_prefix_key: Optional[set[str]] = None,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    cpu_offload: bool = False,
    companion_obj: Any = None,
    ranks_only: Tuple[int, ...] = (),
    type_check: bool = True,
    non_blocking: bool = True,
    block_every_n_tensors: int = -1,
    _tensors_seen: int = 0,
    _tensor_bytes_seen: int = 0,
    iter_object_name: str = "",
) -> Tuple[Dict[str, Any], int, int]:
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
        block_every_n_tensors (int): Sets non_blocking to False for every n-th tensor.
        _tensors_seen (int): Number of tensors seen so far. Used for stats and making some tensor copies blocking.
        _bytes_seen (int): Total size of tensors seen so far. Used for stats.
    """
    if exclude_prefix_key:
        for prefix_key in exclude_prefix_key:
            if iter_object_name.startswith(prefix_key):
                return {}, 0, 0

    # TODO: should we use pytree?
    cpu_device = torch.device("cpu")
    if isinstance(iter_object, ShardedTensor):
        ret = sharded_tensor_func(iter_object, pg, device, companion_obj)
        _tensors_seen = _tensors_seen + 1
        _tensor_bytes_seen = (
            _tensor_bytes_seen
            + iter_object.local_tensor().numel()
            * iter_object.local_tensor().element_size()
        )
    elif isinstance(iter_object, DTensor):
        _tensors_seen = _tensors_seen + 1
        ret = dtensor_func(iter_object, pg, device, companion_obj)
        _tensor_bytes_seen = (
            _tensor_bytes_seen
            + iter_object.to_local().numel() * iter_object.to_local().element_size()
        )
    elif isinstance(iter_object, torch.Tensor):
        _tensors_seen = _tensors_seen + 1
        ret = tensor_func(iter_object, pg, device, companion_obj)
        _tensor_bytes_seen = (
            _tensor_bytes_seen + iter_object.numel() * iter_object.element_size()
        )
    elif (
        isinstance(iter_object, (int, float, str, bytes, io.BytesIO))
        or iter_object is None
    ):
        ret = iter_object
        _tensors_seen = _tensors_seen + 1
        _tensor_bytes_seen = _tensor_bytes_seen + sys.getsizeof(iter_object)

    elif isinstance(iter_object, dict):
        if companion_obj is not None and (
            not isinstance(companion_obj, dict)
            or set(companion_obj.keys()) != set(iter_object.keys())
        ):
            msg = (
                ""
                if not isinstance(companion_obj, dict)
                else "companion_obj.keys()=%s iter_object.keys()=%s"
                % (set(companion_obj.keys()), set(iter_object.keys()))
            )
            logger.error(msg)
            raise CompanionMismatch(msg)

        ret = {}
        for key, value in iter_object.items():
            try:
                obj, _tensors_seen, _tensor_bytes_seen = _iterate_state_dict(
                    value,
                    sharded_tensor_func,
                    dtensor_func,
                    tensor_func,
                    pg=pg,
                    device=device,
                    cpu_offload=cpu_offload,
                    companion_obj=(
                        companion_obj[key] if companion_obj is not None else None
                    ),
                    ranks_only=ranks_only,
                    type_check=type_check,
                    non_blocking=non_blocking,
                    block_every_n_tensors=block_every_n_tensors,
                    _tensors_seen=_tensors_seen,
                    _tensor_bytes_seen=_tensor_bytes_seen,
                    iter_object_name="%s.%s" % (iter_object_name, key)
                    if iter_object_name != ""
                    else key,
                    exclude_prefix_key=exclude_prefix_key,
                )
                ret[key] = obj
            except Exception as e:
                raise RuntimeError("Failed to iterate %s" % key) from e
    elif isinstance(iter_object, (list, tuple)):
        if companion_obj is not None and (
            not isinstance(companion_obj, (list, tuple))
            or len(companion_obj) != len(iter_object)
        ):
            raise CompanionMismatch(
                "type mismatch for key %s. companion_obj=%s %s != %s"
                % (
                    iter_object_name,
                    type(companion_obj),
                    len(companion_obj),
                    len(iter_object),
                )
            )

        ret = []
        for idx, v in enumerate(iter_object):
            obj, _tensors_seen, _tensor_bytes_seen = _iterate_state_dict(
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
                block_every_n_tensors=block_every_n_tensors,
                _tensors_seen=_tensors_seen,
                _tensor_bytes_seen=_tensor_bytes_seen,
                iter_object_name="%s[%s]" % (iter_object_name, idx),
                exclude_prefix_key=exclude_prefix_key,
            )
            ret.append(obj)
        if isinstance(iter_object, tuple):
            ret = tuple(ret)
    elif not type_check:
        ret = copy.deepcopy(iter_object)
        _tensor_bytes_seen = _tensor_bytes_seen + sys.getsizeof(iter_object)
    else:
        raise ValueError("Unexpected value type %s" % type(iter_object))

    if not ranks_only or dist.get_rank(pg) in ranks_only:
        if isinstance(ret, torch.Tensor):
            if cpu_offload and companion_obj is None:
                ret = ret.to(cpu_device)

            if companion_obj is not None:
                if (
                    block_every_n_tensors > 0
                    and _tensors_seen % block_every_n_tensors == 0
                ):
                    non_blocking = False

                # TODO: support DTensor
                companion_obj.copy_(ret, non_blocking=non_blocking)
                ret = companion_obj
    else:
        ret = {} if isinstance(ret, dict) else None

    return ret, _tensors_seen, _tensor_bytes_seen


@torch.no_grad()
def _copy_state_dict(
    state_dict: Dict[str, Any],
    copy_state_dict: Dict[str, Any],
    non_blocking: bool = False,
    block_every_n_tensors=-1,
    type_check: bool = True,
    exclude_prefix_key: Optional[set[str]] = None,
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
        block_every_n_tensors (int): if > 0, sets non_blocking to False for every n-th tensor. This
            is useful for avoiding d2h contention with other workloads, by allowing d2h from other
            streams to be scheduled.
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP. The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        State Dict copy
    """

    ret, _, _ = _iterate_state_dict(
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
        block_every_n_tensors=block_every_n_tensors,
        exclude_prefix_key=exclude_prefix_key,
    )
    return ret


@torch.no_grad()
def _create_cpu_state_dict(
    state_dict: Dict[str, Any], pin_memory: bool = False, share_memory: bool = False
) -> tuple[Dict[str, Any], int, int]:
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

        # sometimes, a tensor might have non-zero size and 0 numel. In this case, pinning memory will fail
        # so we take a best guess at how to replicate the tensor below to maintain symetry in the outputted
        # state dict
        if obj.numel() == 0 or obj.data_ptr() == 0:
            t = torch.zeros_like(obj, device="cpu")
            if share_memory:
                t = t.share_memory_()
            return t

        if share_memory:
            t = torch.empty(*tuple(obj.size()), dtype=obj.dtype)
            t = t.share_memory_()
            if pin_memory:
                try:
                    _pin_shared_mem(t)
                except Exception as e:
                    # Removed ODS logging
                    if not CKPT_PIN_ALLOW_RETRY:
                        raise e

                    logger.warning(
                        "Retrying pinning shared memory, since CKPT_PIN_ALLOW_RETRY=%s. Error was:%s, traceback.format_exc()=%s\n",
                        CKPT_PIN_ALLOW_RETRY,
                        e,
                        traceback.format_exc(),
                    )
                    _pin_shared_mem(t)
            return t
        elif pin_memory:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype).pin_memory()
        else:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype)

    ret, tensor_cnt, tensor_num_bytes = _iterate_state_dict(
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
    return ret, tensor_cnt, tensor_num_bytes


@torch.no_grad()
def _create_cpu_state_dict_with_shm_manager(
    state_dict: Dict[str, Any],
    name: str,
    shm_manager: SharedMemoryManager,
    pin_memory: bool,
) -> tuple[Dict[str, Any], int, int]:
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
    # calculate the size of the shared memory needed
    _, _, tensor_bytes_seen = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        type_check=False,
    )

    # allocate the shared memory
    buffer = shm_manager.create_buffer(
        name=name, size=tensor_bytes_seen, pin_memory=pin_memory
    )
    offset: int = 0

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        _: Any,
    ) -> torch.Tensor:
        nonlocal offset

        if len(obj.size()) == 0:
            return torch.tensor(0, dtype=obj.dtype)

        tensor_bytes = obj.numel() * obj.element_size()
        t = torch.frombuffer(buffer[offset : offset + tensor_bytes], dtype=obj.dtype)
        offset += tensor_bytes
        t = t.reshape_as(obj)
        return t

    ret, tensor_cnt, tensor_num_bytes = _iterate_state_dict(
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
    return ret, tensor_cnt, tensor_num_bytes
