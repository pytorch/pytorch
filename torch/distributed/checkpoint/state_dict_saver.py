# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import inspect
import os
import warnings
from concurrent.futures import Future
from enum import Enum
from typing import cast, Optional, Union
from typing_extensions import deprecated

import torch
import torch.distributed as dist
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
from torch.distributed.checkpoint._async_executor import (  # noqa: TC001
    _AsyncCheckpointExecutor,
)
from torch.distributed.checkpoint._async_process_executor import (
    _ProcessBasedAsyncCheckpointExecutor,
)
from torch.distributed.checkpoint._async_thread_executor import (
    _ThreadBasedAsyncCheckpointExecutor,
)
from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.staging import AsyncStager
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.distributed_c10d import _get_default_group

from .utils import _api_bc_check, _DistWrapper, _profile


__all__ = ["save_state_dict", "save", "async_save", "AsyncCheckpointerType"]


class AsyncCheckpointerType(Enum):
    """Enum for async checkpointer type."""

    THREAD = "thread"
    PROCESS = "process"


@deprecated(
    "`save_state_dict` is deprecated and will be removed in future versions."
    "Please use `save` instead.",
    category=FutureWarning,
)
def save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
) -> Metadata:
    """This method is deprecated. Please switch to 'save'."""
    storage_writer.reset()

    # TODO: test returning `save` here instead.
    with _profile():
        return _save_state_dict(
            state_dict,
            storage_writer,
            process_group,
            coordinator_rank,
            no_dist,
            planner,
        )


@_dcp_method_logger(log_exceptions=True)  # type: ignore[arg-type]
@_api_bc_check
def save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    no_dist: bool = False,
) -> Metadata:
    """
    Save a distributed model in SPMD style.

    This function is different from ``torch.save()`` as it handles
    ``ShardedTensor`` , and ``DTensor`` by having each rank only save their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    save will call ``state_dict`` before serialization.

    .. warning::
        There is no guarantees of Backwards Compatibility across PyTorch versions
        for saved state_dicts.

    .. warning::
        If using the `process_group` argument, make sure that only its ranks
        call `save_state_dict` and that all data in state_dict belong to it.

    .. note::
        When saving checkpoint for FSDP's `ShardingStrategy.HYBRID_SHARD`, only one of
        the shard_group should be calling `save_state_dict` and the corresponding process
        group needs to be passed in.

    .. note::
        If no process group is available, this function assumes the intention is to save the
         state_dict in the local process.

    .. note:
        Rank 0 is assumed to be the coordinator rank.


    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_writer (Optional[StorageWriter]):
            Instance of StorageWriter used to perform writes. If this is not
            specified, DCP will automatically infer the writer based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[SavePlanner]):
            Instance of SavePlanner. If this is not specificed, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        no_dist (bool):
            If ``True``, this function will assume the intent is to load
            a checkpoint without using cross-rank synchronization.
            (Default: ``False``)

    Returns:
        Metadata: Metadata object for the saved checkpoint.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> state_dict = {"model": my_model}

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(
        ...     "/checkpoint/1"
        ... )
        >>> torch.distributed.checkpoint.save(
        >>>     state_dict=state_dict,
        >>>     storage_writer=fs_storage_writer,
        >>> )

    .. note::
        save_state_dict uses collectives to coordinate writes across ranks.
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication takes place.
        In this case, the device used is given by ``torch.cuda.current_device()``
        and it is the user's responsibility to ensure that this is set so that
        each rank has an individual GPU, via ``torch.cuda.set_device()``.
    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save")

    no_dist = no_dist or (not dist.is_available()) or (not dist.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process."
        )

    with _profile():
        storage_writer = cast(
            StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False)
        )

        return _save_state_dict(
            state_dict=_stateful_to_state_dict(state_dict),
            storage_writer=storage_writer,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
        )


@_dcp_method_logger(log_exceptions=True)
def async_save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    async_checkpointer_type: AsyncCheckpointerType = AsyncCheckpointerType.THREAD,
) -> Future:
    """Asynchronous version of ``save``. This code first de-stages the state_dict on to the
    staging storage (defaults to CPU memory), and then calls the `save` in a separate thread.

    .. warning::
        This feature is experimental and subject to change.

    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_writer (Optional[StorageWriter]):
            Instance of StorageWriter used to perform 'stage' and  'save'. If
            this is not specified, DCP will automatically infer the writer based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[SavePlanner]):
            Instance of SavePlanner. If this is not specificed, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)

    Returns:
        Future: A future holding the resultant Metadata object from `save`.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> state_dict = {"model": my_model}

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(
        ...     "/checkpoint/1"
        ... )
        >>> checkpoint_future = torch.distributed.checkpoint.async_save(
        >>>     state_dict=state_dict,
        >>>     storage_writer=fs_storage_writer,
        >>> )
        >>>
        >>> # ... do some work ...
        >>>
        >>> checkpoint_future.result()

    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint.async_save")

    if dist.is_available() and dist.is_initialized():
        pg = process_group or _get_default_group()
        assert (
            torch.device("cpu") in pg._device_types  # type: ignore[attr-defined]
        ), (
            "A CPU backend must be enabled for async save; try initializing process group with 'cpu:gloo,cuda:nccl'"
        )

    storage_writer = cast(
        StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False)
    )

    state_dict = _stateful_to_state_dict(state_dict)

    @_dcp_method_logger(log_exceptions=True)
    def stage_state_dict():
        if isinstance(storage_writer, AsyncStager):
            staged_state_dict = storage_writer.stage(state_dict)
        else:  # provides bwc for storage_writers not implementing AsyncStager
            staged_state_dict = _create_cpu_state_dict(state_dict)
            _copy_state_dict(state_dict, staged_state_dict, type_check=False)

        return staged_state_dict

    staged_state_dict = stage_state_dict()

    executor: _AsyncCheckpointExecutor = (
        _ProcessBasedAsyncCheckpointExecutor()
        if async_checkpointer_type == AsyncCheckpointerType.PROCESS
        else _ThreadBasedAsyncCheckpointExecutor()
    )

    f: Future = executor.execute_save(
        staged_state_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
    )

    @_dcp_method_logger(log_exceptions=True)
    def maybe_synchronize_staging():
        if (
            isinstance(storage_writer, AsyncStager)
            and storage_writer.should_synchronize_after_execute
        ):
            storage_writer.synchronize_staging()

    maybe_synchronize_staging()

    return f


@_dcp_method_logger(log_exceptions=True)
def _stateful_to_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Creates a shallow copy of `state_dict` where `state_dict` is called for each Stateful object."""
    stateful_state_dict = {}
    for key, elem in state_dict.items():
        stateful_state_dict[key] = (
            elem.state_dict() if isinstance(elem, Stateful) else elem
        )
    return stateful_state_dict


def _save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
) -> Metadata:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metadata = None

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        assert planner is not None
        storage_meta = storage_writer.storage_meta()
        if "storage_meta" not in inspect.signature(planner.set_up_planner).parameters:
            warnings.warn(
                "The function definition for SavePlanner.set_up_planner has been updated"
                " to include the storage_meta argument. Please update your implementation"
                " to include this parameter."
            )
            planner.set_up_planner(state_dict, distW.is_coordinator)  # type: ignore[call-arg, arg-type]
        else:
            planner.set_up_planner(
                state_dict=state_dict,
                storage_meta=storage_meta,
                is_coordinator=distW.is_coordinator,
            )
        storage_writer.set_up_storage_writer(distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        nonlocal global_metadata

        assert planner is not None
        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: SavePlan = distW.reduce_scatter("plan", local_step, global_step)

    @_dcp_method_logger(**ckpt_kwargs)
    def write_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)

        all_writes.wait()
        return all_writes.value()

    @_dcp_method_logger(**ckpt_kwargs)
    def finish_checkpoint(all_results):
        assert global_metadata is not None
        storage_writer.finish(metadata=global_metadata, results=all_results)
        return global_metadata

    return distW.all_reduce("write", write_data, finish_checkpoint)
