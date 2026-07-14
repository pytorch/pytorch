# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import inspect
import os
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import cast, TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch.distributed as dist
from torch.distributed._state_dict_utils import STATE_DICT_TYPE
from torch.distributed.checkpoint._async_process_executor import (
    _ProcessBasedAsyncCheckpointExecutor,
)
from torch.distributed.checkpoint._async_thread_executor import (
    _ThreadBasedAsyncCheckpointExecutor,
)
from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.staging import (
    AsyncStager,
    DefaultStager,
    StagingOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.storage import StorageWriter, WriteResult
from torch.distributed.distributed_c10d import _get_default_group

from .utils import _api_bc_check, _DistWrapper, _profile


if TYPE_CHECKING:
    from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor


__all__ = [
    "save_state_dict",
    "save",
    "async_save",
    "AsyncCheckpointerType",
    "AsyncSaveResponse",
]


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
    process_group: dist.ProcessGroup | None = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: SavePlanner | None = None,
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
    checkpoint_id: str | os.PathLike | None = None,
    storage_writer: StorageWriter | None = None,
    planner: SavePlanner | None = None,
    process_group: dist.ProcessGroup | None = None,
    no_dist: bool = False,
    use_collectives: bool = True,
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
            Instance of SavePlanner. If this is not specified, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        no_dist (bool):
            If ``True``, this function will assume the intent is to load
            a checkpoint on a single rank/process.
            (Default: ``False``)
        use_collectives (bool): If ``False``, this function will assume the intent is to save
            a checkpoint without using cross-rank synchronization.
            (Default: ``True``)
            This configuration is experimental and should be used with caution.
            It will change the format of the saved checkpoint and may not be backward compatible.

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
            "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process.",
            stacklevel=2,
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
            use_collectives=use_collectives,
        )


@dataclass
class AsyncSaveResponse:
    """This class contains futures for staging and upload completion.
    It is returned by async_save().
    staging_completion is a future that indicates when local copy
    of state_dict is complete.
    upload_completion is a future that indicates when a checkpoint
    completed saving.
    """

    staging_completion: Future[None]
    upload_completion: Future[None]


@_dcp_method_logger(log_exceptions=True)
def async_save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_writer: StorageWriter | None = None,
    planner: SavePlanner | None = None,
    process_group: dist.ProcessGroup | None = None,
    async_checkpointer_type: AsyncCheckpointerType = AsyncCheckpointerType.THREAD,
    async_stager: AsyncStager | None = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Future | AsyncSaveResponse:
    """Asynchronous version of ``save``. This code first de-stages the state_dict on to the
    staging storage (defaults to CPU memory), and then calls the `save` in a separate thread.

    .. warning::
        This feature is experimental and subject to change.
        MUST CALL CLOSE AFTER LAST CHECKPOINT IS SAVED

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
            Instance of SavePlanner. If this is not specified, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        async_checkpointer_type (AsyncCheckpointerType):
            whether to do checkpoint in separate thread or process
            (Default: ``AsyncCheckpointerType.THREAD``)
        async_stager (AsyncStager):
            provides staging implementation. If storage_writer implements AsyncStager
            and async_stager is provided, async_stager will be used for staging
        no_dist (bool):
            If ``True``, this function will assume the intent is to save
            a checkpoint on a single rank/process.
            (Default: ``False``)
        use_collectives: If False, Save the checkpoint without rank coordination. (Default: ``True``)
            This configuration is experimental and should be used with caution.
            It will change the format of the saved checkpoint and may not be backward compatible.

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
        if torch.device("cpu") not in pg._device_types:
            raise AssertionError(
                "A CPU backend must be enabled for async save; try initializing process group with 'cpu:gloo,cuda:nccl'"
            )

    if async_stager is None:
        if storage_writer is not None and isinstance(storage_writer, AsyncStager):
            # bwc with old storage_writers
            async_stager = storage_writer
        else:
            async_stager = DefaultStager(
                StagingOptions(
                    False,
                    False,
                    False,
                    False,
                )
            )

    state_dict = _stateful_to_state_dict(state_dict)

    @_dcp_method_logger(log_exceptions=True)
    def stage_state_dict() -> Future[STATE_DICT_TYPE] | STATE_DICT_TYPE:
        return async_stager.stage(state_dict)

    staging_future_or_state_dict = stage_state_dict()

    upload_executor: _AsyncCheckpointExecutor = (
        _ProcessBasedAsyncCheckpointExecutor()
        if async_checkpointer_type == AsyncCheckpointerType.PROCESS
        else _ThreadBasedAsyncCheckpointExecutor()
    )

    upload_future: Future = upload_executor.execute_save(
        staging_future_or_state_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )

    if isinstance(staging_future_or_state_dict, Future):
        staging_future = staging_future_or_state_dict
        return_staging_future: Future[None] = Future()

        def callback(
            original_staging_future: Future[STATE_DICT_TYPE],
            return_staging_future: Future[None] = return_staging_future,
        ):
            try:
                original_staging_future.result()
                return_staging_future.set_result(None)
            except Exception as e:
                return_staging_future.set_exception(e)

        if not staging_future.done():
            staging_future.add_done_callback(callback)
        else:
            return_staging_future.set_result(None)

        # return new AsyncSaveResponse for users using new ZOC implementation
        return AsyncSaveResponse(
            staging_completion=return_staging_future, upload_completion=upload_future
        )
    else:

        @_dcp_method_logger(log_exceptions=True)
        def maybe_synchronize_staging():
            if async_stager.should_synchronize_after_execute:
                async_stager.synchronize_staging()

        maybe_synchronize_staging()
        return upload_future


@_dcp_method_logger(log_exceptions=True)
def _stateful_to_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Creates a shallow copy of `state_dict` where `state_dict` is called for each Stateful object."""
    stateful_state_dict = {}
    for key, elem in state_dict.items():
        # Apply _dcp_method_logger to each state_dict() call
        def _elem_to_state_dict(elem):
            return elem.state_dict() if isinstance(elem, Stateful) else elem

        _elem_to_state_dict.__name__ = f"_stateful_to_state_dict.{key}"

        stateful_state_dict[key] = _dcp_method_logger(log_exceptions=True)(
            _elem_to_state_dict
        )(elem)
    return stateful_state_dict


def _save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: dist.ProcessGroup | None = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: SavePlanner | None = None,
    use_collectives: bool = True,
) -> Metadata:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    if planner is None:
        raise AssertionError("planner is None")

    global_metadata = None

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        if planner is None:
            raise AssertionError("planner is None")
        storage_meta = storage_writer.storage_meta()
        if "storage_meta" not in inspect.signature(planner.set_up_planner).parameters:
            warnings.warn(
                "The function definition for SavePlanner.set_up_planner has been updated"
                " to include the storage_meta argument. Please update your implementation"
                " to include this parameter.",
                stacklevel=2,
            )
            planner.set_up_planner(state_dict, distW.is_coordinator)  # type: ignore[call-arg, arg-type]
        else:
            planner.set_up_planner(
                state_dict=state_dict,
                storage_meta=storage_meta,
                is_coordinator=distW.is_coordinator,
            )

        if (
            "kwargs"
            in inspect.signature(storage_writer.set_up_storage_writer).parameters
        ):
            storage_writer.set_up_storage_writer(
                distW.is_coordinator,
                rank=distW.rank,
                use_collectives=use_collectives,
            )
        else:
            storage_writer.set_up_storage_writer(distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        nonlocal global_metadata

        if planner is None:
            raise AssertionError("planner is None")
        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: SavePlan | None = None
    if use_collectives:
        central_plan = distW.reduce_scatter("plan", local_step, global_step)
    else:
        local_plan: SavePlan = local_step()
        global_plan: list[SavePlan] = global_step([local_plan])
        central_plan = global_plan[0]

    @_dcp_method_logger(**ckpt_kwargs)
    def write_data():
        if planner is None:
            raise AssertionError("planner is None")
        if central_plan is None:
            raise AssertionError("central_plan is None")
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)

        all_writes.wait()
        return all_writes.value()

    @_dcp_method_logger(**ckpt_kwargs)
    def finish_checkpoint(all_results):
        if global_metadata is None:
            raise AssertionError("global_metadata is None")
        storage_writer.finish(metadata=global_metadata, results=all_results)
        return global_metadata

    if use_collectives:
        metadata = distW.all_reduce("write", write_data, finish_checkpoint)
    else:
        write_results: list[WriteResult] = write_data()
        metadata = finish_checkpoint([write_results])
        distW.barrier()

    return metadata
