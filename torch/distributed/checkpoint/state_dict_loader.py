import os
import warnings
from typing import Any, cast, Dict, Optional, Set, Union

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.stateful import Stateful

from ._storage_utils import _storage_setup
from .default_planner import DefaultLoadPlanner
from .planner import LoadPlan, LoadPlanner
from .storage import StorageReader
from .utils import _all_gather_keys, _api_bc_check, _DistWrapper, _profile

__all__ = ["load_state_dict", "load"]


def load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[LoadPlanner] = None,
) -> None:
    """This method is deprecated. Please switch to 'load'."""
    warnings.warn(
        "'load_state_dict' is deprecated and will be removed in future versions. "
        "Please use 'load' instead."
    )
    storage_reader.reset()
    with _profile():
        # TODO: test returning `load` here instead.
        return _load_state_dict(
            state_dict,
            storage_reader,
            process_group,
            coordinator_rank,
            no_dist,
            planner,
        )


@_dcp_method_logger(log_exceptions=True)
@_api_bc_check
def load(
    state_dict: Dict[str, Any],
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_reader: Optional[StorageReader] = None,
    planner: Optional[LoadPlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Load a distributed ``state_dict`` in SPMD style.

    Each rank will try to read the least amount of data necessary
    to fullfill the requested `state_dict`. When loading :class:`ShardedTensor`
    or :class:`DTensor` instances, each rank only reads data for their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    load will first call ``state_dict`` before attempting deserialization, followed by
    ``load_state_dict`` once the deserialization is complete.

    .. warning::
        All tensors in ``state_dict`` must be allocated on their
        destination device *prior to* calling this function.

        All non-tensor data is loaded using `torch.load()` and modified in place
        on state_dict.

    .. warning::
        Users must call `load_state_dict` on the root module to ensure load
        pos-processing and non-tensor data properly propagates.

    .. note:
        If no process group is initialized, this function will assume the intent
        is to load a checkpoint into the local process. This can be useful in the
        case of local inference, and when using regular Tensors (as opposed to DTensor
         or ShardedTensor)

    .. note:
        Rank 0 is assumed to be the coordinator rank.

    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_reader (Optional[StorageReader]):
            Instance of StorageWriter used to perform reads. If this is not
            specified, DCP will automatically infer the reader based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[LoadPlanner]):
            Instance of LoadPlanner. If this is not specificed, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)

    Returns:
        None.

    Examples
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()
        >>> optimizer = Adagrad(my_model.parameters())
        >>> model_state_dict = my_model.state_dict()
        >>> fs_storage_reader = torch.distributed.checkpoint.FileSystemReader("/checkpoint/1")

        >>> torch.distributed.checkpoint.load_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_reader=fs_storage_reader,
        >>> )

        >>> # module.load_state_dict() function might have customized steps
        >>> # to flush the state_dict, must call it to
        >>> # ensure correct behavior.
        >>> my_model.load_state_dict(model_state_dict)

    .. note::
        load_state_dict uses collectives to coordinate reads across ranks.
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication takes place.
        In this case, the device used is given by ``torch.cuda.current_device()``
        and it is the user's responsibility to ensure that this is set so that each
        rank has an individual GPU, via ``torch.cuda.set_device()``.
    """

    no_dist = not (dist.is_available() and dist.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is unavailable or uninitialized, assuming the intent is to load in a single process."
        )

    with _profile():
        storage_reader = cast(
            StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
        )

        if no_dist:
            keys = list(state_dict.keys())
        else:
            keys = _all_gather_keys(state_dict, process_group)
            if keys != sorted(state_dict.keys()):
                warnings.warn(
                    "Detected mismatched keys in state dict after all gather!"
                    " This behavior is unsupported and may cause errors may cause errors."
                )

        statetful_sd = {}
        for key in keys:
            if key not in state_dict:
                continue
            elem = state_dict[key]
            statetful_sd[key] = (
                elem.state_dict() if isinstance(elem, Stateful) else elem
            )

        _load_state_dict(
            state_dict=statetful_sd,
            storage_reader=storage_reader,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
        )
        for key in keys:
            if key not in state_dict:
                continue
            elem = state_dict[key]
            if isinstance(elem, Stateful):
                elem.load_state_dict(statetful_sd[key])


def _load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[LoadPlanner] = None,
) -> None:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.load_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_reader, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        assert planner is not None
        metadata = storage_reader.read_metadata()
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
        storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        assert planner is not None
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: LoadPlan = distW.reduce_scatter("plan", local_step, global_step)

    @_dcp_method_logger(**ckpt_kwargs)
    def read_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)

        all_reads.wait()
        return None

    _ = distW.all_gather("read", read_data)


def _load_state_dict_from_keys(
    keys: Optional[Union[Set[str], str]] = None,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_reader: Optional[StorageReader] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Load only the specified keys from the checkpoint, if no keys are specified, the entire
    checkpoint will be loaded. Note, this method completely loads the checkpoint into the
    current process and is not distributed.

    .. warning::


    .. warning::

        All non-tensor data is loaded using `torch.load()`

    .. note:
        As opposed to the usual pattern, this function does not take a state dict as input
        and does not load inplace. Instead, a new state dict is directly initialized and read
        from file.

    .. note:
        If no process group is initialized, this function will assume the intent
        is to load a checkpoint into the local process. This can be useful in the
        case of local inference, and when using regular Tensors (as opposed to DTensor
         or ShardedTensor)

    .. note:
        Rank 0 is assumed to be the coordinator rank.

    Args:
        keys (Optional[Union[Set[str], str]]):
            Loads any key specified in this set. If no keys are specified, the entire checkpoint
            is loaded.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_reader (Optional[StorageReader]):
            Instance of StorageWriter used to perform reads. If this is not
            specified, DCP will automatically infer the reader based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)

    Returns:
        State dict from specified keys
    """
    torch._C._log_api_usage_once(
        "torch.distributed.checkpoint._load_state_dict_from_keys"
    )

    no_dist = not (dist.is_available() and dist.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is unavailable or uninitialized, assuming the intent is to load in a single process."
        )

    storage_reader = cast(
        StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
    )

    if isinstance(keys, str):
        keys = {keys}

    sd: Dict[str, Any] = {}
    _load_state_dict(
        state_dict=sd,
        storage_reader=storage_reader,
        process_group=process_group,
        no_dist=no_dist,
        planner=_EmptyStateDictLoadPlanner(keys=keys or set()),
    )

    return sd
