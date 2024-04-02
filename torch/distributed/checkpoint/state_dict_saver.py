from typing import Optional
import warnings

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from .planner import SavePlanner
from .default_planner import DefaultSavePlanner


from .storage import (
    StorageWriter,
)

from .metadata import Metadata, STATE_DICT_TYPE
from .utils import _DistWrapper

__all__ = ["save_state_dict", "save"]


def save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
) -> Metadata:
    """This method is deprecated. Please switch to 'save'."""
    warnings.warn(
        "'save_state_dict' is deprecated and will be removed in future versions. Please use 'save' instead."
    )

    # TODO: test returning `save` here instead.
    return _save_state_dict(state_dict, storage_writer, process_group, coordinator_rank, no_dist, planner)

def save(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
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
        This function can be used to save a state_dict without having a process group
        initialized by passing ``no_dist=True``.


    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        storage_writer (StorageWriter):
            Instance of StorageWrite use to perform writes.
        process_group (ProcessGroup):
            ProcessGroup to be used for cross-rank synchronization.
        coordinator_rank (int): Rank to use to coordinate the checkpoint.
            rank0 is used by default.
        no_dist (bool): If ``True``, distributed checkpoint will not save
            in SPMD style. (Default: ``False``)

    Returns:
        Metadata: Metadata object for the saved checkpoint.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> model_state_dict = my_model.state_dict()

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter("/checkpoint/1")
        >>> torch.distributed.checkpoint.save_state_dict(
        >>>     state_dict=model_state_dict,
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

    dumpable_state_dict = {}
    for key, elem in state_dict.items():
        dumpable_state_dict[key] = elem.state_dict() if isinstance(elem, Stateful) else elem

    return _save_state_dict(
        dumpable_state_dict,
        storage_writer,
        process_group,
        coordinator_rank,
        no_dist,
        planner
    )

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

    global_metatadata = None

    def local_step():
        assert planner is not None
        planner.set_up_planner(state_dict, distW.is_coordinator)
        storage_writer.set_up_storage_writer(distW.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        nonlocal global_metatadata

        assert planner is not None
        all_local_plans, global_metatadata = planner.create_global_plan(
            all_local_plans
        )
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan = distW.reduce_scatter("plan", local_step, global_step)

    def write_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)

        all_writes.wait()
        return all_writes.value()

    def finish_checkpoint(all_results):
        assert global_metatadata is not None
        storage_writer.finish(metadata=global_metatadata, results=all_results)
        return global_metatadata

    return distW.all_reduce("write", write_data, finish_checkpoint)
