from typing import Any, Dict, Optional

import torch.distributed as dist

from .storage import (
    StorageReader,
)
from .planner import LoadPlanner
from .default_planner import DefaultLoadPlanner

from .utils import _DistWrapper

def load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: LoadPlanner = None
) -> None:
    """
    Load a distributed state_dict in SPMD style.

    Each rank will try to read the least amount of data necessary
    to fullfill the requested `state_dict`.

    When loading ShardedTensor instances, each rank only
    reads data for their local shards.

    All tensors in ``state_dict`` must be allocated on their
    destination device prior to calling this function.

    All non-tensor data is loaded using `torch.load()` and modified in place
    on state_dict.

    Users must call `load_state_dict` on the root module to ensure load
    pos-processing and non-tensor data properly propagates.

    This function can be used for local inference and load a checkpoint
    produced by ``save_state_dict`` without having a process group initialized
    by passing ``no_dist=True`` and by using Tensors instead of ShardedTensors.

    Args:
        state_dict (Dict[str, Any]) : The state_dict to load. Note that this
            state dict will updated in places.
        storage_reader (StorageReader): StorageReader used to load data from.
        process_group (ProcessGroup): ProcessGroup to be used for cross-rank synchronization
        coordinator_rank (int): Rank to use to coordinate the checkpoint, rank0 is used by default
        no_dist (bool): Don't attempt to load in SPMD style. Default to False

    Returns:
        None.

    Examples
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()
        >>> optimizer = Adagrad(my_model.parameters())
        >>> model_state_dict = my_model.state_dict()
        >>> fs_storage_loader = torch.distributed._shard.checkpoint.FileSystemLoader("/checkpoint/1")

        >>> torch.distributed._shard.checkpoint.load_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_reader=fs_storage_loader,
        >>> )

        >>> # module.load_state_dict() function might have customized steps
        >>> # to flush the state_dict, must call it to
        >>> # ensure correct behavior.
        >>> my_model.load_state_dict(model_state_dict)

    .. note:: load_state_dict uses collectives to coordinate reads across ranks.
        For NCCL-based process groups, internal tensor representations of objects
        must be moved to the GPU device before communication takes place. In this
        case, the device used is given by ``torch.cuda.current_device()`` and it
        is the user's responsibility to ensure that this is set so that each rank
        has an individual GPU, via ``torch.cuda.set_device()``
    """
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()


    def local_step():
        assert planner is not None
        metadata = storage_reader.read_metadata()
        planner.init(state_dict, metadata, distW.is_coordinator)
        storage_reader.init(metadata, distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        assert planner is not None
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan = distW.reduce_scatter("plan", local_step, global_step)

    def read_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)

        all_reads.wait()
        return None

    _ = distW.all_gather("read", read_data)
