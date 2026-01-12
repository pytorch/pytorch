# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import inspect
import logging
import os
import warnings
from typing import Any, cast, TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.stateful import Stateful
from ._storage_utils import _storage_setup
from .default_planner import DefaultLoadPlanner
from .planner import LoadPlan, LoadPlanner
from .storage import StorageReader
from .utils import _api_bc_check, _DistWrapper, _profile


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import Metadata

__all__ = ["load_state_dict", "load"]

logger = logging.getLogger()


@deprecated(
    "`load_state_dict` is deprecated and will be removed in future versions. "
    "Please use `load` instead.",
    category=FutureWarning,
)
def load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: StorageReader,
    process_group: dist.ProcessGroup | None = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: LoadPlanner | None = None,
) -> None:
    """This method is deprecated. Please switch to 'load'."""
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
    state_dict: dict[str, Any],
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_reader: StorageReader | None = None,
    planner: LoadPlanner | None = None,
    process_group: dist.ProcessGroup | None = None,
    no_dist: bool = False,
) -> None:
    """
    Load a checkpoint into a distributed state dict in SPMD style.

    Each rank must have the same keys in their ``state_dict`` provided to this
    API. Mismatched keys may result in hangs or errors. If unsure, you can use
    the ``utils._assert_same_keys`` API to check (but may incur communication
    costs).

    Each rank will try to read the least amount of data necessary
    to fulfill the requested `state_dict`. When loading :class:`ShardedTensor`
    or :class:`DTensor` instances, each rank only reads data for their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    load will first call ``state_dict`` before attempting deserialization, followed by
    ``load_state_dict`` once the deserialization is complete.
    For each non-``Stateful`` object, load will deserialize the object, and then replace
    it in the ``state_dict`` with the deserialized object.

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
        state_dict (Dict[str, Any]): The state_dict to load the checkpoint into.
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
            Instance of LoadPlanner. If this is not specified, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        no_dist (bool): If ``True``, this function will assume the intent is to load
            a checkpoint without using cross-rank synchronization. (Default: ``False``)
    Returns:
        None.

    Examples
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()
        >>> optimizer = Adagrad(my_model.parameters())
        >>> model_state_dict = my_model.state_dict()
        >>> fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(
        ...     "/checkpoint/1"
        ... )

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

    no_dist = no_dist or (not dist.is_available()) or (not dist.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.",
            stacklevel=2,
        )

    with _profile():
        storage_reader = cast(
            StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
        )

        # All ranks must have the same keys in their `state_dict` provided to
        # this API.  See documentation for more details.
        # Here we simply sort the keys to ensure that all ranks load values in
        # the same order.
        keys = sorted(state_dict.keys())

        stateful_sd = {}
        for key in keys:
            if key not in state_dict:
                continue
            elem = state_dict[key]
            stateful_sd[key] = elem.state_dict() if isinstance(elem, Stateful) else elem

        _load_state_dict(
            state_dict=stateful_sd,
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
                # If the state_dict is a Stateful object,
                # DCP does an in-place load in the original state dict.
                elem.load_state_dict(stateful_sd[key])
            else:
                # Otherwise, replace the state_dict with the loaded state_dict.
                state_dict[key] = stateful_sd[key]


def _load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: StorageReader,
    process_group: dist.ProcessGroup | None = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: LoadPlanner | None = None,
) -> None:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.load_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_reader, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    use_collectives = True
    metadata: Metadata | None = None

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        nonlocal use_collectives
        nonlocal metadata

        # Use global metadata if available, otherwise fallback to rank local metadata
        try:
            metadata = storage_reader.read_metadata()
        except Exception:
            logger.info(
                "Global metadata is not found. Falling back to rank local metadata."
            )

        if (
            not metadata
            and "kwargs" in inspect.signature(storage_reader.read_metadata).parameters
        ):
            try:
                metadata = storage_reader.read_metadata(rank=distW.rank)  # noqa: F841
                use_collectives = False
            except Exception:
                logger.info("Rank local metadata is not found.")

        if planner is None:
            raise AssertionError("planner is None")
        if metadata is None:
            raise AssertionError("metadata is None")
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)

        if (
            "kwargs"
            in inspect.signature(storage_reader.set_up_storage_reader).parameters
        ):
            storage_reader.set_up_storage_reader(
                metadata,
                distW.is_coordinator,
                rank=distW.rank,
                use_collectives=use_collectives,
            )
        else:
            storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        if planner is None:
            raise AssertionError("planner is None")
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: LoadPlan | None = None
    if use_collectives:
        central_plan = distW.reduce_scatter("plan", local_step, global_step)
    else:
        local_plan: LoadPlan = local_step()
        global_plan: list[LoadPlan] = global_step([local_plan])
        central_plan = global_plan[0]

    @_dcp_method_logger(**ckpt_kwargs)
    def read_data():
        if planner is None:
            raise AssertionError("planner is None")
        if central_plan is None:
            raise AssertionError("central_plan is None")
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)

        all_reads.wait()
        return None

    if use_collectives:
        _ = distW.all_gather("read", read_data)
    else:
        read_data()
        distW.barrier()


def _load_state_dict_from_keys(
    keys: set[str] | str | None = None,
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_reader: StorageReader | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
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
        keys (Optional[Union[set[str], str]]):
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
            "torch.distributed is unavailable or uninitialized, assuming the intent is to load in a single process.",
            stacklevel=2,
        )

    storage_reader = cast(
        StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
    )

    if isinstance(keys, str):
        keys = {keys}

    sd: dict[str, Any] = {}
    _load_state_dict(
        state_dict=sd,
        storage_reader=storage_reader,
        process_group=process_group,
        no_dist=no_dist,
        planner=_EmptyStateDictLoadPlanner(keys=keys),
    )

    return sd
