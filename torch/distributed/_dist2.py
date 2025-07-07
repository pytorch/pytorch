import torch
from torch.distributed.distributed_c10d import (
    get_global_rank,
    get_process_group_ranks,
    _world,
    Backend,
    BackendConfig,
    _get_default_timeout,
    _get_default_store,
    _check_valid_timeout,
    _process_group_name,
    _register_process_group,
    _DistributedBackendOptions,
    _register_process_group,
    PrefixStore,
    ProcessGroup,
)
from datetime import timedelta
from typing import Any, Optional

from torch._C._distributed_c10d import (
    
)

try:
    from torch._C._distributed_c10d import ProcessGroupNCCL
except ImportError:
    _NCCL_AVAILABLE = False

try:
    from torch._C._distributed_c10d import ProcessGroupGloo
except ImportError:
    _GLOO_AVAILABLE = False

__all__ = [
    "split_group",
    # "merge_group",
]

def split_group(self,
    new_ranks: list[int],
    parent_pg: ProcessGroup,
    timeout: Optional[timedelta] = None,
    pg_options: Optional[Any] = None,
    group_desc: Optional[str] = None
) -> ProcessGroup:
    """
    This creates a new subgroup using the specified ranks. The current rank must be included in the list of new_ranks.

    shrink in NCCL is just a special case of split and we can automatically detect it.

    TODO: add more documentation to the args/kwargs
    """
    parent_group_rank = parent_pg.rank()
    global_rank = get_global_rank(parent_pg, parent_group_rank)

    # TODO: add device id fetching
    device_id = parent_pg.bound_device_id
    if not device_id:
        raise RuntimeError(
            "No device associated with the parent pg, not safe to split any process groups"
        )

    global_ranks_parent_pg = get_process_group_ranks(parent_pg)

    if global_rank not in global_ranks_parent_pg:
        raise ValueError(
            f"Global rank {global_rank} is not part of the parent group {parent_pg}"
        )

    parent_backend = parent_pg._get_backend(torch.device("cuda"))
    # if the parent backend does not support splitting, raise error
    # currently this API only support NCCL backend
    if not parent_backend or not parent_backend.supports_splitting:
        raise RuntimeError(
            "No backend for the parent process group or its backend does not support splitting"
        )

    # set the group_desc before the color or no_cloor split
    group_desc = (
        f"{parent_pg.group_desc}:split:{parent_backend.comm_split_count()}"  # type: ignore[attr-defined]
        if group_desc is None
        else group_desc
    )

    parent_backend_str, _ = _world.pg_map[parent_pg]
    # same type of backend as the parent process group
    backend = Backend(parent_backend_str)
    backend_config = BackendConfig(backend)

    if pg_options is None:
        # default pg_options same as the parent process group
        pg_options = parent_backend.options

    # this timeout defaulting/validation is used for all the new_groups/new_subgroups variants,
    # which may just pass their timeout value (or None)
    if timeout is None:
        timeout = _get_default_timeout(backend)
    _check_valid_timeout(timeout)

        # find my group of ranks and my group local rank in split_ranks
    my_group = None
    group_rank = -1

    if len(new_ranks) == 0:
        raise ValueError("the split group cannot be empty")
    if len(new_ranks) > parent_pg.size():
        raise ValueError(
            "the split group's size should be less or equal to the world_size set by init_process_group"
        )
    if len(new_ranks) != len(set(new_ranks)):
        raise ValueError("the split group cannot have duplicate ranks")
    new_ranks = sorted(new_ranks)
    if parent_group_rank in new_ranks:
        my_group = new_ranks
        group_rank = new_ranks.index(parent_group_rank)

    # if my rank does not belong to any sub group,
    # no_color split should be called
    if my_group is None or group_rank == -1:
        if parent_backend_str == Backend.NCCL:
            parent_backend.perform_nocolor_split(device_id)  # type: ignore[attr-defined]
        return None

    default_store = _get_default_store()
    group_name = _process_group_name(my_group, use_hashed_name=False)
    parent_global_to_group_ranks = _world.pg_group_ranks[parent_pg]
    parent_group_to_global_ranks = {
        group_rank: global_rank
        for global_rank, group_rank in parent_global_to_group_ranks.items()
    }
    global_ranks_in_my_group = [parent_group_to_global_ranks[rank] for rank in my_group]

    prefix_store = PrefixStore(f"{group_name}/", default_store)
    # We register the backend after initializing and timeout is set in pg_options.
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        len(my_group),
    )
    pg.bound_device_id = device_id  # type: ignore[union-attr]
    pg_options._timeout = timeout  # type: ignore[union-attr]
    pg_options.group_name = group_name  # type: ignore[union-attr]

    if parent_backend_str == Backend.NCCL:
        pg_options.split_from = parent_backend  # type: ignore[union-attr]
        pg_options.split_color = _process_group_color(my_group)  # type: ignore[union-attr]
        pg_options.global_ranks_in_group = global_ranks_in_my_group  # type: ignore[union-attr]
        backend_type = ProcessGroup.BackendType.NCCL
        if not isinstance(pg_options, ProcessGroupNCCL.Options):
            raise RuntimeError(
                "Expected pg_options argument to be of type ProcessGroupNCCL.Options"
            )
        backend_class = ProcessGroupNCCL(
            prefix_store, group_rank, len(my_group), pg_options
        )
    elif parent_backend_str == Backend.GLOO:
        backend_type = ProcessGroup.BackendType.GLOO
        if not isinstance(pg_options, ProcessGroupGloo.Options):
            raise RuntimeError(
                "Expected pg_options argument to be of type ProcessGroupNCCL.Options"
            )
        backend_class = ProcessGroupGloo(
            prefix_store, group_rank, len(my_group), pg_options._timeout
        )
    else:
        assert parent_backend_str.upper() in Backend._plugins, (
            f"Unknown c10d backend type {parent_backend_str.upper()}"
        )
        backend_plugin = Backend._plugins[parent_backend_str.upper()]
        creator_fn = backend_plugin.creator_fn
        extended_api = backend_plugin.extended_api
        backend_type = ProcessGroup.BackendType.CUSTOM
        if not extended_api:
            backend_class = creator_fn(prefix_store, group_rank, len(my_group), timeout)
        else:
            dist_backend_opts = _DistributedBackendOptions()
            dist_backend_opts.store = prefix_store
            dist_backend_opts.group_rank = group_rank
            dist_backend_opts.group_size = len(my_group)
            backend_class = creator_fn(dist_backend_opts, pg_options)

    pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(torch.device("cuda"), backend_type, backend_class)

    # set group_name and group_desc to backend
    assert group_name is not None
    assert group_desc is not None
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    if parent_backend_str == Backend.NCCL:
        # always eagerly initialize the backend in split_group
        eager_backend = pg._get_backend(device_id)
        eager_backend.eager_connect_single_device(device_id)

    # update global state
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)
    _world.pg_backend_config[pg] = str(backend_config)
    pg_tag = f"ptd:{group_name}"
    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag

    # Create the global rank to group rank mapping
    _world.pg_group_ranks[pg] = {
        global_rank: group_rank
        for group_rank, global_rank in enumerate(global_ranks_in_my_group)
    }

    return pg


# def merge_group(self, store: Store, options: MergeOptions) -> ProcessGroup:
#     """
#     Merge multiple groups together. For the case where we want N independent groups to be merged, you should split to a single rank and then call merge_group across the size 1 groups.

#     Ranks are assigned by the backend.
#     """
#     pass
