from datetime import timedelta
from typing import Optional

import torch
from torch._C._distributed_c10d import Backend
from torch.distributed.distributed_c10d import _check_valid_timeout, ProcessGroup


__all__ = [
    "split_group",
]


def split_group(
    new_ranks: list[int],
    parent_pg: ProcessGroup,
    timeout: Optional[timedelta] = None,
    pg_options: Optional[Backend.Options] = None,
    group_desc: Optional[str] = None,
) -> Optional[ProcessGroup]:
    """
    This creates a new subgroup using the specified ranks. The current rank must be included in the list of new_ranks.

    TODO: add more documentation to the args/kwargs
    """
    if len(new_ranks) == 0:
        raise ValueError("the split group cannot be empty")
    if len(new_ranks) > parent_pg.size():
        raise ValueError(
            "the split group's size should be less or equal to the world_size set by init_process_group"
        )
    if len(new_ranks) != len(set(new_ranks)):
        raise ValueError("the split group cannot have duplicate ranks")
    new_ranks = sorted(new_ranks)

    parent_backend = parent_pg._get_backend(torch.device("cuda"))
    # set the group_desc before the color or no_cloor split
    group_desc = (
        f"{parent_pg.group_desc}:split:{parent_backend.comm_split_count()}"  # type: ignore[attr-defined]
        if group_desc is None
        else group_desc
    )
    # TODO: Need a better way to get the split group name
    group_name = f"{parent_pg.group_name}:split:{list(new_ranks)}"

    if pg_options is None:
        # default pg_options same as the parent process group
        pg_options = parent_backend.options

    # If not set we can reuse the timeout from parent process group.
    if timeout is None:
        timeout = pg_options._timeout
    _check_valid_timeout(timeout)
    pg_options._timeout = timeout
    split_backend = parent_backend.split_group(new_ranks, pg_options, group_desc)

    if not split_backend:
        return None

    # We register the backend after initializing and timeout is set in pg_options.
    pg: ProcessGroup = ProcessGroup(
        split_backend.store(),
        split_backend.rank(),
        split_backend.size(),
    )
    backend_type = parent_pg.default_backend_type
    pg._set_default_backend(backend_type)
    pg._register_backend(torch.device("cuda"), backend_type, split_backend)
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    return pg
