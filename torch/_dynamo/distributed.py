"""
Manages process groups for distributed compilation in TorchDynamo.

This module handles the initialization and management of process groups used for
distributed compilation. Key features:

- Lazy initialization of compilation process groups
- Only creates groups when distributed mode is enabled and available
- Integrates with compiler_collectives configuration setting
- Provides a single global process group for compilation coordination

The process group is created only when needed and if the distributed environment
is properly initialized, making it safe to import and use this module even in
non-distributed scenarios.
"""

from typing import Optional

import torch.distributed as dist

from . import config


_COMPILE_PG: Optional[dist.ProcessGroup] = None


def get_compile_pg() -> Optional[dist.ProcessGroup]:
    if (
        config.enable_compiler_collectives
        and dist.is_available()
        and dist.is_initialized()
    ):
        global _COMPILE_PG
        if _COMPILE_PG is None:
            # , timeout=datetime.timedelta(seconds=2)
            _COMPILE_PG = dist.distributed_c10d._new_group_with_tag(
                pg_tag="pt2_compile_pg"
            )
        return _COMPILE_PG

    return None
