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
