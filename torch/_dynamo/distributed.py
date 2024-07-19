import torch.distributed as dist
from typing import Optional
from . import config
import datetime

_COMPILE_PG = None

def get_compile_pg() -> Optional[dist.ProcessGroup]:
    if config.enable_compile_pg and dist.is_available() and dist.is_initialized():
        from torch._C._distributed_c10d import ProcessGroupNCCL

        global _COMPILE_PG
        if _COMPILE_PG is None:
            _COMPILE_PG = dist.distributed_c10d._new_group_with_tag(pg_tag="pt2_compile_pg", timeout=datetime.timedelta(seconds=2))
        return _COMPILE_PG

    return None
