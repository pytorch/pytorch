# This file contains stuff that is shared between FSDP-1 and FSDP-2.

import os  # noqa: C101

from typing import Any

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.distributed._composable.fsdp.fully_shard import FSDPModule as FSDP2

use_fsdp2 = os.environ.get("FSDP_VERSION", "1") == "2"

def fsdp_type() -> Any:
    return FSDP2 if use_fsdp2 else FSDP1
