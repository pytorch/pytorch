# Owner(s): ["oncall: r2p"]

# This is a helper script for
# test_run.py::ElasticLaunchTest::test_virtual_local_rank. It prints out the
# generated inductor output for a simple function.

import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch._inductor import codecache


@torch.compile
def myfn(x: torch.Tensor) -> torch.Tensor:
    return x + x


dist.init_process_group(backend="nccl")

local_rank = int(os.environ.get("LOCAL_RANK", "cuda:0"))
torch.cuda.set_device(local_rank)


def print_output_code(original_fn):
    def wrapper(msg, *args, **kwargs):
        # Check if this is the "Output code:" message
        if args and "Output code:" in msg:
            print(args[0])

    return wrapper


x = torch.rand(2, 2, device="cuda")

with patch.object(
    codecache.output_code_log,
    "debug",
    side_effect=print_output_code(codecache.output_code_log.debug),
):
    y = myfn(x)

dist.destroy_process_group()
