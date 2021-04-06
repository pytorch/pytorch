import torch


def is_available():
    return hasattr(torch._C, "_spmd_init")

if is_available() and not torch._C._spmd_init():
    raise RuntimeError("Failed to initialize torch.distributed.spmd")


if is_available():
    from torch._C._distributed_spmd import (
        AllReduceComm,
        DefaultBucketer,
        DefaultTrigger,
        Engine,
    )
