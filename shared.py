# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import contextlib
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
from torch.multiprocessing.reductions import init_reductions
init_reductions()
import pickle
from functools import partial
import torch.nn as nn




def worker(rank):
    dist.init_process_group("gloo", rank=rank, world_size=2)
    if rank == 1:
        t = None
    else:
        t = None
        t = torch.ones(10).share_memory_()
        #t = torch.ones(10)

    li = [t]
    dist.broadcast_object_list(li)
    t = li[0]
    print(t)
    print(f"{rank} {t.is_shared()}")
    return


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    mp.spawn(worker, nprocs=2, args=())
