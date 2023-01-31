import torch
import torch.distributed as dist

from functorch import make_fx
from torch._C._distributed_c10d import _create_work_from_future
from torch.futures import Future

import torch.distributed.traceable_collectives as tr_c
from torch.testing._internal.common_distributed import (
    spawn_threads_and_init_comms,
)

import torch.distributed._tensor as dt
import torch.distributed._tensor.placement_types as pt
import logging
import torch._dynamo as td

td.config.log_level=logging.DEBUG

def fwd_fun(input):
    tmp =  tr_c.all_reduce(input + 1, reduceOp="sum", group=[0,1,2,3], tag="")
    tmp1 = input + 10
    return tmp + tmp1


@spawn_threads_and_init_comms(world_size=4)
def main(self):
    print(make_fx(fwd_fun)(torch.rand(10)).graph)


main(None)

