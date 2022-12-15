
import functools
import os
import uuid
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch._dynamo.utils import same
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.graph_manipulation import replace_target_nodes_with

def matmul_cat_col(a, b, c, d, e, f, *, all_reduce):
    z = torch.cat((a, b))
    all_reduce(z)
    return (z,)

def compile(func, example_inputs):
    graph = make_fx(func)(*example_inputs)
    return inductor_compile_fx(graph, example_inputs)

def eager_all_reduce(x):
    # return nccl.all_reduce([x])
    return dist.all_reduce(x, async_op=False)

if __name__ == '__main__':
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12345")
    # our_uuid = uuid.UUID('a8098c1a-f86e-11da-bd1a-00112444be1e')
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')
    # nccl.init_rank(world_size, our_uuid.bytes*8, rank)
    inputs = (torch.ones(4,4, device="cuda") + rank,) * 6
    correct_out = matmul_cat_col(*inputs, all_reduce=eager_all_reduce)

    compiled_matmul_cat_col = compile(
        functools.partial(matmul_cat_col, all_reduce=torch.ops.c10d.traceable_allreduce),
        inputs
    )
    inductor_out = compiled_matmul_cat_col(*inputs)
    print(f"rank {rank}: {correct_out}, {inductor_out}")
    assert same(correct_out, inductor_out)
