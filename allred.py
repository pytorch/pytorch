
import os
import torch
from torch._dynamo.utils import same
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.graph_manipulation import replace_target_nodes_with

def matmul_cat_col(a, b, c, d, e, f):
    x = torch.matmul(a, b)
    y = torch.matmul(c, d)
    z = torch.cat((x, y))
    z = torch.ops.c10d.traceable_allreduce(z)
    g = torch.matmul(e, f)
    return (torch.add(z, g.repeat(2, 1)), )

def compile(func, example_inputs):
    graph = make_fx(func)(*example_inputs)
    replace_target_nodes_with(graph, "call_function", torch.ops.c10d.allreduce_.default, "call_function", torch.ops.c10d.traceable_allreduce.default)
    print(graph)
    return inductor_compile_fx(graph, example_inputs)

if __name__ == '__main__':
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12345")
    for _ in range(int(os.getenv("RANK"))):
        # advance random seed to a unique seed per rank        
        torch.randn(1)
    inputs = (torch.randn(4,4, device="cuda"),) * 6
    torch.distributed.init_process_group(backend='nccl')
    correct_out = matmul_cat_col(*inputs)
    compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
    inductor_out = compiled_matmul_cat_col(*inputs)
    assert same(correct_out, inductor_out)
    print(inductor_out)
