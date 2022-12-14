
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
    z = torch.ops.c10d.traceable_allreduce([z])
    g = torch.matmul(e, f)
    z = z[0]
    return (torch.add(z, g.repeat(2, 1)), )

def compile(func, example_inputs):
    graph = make_fx(func)(*example_inputs)
    replace_target_nodes_with(graph, "call_function", torch.ops.c10d.allreduce_.default, "call_function", torch.ops.c10d.traceable_allreduce.default)
    print(graph)
    return inductor_compile_fx(graph, example_inputs)

if __name__ == '__main__':
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    inputs = (torch.randn(4,4, device="cuda"),) * 6
    torch.distributed.init_process_group(backend='nccl')
    correct_out = matmul_cat_col(*inputs)
    compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
    inductor_out = compiled_matmul_cat_col(*inputs)
    assert same(correct_out, inductor_out)
    print(inductor_out)


# ==========
# def forward(self, a_1, b_1, c_1, d_1, e_1, f_1):
#     mm = torch.ops.aten.mm.default(a_1, b_1);  a_1 = b_1 = None
#     mm_1 = torch.ops.aten.mm.default(c_1, d_1);  c_1 = d_1 = None
#     cat = torch.ops.aten.cat.default([mm, mm_1]);  mm = mm_1 = None
#     _tensor_constant0 = self._tensor_constant0
#     _tensor_constant1 = self._tensor_constant1
#     allreduce__default = torch.ops.c10d.allreduce_.default([cat], _tensor_constant0, _tensor_constant1, -1);  cat = _tensor_constant0 = _tensor_constant1 = None
#     comm_result = torch.distributed._spmd.comm_tensor._wrap_comm_result(allreduce__default);  allreduce__default = None
#     getitem = comm_result[0]
#     getitem_1 = getitem[0];  getitem = None
#     getitem_2 = comm_result[1];  comm_result = None
#     mm_2 = torch.ops.aten.mm.default(e_1, f_1);  e_1 = f_1 = None
#     repeat = torch.ops.aten.repeat.default(mm_2, [2, 1]);  mm_2 = None
#     wait_comm = torch.distributed._spmd.comm_tensor._wait_comm(getitem_1);  getitem_1 = None
#     add = torch.ops.aten.add.Tensor(wait_comm, repeat);  wait_comm = repeat = None
#     return add
