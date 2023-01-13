import torch
from torch.fx.experimental.proxy_tensor import make_fx
# from torch._decomp import decomposition_table
from torch._subclasses import FakeTensorMode


def addmm(input, mat1, mat2):
    return input + mat1 @ mat2 + 2


# print(mode.from_tensor(torch.randn(3, 8)))
# print(mode.from_tensor(test_tensor))
# with FakeTensorMode():
# input = torch.randn((2^64, 2^64 ))
    # mat1 = torch.randn(3, 8)
    # mat2 = torch.randn(8, 3)

    # compiled_fn = make_fx(addmm, tracing_mode="fake")(input, mat1, mat2)
    # print(compiled_fn.graph)


with FakeTensorMode() as mode:
    input = torch.randn(3, 3)
    mat1 = torch.randn(3, 8)
    mat2 = torch.randn(8, 3)

    compiled_fn = make_fx(addmm)(input, mat1, mat2)
    print(compiled_fn.graph)


for node in compiled_fn.graph.nodes:
    print(node.meta)

# no make_fx
with FakeTensorMode() as mode:
    mat1 = torch.randn(3, 8)
    mat2 = torch.randn(8, 3)
    compiled_fn = make_fx(torch.ops.aten.mm)(mat1, mat2)
    print(type(compiled_fn.graph))
    # print(compiled_fn.graph.nodes.output.meta)
    # res = mat1.mm(mat2)
    # print(res.)
