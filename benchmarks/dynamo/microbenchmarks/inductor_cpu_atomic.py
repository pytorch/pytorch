import itertools

import torch
import torch._dynamo
from benchmark_helper import time_with_torch_timer


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_scatter_add(dst, src, index):
    return torch.scatter_add(dst, 1, index, src)


def torch_scatter_add(dst, src, index):
    return torch.scatter_add(dst, 1, index, src)


def test_total_time(shapes, types):
    print(
        "shape; type; torch scatter_add; inductor scatter_add; torch scatter_add (worst case); inductor scatter_add (worst case)"
    )
    for shape, dtype in itertools.product(shapes, types):
        print(shape, dtype, sep="; ", end="; ")

        torch.manual_seed(1)
        if dtype.is_floating_point:
            src = torch.randn(shape, device="cpu", dtype=dtype)
            dst = torch.randn(shape, device="cpu", dtype=dtype)
        else:
            src = torch.randint(0, shape[1], shape, device="cpu", dtype=dtype)
            dst = torch.randint(0, shape[1], shape, device="cpu", dtype=dtype)
        index = torch.randint(0, shape[1], shape, device="cpu", dtype=torch.int64)
        worst_index = torch.tensor([[0] * shape[1]], device="cpu", dtype=torch.int64)

        torch_result = torch_scatter_add(dst, src, index)
        inductor_result = inductor_scatter_add(dst, src, index)
        torch.testing.assert_close(torch_result, inductor_result)

        torch_ms = (
            time_with_torch_timer(torch_scatter_add, (dst, src, index)).mean * 1000
        )
        inductor_ms = (
            time_with_torch_timer(inductor_scatter_add, (dst, src, index)).mean * 1000
        )
        torch_worst_ms = (
            time_with_torch_timer(torch_scatter_add, (dst, src, worst_index)).mean
            * 1000
        )
        inductor_worst_ms = (
            time_with_torch_timer(inductor_scatter_add, (dst, src, worst_index)).mean
            * 1000
        )

        print(torch_ms, inductor_ms, torch_worst_ms, inductor_worst_ms, sep="; ")

        torch._dynamo.reset()


if __name__ == "__main__":
    shapes = [
        ([1, 4096]),
        ([1, 65536]),
    ]
    types = [
        torch.float32,
        torch.int32,
    ]
    print("test total time")
    test_total_time(shapes, types)

# Results preview on 5800H
"""
test total time
shape; type; torch scatter_add; inductor scatter_add; torch scatter_add (worst case); inductor scatter_add (worst case)
[1, 4096]; torch.float32; 0.14733232000025964; 0.05388864999986254; 0.1451428800010035; 0.06496850000075938
[1, 4096]; torch.int32; 0.1440268700002889; 0.05882900999949925; 0.1429359899998417; 0.07036211000013282
[1, 65536]; torch.float32; 1.3435545300012564; 0.15207924000151252; 1.2523296799986383; 3.1408327299982375
[1, 65536]; torch.int32; 1.3407247500003905; 0.12999147000073208; 1.2956029100018895; 0.853825209999286
"""
