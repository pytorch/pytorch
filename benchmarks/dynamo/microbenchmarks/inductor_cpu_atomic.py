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
[1, 4096]; torch.float32; 0.0052365999999892665; 0.017564740001034806; 0.013978460001453641; 0.024368539998249616
[1, 4096]; torch.int32; 0.005814169999212027; 0.016935479998210212; 0.005414689994722721; 0.017164559994853335
[1, 65536]; torch.float32; 0.1297304199943028; 0.13822156000060204; 0.24826899999425223; 3.4930834800070443
[1, 65536]; torch.int32; 0.126586229998793; 0.1042632999997295; 0.10594573999696877; 1.0463533599977382
"""
