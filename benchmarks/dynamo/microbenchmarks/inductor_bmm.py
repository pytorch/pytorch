from benchmark_helper import time_with_torch_timer

import torch

import torch._dynamo
import torch._dynamo.config
import torch._inductor.config as config


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_aten_bmm(a, b):
    return torch.bmm(a, b)


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_bmm(a, b):
    return torch.bmm(a, b)


def torch_bmm(a, b):
    return torch.bmm(a, b)


def test_total_time(shapes):
    print("shape; torch bmm; inductor aten bmm; inductor triton bmm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

        config.triton.use_bmm = False
        inductor_aten_bmm(a, b)

        config.triton.use_bmm = True
        inductor_triton_bmm(a, b)

        torch_ms = time_with_torch_timer(torch_bmm, (a, b)).mean * 1000

        config.triton.use_bmm = False
        ind_aten_ms = time_with_torch_timer(inductor_aten_bmm, (a, b)).mean * 1000

        config.triton.use_bmm = True
        ind_triton_ms = time_with_torch_timer(inductor_triton_bmm, (a, b)).mean * 1000

        print(torch_ms, ind_aten_ms, ind_triton_ms, sep="; ")


if __name__ == "__main__":
    shapes = [
        # BERT (all)
        ([192, 128, 64], [192, 64, 128]),
        ([192, 128, 128], [192, 128, 64]),
        # hf_GPT2 (all)
        ([12, 1024, 1024], [12, 1024, 64]),
        ([12, 1024, 64], [12, 64, 1024]),
        # hf_Albert (all)
        ([12, 512, 64], [12, 64, 512]),
        ([12, 512, 512], [12, 512, 64]),
    ]

    test_total_time(shapes)
