import torch

import torch._dynamo
import torch._dynamo.config
import torch._inductor.config as config
from triton.ops.matmul import matmul
from benchmark_helper import time_with_torch_timer
config.max_autotune_gemm = True

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
# The flag below controls whether to allow GROUP_M to be 4 for inductor GEMMs.
config.matmul_allow_group_m_of_4 = True

@torch._dynamo.optimize("inductor", nopython=True)
def inductor_aten_mm(a, b):
    return torch.mm(a, b)


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_mm(a, b):
    return torch.mm(a, b)

@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_bp_mm(a, b):
    return torch.mm(a, b)



def torch_mm(a, b):
    return torch.mm(a, b)


def triton_mm(a, b):
    return matmul(a, b)


def test_total_time(shapes):
    print("shape; torch mm; triton mm; inductor aten mm; inductor triton mm; inductor triton bp mm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

        config.max_autotune_gemm_backends = "aten"
        inductor_aten_mm(a, b)

        config.max_autotune_gemm_backends = "triton"
        inductor_triton_mm(a, b)

        config.use_block_pointer_mm_kernel = True
        inductor_triton_bp_mm(a,b)


        torch_ms = time_with_torch_timer(torch_mm, (a, b)).mean * 1000

        triton_ms = time_with_torch_timer(triton_mm, (a, b)).mean * 1000

        config.max_autotune_gemm_backends = "aten"
        ind_aten_ms = time_with_torch_timer(inductor_aten_mm, (a, b)).mean * 1000

        config.max_autotune_gemm_backends = "triton"
        ind_triton_ms = time_with_torch_timer(inductor_triton_mm, (a, b)).mean * 1000
        ind_triton_bp_ms = time_with_torch_timer(inductor_triton_bp_mm, (a, b)).mean * 1000


        print(torch_ms, triton_ms, ind_aten_ms, ind_triton_ms, ind_triton_bp_ms, sep="; ")

        torch._dynamo.reset()


def test_GPU_time(shapes):
    print("shape; torch mm; triton mm; inductor aten mm; inductor triton mm; inductor triton bp mm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

        config.max_autotune_gemm_backends = "aten"
        inductor_aten_mm(a, b)

        config.max_autotune_gemm_backends = "triton"
        inductor_triton_mm(a, b)

        config.use_block_pointer_mm_kernel = True
        inductor_triton_bp_mm(a,b)

        torch_ms, _, _ = triton.testing.do_bench(lambda: torch_mm(a, b))
        triton_ms, _, _ = triton.testing.do_bench(lambda: triton_mm(a, b))
        ind_aten_ms, _, _ = triton.testing.do_bench(lambda: inductor_aten_mm(a, b))
        ind_triton_ms, _, _ = triton.testing.do_bench(lambda: inductor_triton_mm(a, b))
        ind_triton_bp_ms, _, _ = triton.testing.do_bench(lambda: inductor_triton_bp_mm(a, b))
        print(torch_ms, triton_ms, ind_aten_ms, ind_triton_ms, ind_triton_bp_ms, sep="; ")

        torch._dynamo.reset()


if __name__ == "__main__":
    shapes = [
        # alexnet
        ([128, 9216], [9216, 4096]),
        ([128, 4096], [4096, 4096]),
        ([128, 4096], [4096, 1000]),
        # BERT
        ([2048, 768], [768, 768]),
        ([2048, 768], [768, 3072]),
        ([2048, 3072], [3072, 768]),
        # hf_GPT2
        ([1024, 768], [768, 768]),
        ([1024, 768], [768, 3072]),
        ([1024, 3072], [3072, 768]),
        ([1024, 768], [768, 2304]),
    ]
    print("test total time")
    test_total_time(shapes)

    print("test GPU time")
    test_GPU_time(shapes)


# Results Preview on AWS AI cluster
"""
"""
