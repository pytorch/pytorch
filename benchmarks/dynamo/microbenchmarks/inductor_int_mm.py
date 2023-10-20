import torch
import triton
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
# config.matmul_allow_group_m_of_4 = True



@torch._dynamo.optimize("inductor", nopython=True)
def inductor_aten_mm(a, b):
    return torch._int_mm(a, b)


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_mm(a, b):
    return torch._int_mm(a, b)

@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_bp_mm(a, b):
    return torch._int_mm(a, b)



def torch_mm(a, b):
    return torch._int_mm(a, b)


def triton_mm(a, b):
    return matmul(a, b)


def test_total_time(shapes):
    print("| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |")
    print("|--------------------------|------|------|------|------|------|")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        a = torch.randint(-128, 127, a_shape, device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 127, b_shape, device="cuda", dtype=a.dtype).t().contiguous().t()

        torch_ms = time_with_torch_timer(torch_mm, (a, b)).median * 1000
        triton_ms = time_with_torch_timer(triton_mm, (a, b)).median * 1000

        config.max_autotune_gemm_backends = "aten"
        inductor_aten_mm(a, b)
        inductor_aten_mm(a, b)
        inductor_aten_mm(a, b)
        ind_aten_ms = time_with_torch_timer(inductor_aten_mm, (a, b)).median * 1000

        config.max_autotune_gemm_backends = "triton"
        inductor_triton_mm(a, b)
        inductor_triton_mm
        inductor_triton_mm
        ind_triton_ms = time_with_torch_timer(inductor_triton_mm, (a, b)).median * 1000

        config.use_block_pointer_mm_kernel = True
        inductor_triton_bp_mm(a,b)
        inductor_triton_bp_mm(a,b)
        inductor_triton_bp_mm(a,b)
        ind_triton_bp_ms = time_with_torch_timer(inductor_triton_bp_mm, (a, b)).median * 1000

        print(f"|{a_shape} x {b_shape}|{torch_ms:.4f}|{triton_ms:.4f}|{ind_aten_ms:.4f}|{ind_triton_ms:.4f}|{ind_triton_bp_ms:.4f}|")
        torch._dynamo.reset()


def test_GPU_time(shapes):
    print("| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |")
    print("|--------------------------|------|------|------|------|------|")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]

        a = torch.randint(-128, 127, a_shape, device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 127, b_shape, device="cuda", dtype=a.dtype).t().contiguous().t()

        torch_ms = triton.testing.do_bench(lambda: torch_mm(a, b))
        triton_ms = triton.testing.do_bench(lambda: triton_mm(a, b))

        config.max_autotune_gemm_backends = "aten"
        inductor_aten_mm(a, b)
        inductor_aten_mm(a, b)
        inductor_aten_mm(a, b)
        ind_aten_ms = triton.testing.do_bench(lambda: inductor_aten_mm(a, b))

        config.max_autotune_gemm_backends = "triton"
        inductor_triton_mm(a, b)
        inductor_triton_mm(a, b)
        inductor_triton_mm(a, b)
        ind_triton_ms = triton.testing.do_bench(lambda: inductor_triton_mm(a, b))

        config.use_block_pointer_mm_kernel = True
        inductor_triton_bp_mm(a,b)
        inductor_triton_bp_mm(a,b)
        inductor_triton_bp_mm(a,b)
        ind_triton_bp_ms = triton.testing.do_bench(lambda: inductor_triton_bp_mm(a, b))

        print(f"|{a_shape} x {b_shape}|{torch_ms:.4f}|{triton_ms:.4f}|{ind_aten_ms:.4f}|{ind_triton_ms:.4f}|{ind_triton_bp_ms:.4f}|")

        torch._dynamo.reset()


if __name__ == "__main__":
    shapes = [
        # # alexnet
        # ([128, 9216], [9216, 4096]),
        # ([128, 4096], [4096, 4096]),
        # ([128, 4096], [4096, 1000]),
        # # BERT
        # ([2048, 768], [768, 768]),
        # ([2048, 768], [768, 3072]),
        # ([2048, 3072], [3072, 768]),
        # # hf_GPT2
        # ([1024, 768], [768, 768]),
        # ([1024, 768], [768, 3072]),
        # ([1024, 3072], [3072, 768]),
        # ([1024, 768], [768, 2304]),
        # SAM vit_h
        ([78400, 3840], [3840, 1280]),
        ([78400, 1280], [1280, 1280]),
        ([65536, 5120], [5120, 1280]),
        ([65536, 1280], [1280, 5120]),
    ]
    with torch.no_grad():
        print("test total time")
        test_total_time(shapes)

        # print("test GPU time")
        # test_GPU_time(shapes)


# Results Preview on AWS AI cluster
"""

---------------------------------------WITHOUT GROUP_M=4 and without new configs---------------------------------------
test total time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[78400, 3840] x [3840, 1280]|2.8457|1.6649|2.8437|1.7955|1.7744|
|[78400, 1280] x [1280, 1280]|1.0240|0.6167|1.0396|0.6918|0.7095|
|[65536, 5120] x [5120, 1280]|3.1365|1.8394|3.1267|1.9237|1.9252|
|[65536, 1280] x [1280, 5120]|2.6787|2.0476|2.6698|2.4131|2.4222|
---------------------------------------WITHOUT GROUP_M=4---------------------------------------

---------------------------------------WITHOUT GROUP_M=4 and without bp kernel branches---------------------------------------



---------------------------------------ALLOW GROUP_M=4---------------------------------------






---------------------------------------ALLOW GROUP_M=4 but without new configs---------------------------------------

"""
