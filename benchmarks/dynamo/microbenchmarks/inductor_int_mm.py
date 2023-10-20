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
config.matmul_allow_group_m_of_4 = True



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
        # SAM vit_h
        # ([78400, 3840], [3840, 1280]),
        # ([78400, 1280], [1280, 1280]),
        # ([65536, 5120], [5120, 1280]),
        # ([65536, 1280], [1280, 5120]),
    ]
    with torch.no_grad():
        print("test total time")
        test_total_time(shapes)

        print("test GPU time")
        test_GPU_time(shapes)


# Results Preview on AWS AI cluster
"""
---------------------------------------ALLOW GROUP_M=4---------------------------------------
test GPU time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[128, 9216] x [9216, 4096]|0.0692|0.0653|0.0696|0.0743|0.0743|
|[128, 4096] x [4096, 4096]|0.0335|0.0331|0.0339|0.0377|0.0381|
|[128, 4096] x [4096, 1000]|0.0406|0.0261|0.0406|0.0356|0.0352|
|[2048, 768] x [768, 768]|0.0210|0.0150|0.0210|0.0161|0.0166|
|[2048, 768] x [768, 3072]|0.0448|0.0341|0.0449|0.0400|0.0404|
|[2048, 3072] x [3072, 768]|0.0450|0.0315|0.0446|0.0317|0.0317|
|[1024, 768] x [768, 768]|0.0154|0.0116|0.0151|0.0146|0.0142|
|[1024, 768] x [768, 3072]|0.0264|0.0206|0.0264|0.0239|0.0239|
|[1024, 3072] x [3072, 768]|0.0322|0.0242|0.0324|0.0290|0.0294|
|[1024, 768] x [768, 2304]|0.0255|0.0199|0.0252|0.0212|0.0208|

---------------------------------------WITHOUT GROUP_M=4---------------------------------------
test GPU time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[128, 9216] x [9216, 4096]|0.0694|0.0661|0.0693|0.0744|0.0744|
|[128, 4096] x [4096, 4096]|0.0336|0.0334|0.0337|0.0378|0.0378|
|[128, 4096] x [4096, 1000]|0.0409|0.0261|0.0409|0.0356|0.0352|
|[2048, 768] x [768, 768]|0.0211|0.0150|0.0210|0.0164|0.0160|
|[2048, 768] x [768, 3072]|0.0443|0.0341|0.0448|0.0405|0.0405|
|[2048, 3072] x [3072, 768]|0.0441|0.0310|0.0440|0.0316|0.0312|
|[1024, 768] x [768, 768]|0.0151|0.0118|0.0155|0.0147|0.0143|
|[1024, 768] x [768, 3072]|0.0264|0.0206|0.0264|0.0236|0.0237|
|[1024, 3072] x [3072, 768]|0.0326|0.0242|0.0322|0.0292|0.0292|
|[1024, 768] x [768, 2304]|0.0250|0.0198|0.0250|0.0208|0.0212|

---------------------------------------WITHOUT GROUP_M=4 and without bp kernel branches---------------------------------------
test GPU time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[128, 9216] x [9216, 4096]|0.0694|0.0664|0.0694|0.0748|0.0747|
|[128, 4096] x [4096, 4096]|0.0335|0.0330|0.0337|0.0381|0.0377|
|[128, 4096] x [4096, 1000]|0.0405|0.0264|0.0408|0.0355|0.0355|
|[2048, 768] x [768, 768]|0.0211|0.0153|0.0216|0.0164|0.0160|
|[2048, 768] x [768, 3072]|0.0441|0.0341|0.0441|0.0403|0.0403|
|[2048, 3072] x [3072, 768]|0.0446|0.0319|0.0445|0.0312|0.0316|
|[1024, 768] x [768, 768]|0.0154|0.0152|0.0154|0.0147|0.0147|
|[1024, 768] x [768, 3072]|0.0261|0.0203|0.0265|0.0240|0.0236|
|[1024, 3072] x [3072, 768]|0.0323|0.0245|0.0323|0.0296|0.0296|
|[1024, 768] x [768, 2304]|0.0255|0.0198|0.0255|0.0212|0.0213|


---------------------------------------WITHOUT GROUP_M=4 and without new configs---------------------------------------

test GPU time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[128, 9216] x [9216, 4096]|0.0693|0.0660|0.0697|0.0743|0.0743|
|[128, 4096] x [4096, 4096]|0.0341|0.0335|0.0340|0.0381|0.0377|
|[128, 4096] x [4096, 1000]|0.0408|0.0260|0.0404|0.0353|0.0352|
|[2048, 768] x [768, 768]|0.0210|0.0154|0.0210|0.0212|0.0208|
|[2048, 768] x [768, 3072]|0.0443|0.0345|0.0443|0.0452|0.0448|
|[2048, 3072] x [3072, 768]|0.0441|0.0313|0.0446|0.0499|0.0503|
|[1024, 768] x [768, 768]|0.0154|0.0117|0.0150|0.0142|0.0142|
|[1024, 768] x [768, 3072]|0.0265|0.0205|0.0261|0.0263|0.0263|
|[1024, 3072] x [3072, 768]|0.0321|0.0246|0.0321|0.0325|0.0321|
|[1024, 768] x [768, 2304]|0.0250|0.0195|0.0250|0.0238|0.0239|

---------------------------------------ALLOW GROUP_M=4 but without new configs---------------------------------------

test GPU time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[128, 9216] x [9216, 4096]|0.0696|0.0666|0.0697|0.0747|0.0747|
|[128, 4096] x [4096, 4096]|0.0337|0.0331|0.0341|0.0381|0.0377|
|[128, 4096] x [4096, 1000]|0.0409|0.0264|0.0408|0.0356|0.0356|
|[2048, 768] x [768, 768]|0.0211|0.0154|0.0210|0.0214|0.0210|
|[2048, 768] x [768, 3072]|0.0445|0.0345|0.0447|0.0450|0.0446|
|[2048, 3072] x [3072, 768]|0.0442|0.0310|0.0442|0.0500|0.0499|
|[1024, 768] x [768, 768]|0.0154|0.0120|0.0154|0.0142|0.0146|
|[1024, 768] x [768, 3072]|0.0265|0.0202|0.0265|0.0256|0.0256|
|[1024, 3072] x [3072, 768]|0.0321|0.0244|0.0321|0.0325|0.0321|
|[1024, 768] x [768, 2304]|0.0254|0.0200|0.0250|0.0234|0.0238|

"""
