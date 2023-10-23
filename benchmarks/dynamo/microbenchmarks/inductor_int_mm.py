import torch
import triton
import torch._dynamo
import torch._dynamo.config
import torch._inductor.config as config
from triton.ops.matmul import (matmul, _matmul)
from benchmark_helper import time_with_torch_timer
config.coordinate_descent_tuning = True
config.coordinate_descent_check_all_directions = True
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
    for (key, item) in _matmul.kernel.cache.items():
        print(key, item)
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

        print("test GPU time")
        test_GPU_time(shapes)


# Results Preview on AWS AI cluster
"""
---------------------------------------WORST---------------------------------------
test total time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[78400, 3840] x [3840, 1280]|2.8103|1.6544|2.8128|1.7828|1.7566|
|[78400, 1280] x [1280, 1280]|1.0351|0.6128|1.0360|0.7081|0.6888|
|[65536, 5120] x [5120, 1280]|3.1008|1.8279|3.1213|1.9045|1.9116|
|[65536, 1280] x [1280, 5120]|2.6623|2.0272|2.6378|2.4017|2.4073|
test GPU time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[78400, 3840] x [3840, 1280]|2.8413|1.6765|2.8672|1.7841|1.7865|
|[78400, 1280] x [1280, 1280]|1.0611|0.6202|1.0442|0.7261|0.7172|
|[65536, 5120] x [5120, 1280]|3.1531|1.8451|3.1657|1.9496|1.9466|
|[65536, 1280] x [1280, 5120]|2.7056|2.0416|2.6854|2.3970|2.4349|
---------------------------------------NEW CONFIGS, NO BP BRANCH, GROUP_M 4, NO COORD_DESC---------------------------------------
test total time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[78400, 3840] x [3840, 1280]|2.8245|1.6619|2.8141|1.7536|1.7524|
|[78400, 1280] x [1280, 1280]|1.0218|0.6109|1.0341|0.6977|0.6929|
|[65536, 5120] x [5120, 1280]|3.1039|1.8319|3.1220|1.8981|1.8963|
|[65536, 1280] x [1280, 5120]|2.6539|2.0095|2.6515|2.3723|2.3612|
test GPU time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[78400, 3840] x [3840, 1280]|2.7450|1.6660|2.8623|1.7786|1.7829|
|[78400, 1280] x [1280, 1280]|1.0656|0.6223|1.0510|0.7029|0.7046|
|[65536, 5120] x [5120, 1280]|3.1542|1.8781|3.1473|1.9223|1.9042|
|[65536, 1280] x [1280, 5120]|2.6847|2.1089|2.6759|2.4057|2.3827|
---------------------------------------BEST--------------------------------------
test total time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[78400, 3840] x [3840, 1280]|2.8240|1.6727|2.8278|1.7537|1.7619|
|[78400, 1280] x [1280, 1280]|1.0632|0.6032|1.0367|0.6984|0.6945|
|[65536, 5120] x [5120, 1280]|3.1099|1.8358|3.0890|1.9131|1.9114|
|[65536, 1280] x [1280, 5120]|2.6537|2.0301|2.6527|2.3573|2.3718|
test GPU time
| shape | torch int_mm | triton matmul | inductor aten int_mm | inductor triton int_mm | inductor triton bp int_mm |
|--------------------------|------|------|------|------|------|
|[78400, 3840] x [3840, 1280]|2.8312|1.6641|2.8678|1.7451|1.7754|
|[78400, 1280] x [1280, 1280]|1.0573|0.6212|1.0472|0.7091|0.6916|
|[65536, 5120] x [5120, 1280]|3.1163|1.8380|3.1533|1.9126|1.9305|
|[65536, 1280] x [1280, 5120]|2.6471|2.0384|2.6961|2.3507|2.3852|

triton matmul configs:

(78400, 1280, 3840) BLOCK_M: 128, BLOCK_N: 256, BLOCK_K: 128, SPLIT_K: 1, num_warps: 8, num_ctas: 1, num_stages: 3, enable_warp_specialization: False, enable_persistent: False
(78400, 1280, 1280) BLOCK_M: 128, BLOCK_N: 256, BLOCK_K: 128, SPLIT_K: 1, num_warps: 8, num_ctas: 1, num_stages: 3, enable_warp_specialization: False, enable_persistent: False
(65536, 1280, 5120) BLOCK_M: 128, BLOCK_N: 256, BLOCK_K: 128, SPLIT_K: 1, num_warps: 8, num_ctas: 1, num_stages: 3, enable_warp_specialization: False, enable_persistent: False
(65536, 5120, 1280) BLOCK_M: 128, BLOCK_N: 256, BLOCK_K: 128, SPLIT_K: 1, num_warps: 8, num_ctas: 1, num_stages: 3, enable_warp_specialization: False, enable_persistent: False

"""
