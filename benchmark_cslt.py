import time

import torch

# import xformers.ops as xops
# import xformers.ops.sp24_fairinternal as sp24


def benchmark_fn(name, fn, *args, **kwargs):
    g = torch.cuda.CUDAGraph()
    fn(*args, **kwargs)
    with torch.cuda.graph(g):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    begin = time.time()
    reps = 100
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    dt = time.time() - begin
    dt_us = int(dt * 1000000) / reps
    print(f"{name}:", dt_us, "us")
    return dt_us


print("CuSparseLt v0.5.0")
total_sp24 = 0
total_dense = 0
for count, input_shape, weight_shape in [
    (32, [56, 11008], [4096, 11008]),
    (32, [56, 4096], [12288, 4096]),
    (32, [56, 4096], [22016, 4096]),
    (32, [56, 4096], [4096, 4096]),
    (1, [56, 4096], [65536, 4096]),
    (4, [8, 11008], [4096, 11008]),
    (5, [8, 4096], [12288, 4096]),
    (4, [8, 4096], [22016, 4096]),
    (5, [8, 4096], [4096, 4096]),
]:
    input = torch.randn(input_shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(weight_shape, device="cuda", dtype=torch.float16)
    weight_sp24 = torch._cslt_compress(weight)
    optimal_alg_id = torch._cslt_sparse_mm_search(
        weight_sp24, input.T, transpose_result=True
    )
    print(
        f"input: {list(input.shape)} | weight: {list(weight.shape)} | optimal_alg_id: {optimal_alg_id}"
    )
    total_sp24 += count * benchmark_fn(
        "  gemm@sp24",
        torch._cslt_sparse_mm,
        weight_sp24,
        input.T,
        transpose_result=True,
        alg_id=optimal_alg_id,
    )
    total_dense += count * benchmark_fn("  gemm     ", torch.mm, weight, input.T)

print("Total time:")
print(f"  sp24 : {int(total_sp24)}us")
print(f"  dense: {int(total_dense)}us")
