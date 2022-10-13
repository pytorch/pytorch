import torch


def estimate_matmul_time(
    # backend, device,
    num_warps,
    num_stages,
    A,
    B,
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    SPLIT_K,
    debug=False,
    **kwargs,
):
    """return estimated running time in ms
    = max(compute, loading) + store"""
    import triton
    import triton._C.libtriton.triton as _triton
    from triton.ops.matmul_perf_model import (
        get_dram_gbps as get_dram_gbps,
        get_tflops as get_tflops,
    )

    backend = _triton.runtime.backend.CUDA
    device = torch.cuda.current_device()
    dtype = A.dtype
    dtsize = A.element_size()

    num_cta_m = triton.cdiv(M, BLOCK_M)
    num_cta_n = triton.cdiv(N, BLOCK_N)
    num_cta_k = SPLIT_K
    num_ctas = num_cta_m * num_cta_n * num_cta_k

    # If the input is smaller than the block size
    M, N = max(M, BLOCK_M), max(N, BLOCK_N)

    # time to compute
    total_ops = 2 * M * N * K / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(backend, device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # time to load data
    num_sm = _triton.runtime.num_sm(backend, device)
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(
        1, num_ctas / 32
    )  # 32 active ctas are enough to saturate
    active_cta_ratio_bw2 = max(
        min(1, (num_ctas - 32) / (108 - 32)), 0
    )  # 32-108, remaining 5%
    dram_bw = get_dram_gbps(backend, device) * (
        active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05
    )  # in GB/s
    l2_bw = dram_bw * 4  # rough estimation (should be 4.7 for A100?)
    # assume 80% of (following) loads are in L2 cache
    load_a_dram = M * K * dtsize * (1 + 0.2 * (num_cta_n - 1))
    load_a_l2 = M * K * dtsize * 0.8 * (num_cta_n - 1)
    load_b_dram = N * K * dtsize * (1 + 0.2 * (num_cta_m - 1))
    load_b_l2 = N * K * dtsize * 0.8 * (num_cta_m - 1)
    # total
    total_dram = (load_a_dram + load_b_dram) / (1024 * 1024)  # MB
    total_l2 = (load_a_l2 + load_b_l2) / (1024 * 1024)
    # loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # estimate storing time
    store_bw = dram_bw * 0.6  # :o
    store_c_dram = M * N * dtsize * SPLIT_K / (1024 * 1024)  # MB
    if SPLIT_K == 1:
        store_ms = store_c_dram / store_bw
    else:
        reduce_bw = store_bw
        store_ms = store_c_dram / reduce_bw
        # c.zero_()
        zero_ms = M * N * 2 / (1024 * 1024) / store_bw
        store_ms += zero_ms

    total_time_ms = max(compute_ms, load_ms) + store_ms
    if debug:
        print(
            f"Total time: {total_time_ms}ms, compute time: {compute_ms}ms, "
            f"loading time: {load_ms}ms, store time: {store_ms}ms, "
            f"Activate CTAs: {active_cta_ratio*100}%"
        )
    return total_time_ms
