import heapq

import torch


def estimate_conv_time(
    # backend, device,
    num_warps,
    num_stages,
    x,
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    BLOCK_M,
    BLOCK_K,
    BLOCK_N,
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
    dtype = x.dtype
    dtsize = x.element_size()

    M = BATCH * OUT_H * OUT_W
    N = KERNEL_N
    K = KERNEL_H * KERNEL_W * IN_C
    num_cta_m = triton.cdiv(M, BLOCK_M)
    num_cta_n = triton.cdiv(N, BLOCK_N)
    num_cta_k = 1
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
    store_c_dram = M * N * dtsize / (1024 * 1024)  # MB
    store_ms = store_c_dram / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms
    if debug:
        print(
            f"Total time: {total_time_ms}ms, compute time: {compute_ms}ms, "
            f"loading time: {load_ms}ms, store time: {store_ms}ms, "
            f"Activate CTAs: {active_cta_ratio*100}%"
        )
    return total_time_ms


def early_config_prune(configs, named_args):
    import triton._C.libtriton.triton as _triton

    backend = _triton.runtime.backend.CUDA
    device = torch.cuda.current_device()
    cc = _triton.runtime.cc(backend, device)
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    dtsize = named_args["x"].element_size()
    # dtype = named_args["x"].dtype

    # 1. make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_stages,
        )
        max_shared_memory = _triton.runtime.max_shared_memory(backend, device)
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs

    # group configs by (BLOCK_M,_N,_K, num_warps)
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_warps,
            config.num_stages,
        )

        key = (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]

    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps = k
        if cc >= 80:
            # compute cycles (only works for ampere GPUs)
            mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8

            ldgsts_latency = 300  # Does this matter?
            optimal_num_stages = ldgsts_latency / mma_cycles

            # nearest stages, prefer large #stages
            nearest = heapq.nsmallest(
                2,
                v,
                key=lambda x: 10 + abs(x[1] - optimal_num_stages)
                if (x[1] - optimal_num_stages) < 0
                else x[1] - optimal_num_stages,
            )

            for n in nearest:
                pruned_configs.append(n[0])
        else:  # Volta & Turing only supports num_stages <= 2
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs
