// Copyright (c) 2022, Tri Dao.

#pragma once

#include <vector>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/transformers/cuda/flash_attn/fmha.h>
#include <ATen/native/transformers/cuda/flash_attn/static_switch.h>
#include <ATen/native/transformers/cuda/flash_attn/fmha_fprop_kernel_1xN.h>

namespace pytorch_fmha {

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 95%
// of the best efficiency.
// [2022-11-25] TD: Mark this as "inline" otherwise we get "multiple definition" error.
inline int num_splits_heuristic_fwd(int batch_nheads, int num_SMs, int ctas_per_sm, int max_splits) {
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        float n_waves = float(batch_nheads * num_splits) / (num_SMs * ctas_per_sm);
        float eff = n_waves / ceil(n_waves);
        // printf("num_splits = %d, eff = %f\n", num_splits, eff);
        if (eff > max_efficiency) { max_efficiency = eff; }
        efficiency.push_back(eff);
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (efficiency[num_splits - 1] > 0.95 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax>
__global__ void fmha_fwd_loop_kernel(FMHA_fprop_params params) {
    fmha::device_1xN_loop<Kernel_traits, Is_dropout, Is_causal, Return_softmax>(params);
}

template<typename Kernel_traits>
void run_fmha_fwd_loop(Launch_params<FMHA_fprop_params> &launch_params) {
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    const int loop_steps = (launch_params.params.seqlen_k + blocksize_c - 1) / blocksize_c;

    constexpr int smem_size_softmax_lse = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;
    // Don't need smem_size_softmax_lse if we're not looping
    const int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>()
        + (loop_steps > 1 ? smem_size_softmax_lse : 0);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21
    BOOL_SWITCH(launch_params.is_dropout, IsDropoutConst, ([&] {
        auto kernel = launch_params.params.is_causal
            ? (launch_params.return_softmax
               ? &fmha_fwd_loop_kernel<Kernel_traits, IsDropoutConst, true, true>
               : &fmha_fwd_loop_kernel<Kernel_traits, IsDropoutConst, true, false>)
            : (launch_params.return_softmax
               ? &fmha_fwd_loop_kernel<Kernel_traits, IsDropoutConst, false, true>
               : &fmha_fwd_loop_kernel<Kernel_traits, IsDropoutConst, false, false>);
        if( smem_size >= 48 * 1024 ) {
            FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        // Automatically set num_splits to maximize occupancy
        if (launch_params.params.num_splits <= 0) {
            int ctas_per_sm;
            cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size);
            auto dprops = at::cuda::getCurrentDeviceProperties();
            // printf("CTAS_PER_SM = %d, nSMs = %d\n", ctas_per_sm, dprops->multiProcessorCount);
            constexpr int M = Kernel_traits::Cta_tile_p::M;
            launch_params.params.num_splits = num_splits_heuristic_fwd(
                launch_params.params.b * launch_params.params.h, dprops->multiProcessorCount,
                ctas_per_sm,
                /*max_splits=*/std::min(30, (launch_params.params.seqlen_q + M - 1 / M))
            );
        }
        // printf("smem_size = %d\n", smem_size);
        dim3 grid(launch_params.params.b, launch_params.params.h, launch_params.params.num_splits);
        kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
            launch_params.params);
        FMHA_CHECK_CUDA(cudaPeekAtLastError());
    }));
}

}; // namespace pytorch_fmha
