// Copyright (c) 2022, Tri Dao.

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/transformers/cuda/flash_attn/static_switch.h>
#include <ATen/native/transformers/cuda/flash_attn/flash.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_bwd_kernel.h>

namespace pytorch_flash {

template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(Flash_bwd_params params) {
    compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_clear_dkvaccum_kernel(Flash_bwd_params params) {
    clear_dKVaccum<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_M, bool Is_even_K>
__global__ void flash_bwd_dq_dk_dv_loop_kernel(Flash_bwd_params params) {
    compute_dq_dk_dv<Kernel_traits, Is_dropout, Is_causal, Is_even_M, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_MN, bool Is_even_K>
__global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel(Flash_bwd_params params) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Is_even_MN, Is_even_K>(params);
    #else
        printf("FATAL: FlashAttention requires to be build with sm80-sm90, but was built for < 5.3!");
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_N, bool Is_even_K>
__global__ void flash_bwd_dq_dk_dv_loop_seqq_parallel_kernel(Flash_bwd_params params) {
    compute_dq_dk_dv_seqq_parallel<Kernel_traits, Is_dropout, Is_causal, Is_even_N, Is_even_K>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(Flash_bwd_params params) {
    convert_dQ<Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_kernel(Flash_bwd_params params) {
    convert_dKV<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    dim3 grid_n(num_n_block, params.b, params.h);

    flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    // printf("smem_size_dq_dk_dv = %d\n", smem_size_dq_dk_dv);
    BOOL_SWITCH(params.is_causal, IsCausalConst, [&] {
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
                auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout, IsCausalConst, IsEvenMNConst, IsEvenKConst>;
                // auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout, IsCausalConst, IsEvenMNConst, true>;
                if (smem_size_dq_dk_dv >= 48 * 1024)  {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd_seqq_parallel(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    dim3 grid_n(num_n_block, params.b, params.h_k);
    flash_bwd_clear_dkvaccum_kernel<Kernel_traits><<<grid_n, Kernel_traits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    // We also use is_even_N to set Unpadded in the BlockInfo constructor, so we need to check
    // for cu_seqlens_k as well.
    const bool is_even_N = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1rowblock;
    // printf("smem_size_dq_dk_dv = %d\n", smem_size_dq_dk_dv);
    BOOL_SWITCH(params.is_causal, IsCausalConst, [&] {
        BOOL_SWITCH(is_even_N, IsEvenNConst, [&] {
            BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
                auto kernel = &flash_bwd_dq_dk_dv_loop_seqq_parallel_kernel<Kernel_traits, Is_dropout, IsCausalConst, IsEvenNConst, IsEvenKConst>;
                // auto kernel = &flash_bwd_dq_dk_dv_loop_seqq_parallel_kernel<Kernel_traits, false, false, IsEvenNConst, IsEvenKConst>;
                if (smem_size_dq_dk_dv >= 48 * 1024)  {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                }
                kernel<<<grid_m, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });

    auto kernel_dkv = &flash_bwd_convert_dkv_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemKVSize >= 48 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dkv, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemKVSize));
    }
    kernel_dkv<<<grid_n, Kernel_traits::kNThreads, Kernel_traits::kSmemKVSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
//

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    if (configure) return;
    // dim3 grid(params.b, params.h);
    // const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    // dim3 grid_m(num_m_block, params.b, params.h);

    // if (params.h == params.h_k) {  // No multi-query or grouped-query attention (MQA/GQA)
        run_flash_bwd_seqk_parallel<Kernel_traits, Is_dropout>(params, stream, configure);
    // } else {
        // run_flash_bwd_seqq_parallel<Kernel_traits, Is_dropout>(params, stream, configure);
    // }

    // // We also use is_even_M to set Unpadded in the BlockInfo constructor, so we need to check
    // // for cu_seqlens_q as well.
    // const bool is_even_M = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0;
    // const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    // constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize;
    // BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
    //     BOOL_SWITCH(params.is_causal, IsCausalConst, [&] {
    //         BOOL_SWITCH(is_even_M, IsEvenMConst, [&] {
    //             BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
    //                 // auto kernel = &flash_bwd_dq_dk_dv_loop_kernel<Kernel_traits, Is_dropout, IsCausalConst>;
    //                 auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout, IsCausalConst, IsEvenMConst, IsEvenKConst>;
    //                 if (smem_size_dq_dk_dv >= 48 * 1024)  {
    //                     C10_CUDA_CHECK(cudaFuncSetAttribute(
    //                         kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
    //                 }
    //                 kernel<<<grid_m, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
    //                 C10_CUDA_KERNEL_LAUNCH_CHECK();
    //             });
    //         });
    //     });
    // });

    // auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    // if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
    //     C10_CUDA_CHECK(cudaFuncSetAttribute(
    //         kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    // }
    // kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params);
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
}
//

template<typename T>
void run_mha_bwd_hdim32(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 32;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
            if constexpr(!Is_dropout) {  // We can afford more registers to keep V in registers
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream, configure);
            } else {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
            }
        } else {  // 96 KB
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream, configure);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // Changing AtomLayoutMdQ from 2 to 4 takes the same time
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>>(params, stream, configure);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, true, false, T>>(params, stream, configure);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 2, 4, 4, false, false, T>>(params, stream, configure);
        // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream, configure);
        // This is slightly faster. We want to split M more so we need fewer registers to store LSE.
        if (max_smem_per_block >= 144 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
            // This has a lot of register spilling
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream, configure);
        } else {
            // if (params.h == params.h_k) {
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>, Is_dropout>(params, stream, configure);
            // } else {
            //     run_flash_bwd_seqq_parallel<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout>(params, stream, configure);
            // }
        }
    });
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, 2, 2, 2, true, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 128, 4, 1, 4, 1, false, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 16, 128, 4, 1, 4, 1, false, false, T>>(params, stream, configure);
    // M=128, N=64 is quite slow, I think because we need to read/write dQaccum twice as many times
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 2, 2, 2, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, false, T>>(params, stream, configure);
    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 4, false, T>>(params, stream, configure);

    // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 4, 4, 2, 4, false, false, T>>(params, stream, configure);
}

template<typename T>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 96;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (params.h == params.h_k) {
            if (max_smem_per_block >= 116 * 1024) {
                if constexpr(!Is_dropout) {  // 92KB
                    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream, configure);
                } else {  // 116 KB
                    // This is faster for dropout since we don't have many registers to spare
                    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
                }
            } else {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream, configure);
            }
        // } else {
            // run_flash_bwd_seqq_parallel<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 4, 4, false, false, T>>(params, stream, configure);
        // }
    });
}

template<typename T>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // if (params.h == params.h_k) {
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 128, 8, 2, 2, 2, false, false, T>>(params, stream, configure);
            // This is faster, in the case of sequence-parallel bwd (where we need fewer registers).
            // Out of these three, the 2nd one is slightly faster (2% faster than the first). Idk why.
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 2, 2, false, false, T>>(params, stream, configure);
            if (max_smem_per_block >= 144 * 1024) {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd_seqk_parallel<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd_seqk_parallel<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, true, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd_seqq_parallel<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, true, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream, configure);
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 2, true, false, T>, Is_dropout>(params, stream, configure);
            } else {
                // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream, configure);
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>, Is_dropout>(params, stream, configure);
            }
            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>>(params, stream, configure);

            // run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 4, 4, false, false, T>>(params, stream, configure);
        // } else {
            // run_flash_bwd_seqq_parallel<Flash_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 4, 4, false, false, T>>(params, stream, configure);
        // }
    });
}

template<typename T>
void run_mha_bwd_hdim160(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 160;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 116 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, true, T>, Is_dropout>(params, stream, configure);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 192;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 136 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream, configure);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>, Is_dropout>(params, stream, configure);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim224(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 224;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream, configure);
    });
}

template<typename T>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 176 * 1024) {  // H100
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream, configure);
        } else {  // A100, we don't do double buffering to save smem
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_dropout>(params, stream, configure);
        }
    });
}


}; // namespace pytorch_fmha
