#ifndef OORT_bwd_kernel_dq_H
#define OORT_bwd_kernel_dq_H

namespace oort {

template<int32_t BLOCK_M,
         int32_t BLOCK_DMODEL,
         int32_t BLOCK_N,
         bool CAUSAL,
         bool ENABLE_DROPOUT>
struct bwd_kernel_dq {

 hipError_t operator()(dim3 grid, dim3 block, const __fp16* Q,
                       const __fp16* K,
                       const __fp16* V,
                       float sm_scale,
                       const __fp16* Out,
                       const __fp16* dO,
                       const __fp16* dQ,
                       const float* L,
                       const float* D,
                       uint64_t stride_qz,
                       uint64_t stride_qh,
                       uint64_t stride_qm,
                       uint64_t stride_qk,
                       uint64_t stride_kz,
                       uint64_t stride_kh,
                       uint64_t stride_kn,
                       uint64_t stride_kk,
                       uint64_t stride_vz,
                       uint64_t stride_vh,
                       uint64_t stride_vk,
                       uint64_t stride_vn,
                       uint64_t Z,
                       uint64_t H,
                       uint64_t seqlen_q,
                       uint64_t seqlen_k,
                       float dropout_p,
                       uint64_t philox_seed,
                       uint32_t philox_offset_base, hipStream_t stream);

 hipError_t operator()(dim3 grid, dim3 block, const __bf16* Q,
                       const __bf16* K,
                       const __bf16* V,
                       float sm_scale,
                       const __bf16* Out,
                       const __bf16* dO,
                       const __bf16* dQ,
                       const float* L,
                       const float* D,
                       uint64_t stride_qz,
                       uint64_t stride_qh,
                       uint64_t stride_qm,
                       uint64_t stride_qk,
                       uint64_t stride_kz,
                       uint64_t stride_kh,
                       uint64_t stride_kn,
                       uint64_t stride_kk,
                       uint64_t stride_vz,
                       uint64_t stride_vh,
                       uint64_t stride_vk,
                       uint64_t stride_vn,
                       uint64_t Z,
                       uint64_t H,
                       uint64_t seqlen_q,
                       uint64_t seqlen_k,
                       float dropout_p,
                       uint64_t philox_seed,
                       uint32_t philox_offset_base, hipStream_t stream);

};


template struct bwd_kernel_dq<16 /* BLOCK_M */,
                              16 /* BLOCK_DMODEL */,
                              16 /* BLOCK_N */,
                              true /* CAUSAL */,
                              false /* ENABLE_DROPOUT */>;
template struct bwd_kernel_dq<16 /* BLOCK_M */,
                              16 /* BLOCK_DMODEL */,
                              16 /* BLOCK_N */,
                              false /* CAUSAL */,
                              true /* ENABLE_DROPOUT */>;
template struct bwd_kernel_dq<16 /* BLOCK_M */,
                              16 /* BLOCK_DMODEL */,
                              16 /* BLOCK_N */,
                              false /* CAUSAL */,
                              false /* ENABLE_DROPOUT */>;
template struct bwd_kernel_dq<16 /* BLOCK_M */,
                              16 /* BLOCK_DMODEL */,
                              16 /* BLOCK_N */,
                              true /* CAUSAL */,
                              true /* ENABLE_DROPOUT */>;
}; // namespace oort

#endif

