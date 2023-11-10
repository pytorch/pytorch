#ifndef OORT_attn_fwd_H
#define OORT_attn_fwd_H

namespace oort {

template<int32_t STAGE,
         int32_t BLOCK_M,
         int32_t BLOCK_DMODEL,
         int32_t BLOCK_N,
         bool pre_load_v>
struct attn_fwd {

 hipError_t operator()(dim3 grid, dim3 block, const __fp16* Q,
                       const __fp16* K,
                       const __fp16* V,
                       float sm_scale,
                       const float* M,
                       const __fp16* Out,
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
                       uint64_t stride_oz,
                       uint64_t stride_oh,
                       uint64_t stride_om,
                       uint64_t stride_on,
                       uint64_t Z,
                       uint64_t H,
                       uint64_t seqlen_q,
                       uint64_t seqlen_k, hipStream_t stream);

 hipError_t operator()(dim3 grid, dim3 block, const __bf16* Q,
                       const __bf16* K,
                       const __bf16* V,
                       float sm_scale,
                       const float* M,
                       const __bf16* Out,
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
                       uint64_t stride_oz,
                       uint64_t stride_oh,
                       uint64_t stride_om,
                       uint64_t stride_on,
                       uint64_t Z,
                       uint64_t H,
                       uint64_t seqlen_q,
                       uint64_t seqlen_k, hipStream_t stream);

};


template struct attn_fwd<1 /* STAGE */,
                         128 /* BLOCK_M */,
                         128 /* BLOCK_DMODEL */,
                         64 /* BLOCK_N */,
                         true /* pre_load_v */>;
template struct attn_fwd<3 /* STAGE */,
                         128 /* BLOCK_M */,
                         64 /* BLOCK_DMODEL */,
                         64 /* BLOCK_N */,
                         true /* pre_load_v */>;
template struct attn_fwd<1 /* STAGE */,
                         128 /* BLOCK_M */,
                         16 /* BLOCK_DMODEL */,
                         64 /* BLOCK_N */,
                         true /* pre_load_v */>;
template struct attn_fwd<1 /* STAGE */,
                         128 /* BLOCK_M */,
                         64 /* BLOCK_DMODEL */,
                         64 /* BLOCK_N */,
                         true /* pre_load_v */>;
template struct attn_fwd<3 /* STAGE */,
                         128 /* BLOCK_M */,
                         16 /* BLOCK_DMODEL */,
                         64 /* BLOCK_N */,
                         true /* pre_load_v */>;
template struct attn_fwd<1 /* STAGE */,
                         128 /* BLOCK_M */,
                         32 /* BLOCK_DMODEL */,
                         64 /* BLOCK_N */,
                         true /* pre_load_v */>;
template struct attn_fwd<3 /* STAGE */,
                         128 /* BLOCK_M */,
                         128 /* BLOCK_DMODEL */,
                         64 /* BLOCK_N */,
                         true /* pre_load_v */>;
template struct attn_fwd<3 /* STAGE */,
                         128 /* BLOCK_M */,
                         32 /* BLOCK_DMODEL */,
                         64 /* BLOCK_N */,
                         true /* pre_load_v */>;
}; // namespace oort

#endif

