#ifndef OORT_bwd_kernel_dk_dv_H
#define OORT_bwd_kernel_dk_dv_H

namespace oort {

template<int32_t BLOCK_M,
         int32_t BLOCK_DMODEL,
         int32_t BLOCK_N>
struct bwd_kernel_dk_dv {

 hipError_t operator()(dim3 grid, dim3 block, const __bf16* Q,
                       const __bf16* K,
                       const __bf16* V,
                       float sm_scale,
                       const __bf16* Out,
                       const __bf16* DO,
                       const __bf16* DK,
                       const __bf16* DV,
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
                       uint64_t seqlen_k, hipStream_t stream);

 hipError_t operator()(dim3 grid, dim3 block, const __fp16* Q,
                       const __fp16* K,
                       const __fp16* V,
                       float sm_scale,
                       const __fp16* Out,
                       const __fp16* DO,
                       const __fp16* DK,
                       const __fp16* DV,
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
                       uint64_t seqlen_k, hipStream_t stream);

};


template struct bwd_kernel_dk_dv<16 /* BLOCK_M */,
                                 64 /* BLOCK_DMODEL */,
                                 16 /* BLOCK_N */>;
template struct bwd_kernel_dk_dv<16 /* BLOCK_M */,
                                 32 /* BLOCK_DMODEL */,
                                 16 /* BLOCK_N */>;
template struct bwd_kernel_dk_dv<16 /* BLOCK_M */,
                                 128 /* BLOCK_DMODEL */,
                                 16 /* BLOCK_N */>;
template struct bwd_kernel_dk_dv<16 /* BLOCK_M */,
                                 16 /* BLOCK_DMODEL */,
                                 16 /* BLOCK_N */>;
}; // namespace oort

#endif

