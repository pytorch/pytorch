#ifndef OORT_bwd_preprocess_H
#define OORT_bwd_preprocess_H

namespace oort {

template<int32_t BLOCK_M,
         int32_t D_HEAD>
struct bwd_preprocess {

 hipError_t operator()(dim3 grid, dim3 block, const __bf16* Out,
                       const __bf16* DO,
                       const __bf16* NewDO,
                       const float* Delta, hipStream_t stream);

 hipError_t operator()(dim3 grid, dim3 block, const __fp16* Out,
                       const __fp16* DO,
                       const __fp16* NewDO,
                       const float* Delta, hipStream_t stream);

};


template struct bwd_preprocess<128 /* BLOCK_M */,
                               32 /* D_HEAD */>;
template struct bwd_preprocess<64 /* BLOCK_M */,
                               64 /* D_HEAD */>;
template struct bwd_preprocess<128 /* BLOCK_M */,
                               64 /* D_HEAD */>;
template struct bwd_preprocess<64 /* BLOCK_M */,
                               128 /* D_HEAD */>;
template struct bwd_preprocess<64 /* BLOCK_M */,
                               32 /* D_HEAD */>;
template struct bwd_preprocess<128 /* BLOCK_M */,
                               128 /* D_HEAD */>;
template struct bwd_preprocess<128 /* BLOCK_M */,
                               16 /* D_HEAD */>;
template struct bwd_preprocess<64 /* BLOCK_M */,
                               16 /* D_HEAD */>;
}; // namespace oort

#endif

