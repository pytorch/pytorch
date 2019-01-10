#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct sigmoid_updateGradInput_functor {
  __device__ __forceinline__ void operator()(T* gradInput, const T *output, const T *gradOutput) const {
    *gradInput = *gradOutput * (1.f - *output) * (*output);
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct sigmoid_updateGradInput_functor<half> {
  __device__ __forceinline__ void operator()(half* gradInput, const half *output, const half *gradOutput) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    const half one = __float2half(1.f);
    *gradInput = __hmul(*gradOutput, __hmul(__hadd(one, __hneg(*output)), *output));
#else
    const float out = __half2float(*output);
    const float go = __half2float(*gradOutput);
    *gradInput = __float2half(go * (1.f - out) * out);
#endif
  }
};
#endif

#include "generic/Sigmoid.cu"
#include "THCGenerateFloatTypes.h"
