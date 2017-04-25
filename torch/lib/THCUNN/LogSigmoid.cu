#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct logSigmoid_updateOutput_functor
{
  __device__ void operator()(T *output, const T *input) const {
    *output = -THCNumerics<T>::log(1.f + THCNumerics<T>::exp(- *input));
  }
};

template <typename T>
struct logSigmoid_updateGradInput_functor
{
  __device__ void operator()(T *gradInput, const T *input, const T *gradOutput) const {
    const T z = THCNumerics<T>::exp(- *input);
    *gradInput = *gradOutput * z / (1.f + z);
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct logSigmoid_updateOutput_functor<half> {
  __device__ __forceinline__ void operator()(half* output, const half *input) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    const half one = __float2half(1.f);
    *output = __hneg(THCNumerics<half>::log(one + THCNumerics<half>::exp(__hneg(*input))));
#else
    float in = __half2float(*input);
    *output = __float2half(-THCNumerics<float>::log(1.f + THCNumerics<float>::exp(-in)));
#endif
  }
};

template <>
struct logSigmoid_updateGradInput_functor<half> {
  __device__ __forceinline__ void operator()(half* gradInput, const half *input, const half *gradOutput) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    const half one = __float2half(1.f);
    const half in_exp = THCNumerics<half>::exp(__hneg(*input));
    *gradInput = hdiv(__hmul(*gradOutput, in_exp), __hadd(one, in_exp));
#else
    const float in_exp = THCNumerics<float>::exp(-(__half2float(*input)));
    const float go = __half2float(*gradOutput);
    *gradInput = __float2half(go * in_exp / (1.f + in_exp));
#endif
  }
};
#endif

#include "generic/LogSigmoid.cu"
#include "THCGenerateFloatTypes.h"
