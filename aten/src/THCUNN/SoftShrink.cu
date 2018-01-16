#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct SoftShrinkUpdateOutput
{
  const T lambda_;

  SoftShrinkUpdateOutput(T lambda)
    : lambda_(lambda)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    if (x > lambda_) *out = x - lambda_;
    else if (x < -lambda_) *out = x + lambda_;
    else *out = ScalarConvert<int, T>::to(0);
  }
};

template <typename T>
struct SoftShrinkUpdateGradInput
{
  const T lambda_;

  SoftShrinkUpdateGradInput(T lambda)
    : lambda_(lambda)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T *input, T *gradOutput) const
  {
    T x = *input;
    if (x > lambda_ || x < -lambda_)
      *gradInput = *gradOutput;
    else
      *gradInput = ScalarConvert<int, T>::to(0);
  }
};

#include "generic/SoftShrink.cu"
#include "THCGenerateFloatTypes.h"
