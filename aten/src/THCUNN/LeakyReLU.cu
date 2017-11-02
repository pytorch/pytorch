#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct LeakyReLUUpdateOutput
{
  const T negval_;

  LeakyReLUUpdateOutput(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    *out = (x > 0) ? x : x * negval_;
  }
};

// in-place variant
template <typename T>
struct LeakyReLUUpdateOutputIP
{
  const T negval_;

  LeakyReLUUpdateOutputIP(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(T *x)
  {
    *x = (*x > 0) ? *x : negval_ * (*x);
  }
};

template <typename T>
struct LeakyReLUUpdateGradInput
{
  const T negval_;

  LeakyReLUUpdateGradInput(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(
    T* gradInput,
    T* input,
    T* gradOutput) const
  {
    *gradInput = (*input > 0) ? *gradOutput : (*gradOutput) * negval_;
  }
};

template <typename T>
struct LeakyReLUUpdateGradInputIP
{
  const T negval_;

  LeakyReLUUpdateGradInputIP(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(
    T* gradOutput,
    T* input) const
  {
    *gradOutput = (*input > 0) ? *gradOutput : (*gradOutput) * negval_;
  }
};

#include "generic/LeakyReLU.cu"
#include "THCGenerateFloatTypes.h"
