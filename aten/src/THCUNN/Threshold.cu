#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct ThresholdUpdateOutput
{
  const T threshold_;
  const T val_;

  ThresholdUpdateOutput(T threshold, T val)
    : threshold_(threshold)
    , val_(val)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    *out = (x > threshold_) ? x : val_;
  }
};

// in-place variant
template <typename T>
struct ThresholdUpdateOutputIP
{
  const T threshold_;
  const T val_;

  ThresholdUpdateOutputIP(T threshold, T val)
    : threshold_(threshold)
    , val_(val)
  {}

  __device__ __forceinline__ void operator()(T *x)
  {
    *x = (*x > threshold_) ? *x : val_;
  }
};

template <typename T>
struct ThresholdUpdateGradInput
{
  const T threshold_;

  ThresholdUpdateGradInput(T threshold)
    : threshold_(threshold)
  {}

  __device__ __forceinline__ void operator()(
    T *gradInput, T *input, T *gradOutput) const
  {
    *gradInput = (*input > threshold_) ? *gradOutput : ScalarConvert<int, T>::to(0);
  }
};

template <typename T>
struct ThresholdUpdateGradInputIP
{
  const T threshold_;

  ThresholdUpdateGradInputIP(T threshold)
    : threshold_(threshold)
  {}

  __device__ __forceinline__ void operator()(
    T *gradOutput, T *input) const
  {
    *gradOutput = (*input > threshold_) ? *gradOutput : ScalarConvert<int, T>::to(0);
  }
};

#include "generic/Threshold.cu"
#include "THCGenerateFloatTypes.h"
