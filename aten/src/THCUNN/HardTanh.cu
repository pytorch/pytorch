#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct hardtanhupdateOutput_functor
{
  const T max_val_;
  const T min_val_;

  hardtanhupdateOutput_functor(T min_val, T max_val)
    : min_val_(min_val)
    , max_val_(max_val)
  {}

  __device__ void operator()(T *output, const T *input) const
  {
    if (*input < min_val_)
      *output = min_val_;
    else if (*input <= max_val_)
      *output = *input;
    else
      *output = max_val_;
  }

  __device__ void operator()(T *input) const
  {
    if (*input < min_val_)
      *input = min_val_;
    else if (*input > max_val_)
      *input = max_val_;
  }
};

template <typename T>
struct hardtanhupdateGradInput_functor
{
  const T max_val_;
  const T min_val_;

  hardtanhupdateGradInput_functor(T min_val, T max_val)
    : min_val_(min_val)
    , max_val_(max_val)
  {}

  __device__ void operator()(T *gradInput, const T *input, const T *gradOutput) const
  {
    if (*input <= min_val_ || *input >= max_val_)
      *gradInput = ScalarConvert<int, T>::to(0);
    else
      *gradInput = *gradOutput;
  }

  __device__ void operator()(T *gradInput, const T *input) const
  {
    if (*input <= min_val_ || *input >= max_val_)
      *gradInput = ScalarConvert<int, T>::to(0);
  }
};

#include "generic/HardTanh.cu"
#include "THCGenerateFloatTypes.h"
