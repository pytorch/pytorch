#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct tanhupdateOutput_functor
{
  __device__ void operator()(T *output, const T *input) const
  {
    *output = tanh(*input);
  }
};

template <typename T>
struct tanhupdateGradInput_functor
{
  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = *gradOutput * (1 - *output * *output);
  }
};

#include "generic/Tanh.cu"
#include "THCGenerateFloatTypes.h"
