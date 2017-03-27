#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct logSigmoid_updateOutput_functor
{
  __device__ void operator()(T *output, const T *input) const
  {
    T z = exp(-*input);
    *output = ScalarConvert<double, T>::to(-log(1. + z));
  }
};

template <typename T>
struct logSigmoid_updateGradInput_functor
{
  __device__ void operator()(T *gradInput, const T *input, const T *gradOutput) const
  {
    T z = exp(-*input);
    *gradInput = ScalarConvert<double, T>::to(*gradOutput * z / (1. + z));
  }
};

#include "generic/LogSigmoid.cu"
#include "THCGenerateFloatTypes.h"
