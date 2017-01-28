#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct sigmoidupdateOutput_functor
{
  __device__ void operator()(T *output, const T *input) const
  {
    *output = ScalarConvert<double, T>::to(1./(1.+ exp(-*input)));
  }
};

template <typename T>
struct sigmoidupdateGradInput_functor
{
  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = ScalarConvert<double, T>::to(*gradOutput * (1.-*output) * (*output));
  }
};

#include "generic/Sigmoid.cu"
#include "THCGenerateFloatTypes.h"
