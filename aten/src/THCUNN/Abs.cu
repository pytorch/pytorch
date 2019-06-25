#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCApply.cuh>

template <typename T>
struct absupdateOutput_functor
{
  __device__ void operator()(T* output, const T* input) const
  {
    *output = THCNumerics<T>::abs(*input);
  }
};

template <typename T>
struct absupdateGradInput_functor
{
  __device__ void operator()(T* gradInput, const T* input, const T* gradOutput) const
  {
    *gradInput = *input < 0 ? - *gradOutput : *gradOutput;
  }
};

#include <THCUNN/generic/Abs.cu>
#include <THC/THCGenerateFloatTypes.h>
