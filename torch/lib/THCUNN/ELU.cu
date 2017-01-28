#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct ELUupdateOutput_functor
{
  const T alpha_;

  ELUupdateOutput_functor(T alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(T *output, const T *input) const
  {
    *output = *input <= 0 ? (exp(*input) - 1) * alpha_ : *input;
  }
};

// in-place variant
template <typename T>
struct ELUupdateOutputIP_functor
{
  const T alpha_;

  ELUupdateOutputIP_functor(T alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(T *x) const
  {
    *x = *x <= 0 ? (exp(*x) - 1) * alpha_ : *x;
  }
};

template <typename T>
struct ELUupdateGradInput_functor
{
  const T alpha_;

  ELUupdateGradInput_functor(T alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
};

template <typename T>
struct ELUupdateGradInputIP_functor
{
  const T alpha_;

  ELUupdateGradInputIP_functor(T alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(T *gradOutput, const T *output) const
  {
    *gradOutput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
};

#include "generic/ELU.cu"
#include "THCGenerateFloatTypes.h"
