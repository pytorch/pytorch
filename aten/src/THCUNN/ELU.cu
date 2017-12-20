#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct ELUupdateOutput_functor
{
  const T poscoef_;
  const T negcoef_;

  ELUupdateOutput_functor(T poscoef, T negcoef)
    : poscoef_(poscoef)
    , negcoef_(negcoef)
  {}

  __device__ void operator()(T *output, const T *input) const
  {
    *output = *input <= 0 ? (exp(*input) - 1) * poscoef_ : *input * negcoef_;
  }
};

// in-place variant
template <typename T>
struct ELUupdateOutputIP_functor
{
  const T poscoef_;
  const T negcoef_;

  ELUupdateOutputIP_functor(T poscoef, T negcoef)
    : poscoef_(poscoef)
    , negcoef_(negcoef)
  {}

  __device__ void operator()(T *x) const
  {
    *x = *x <= 0 ? (exp(*x) - 1) * poscoef_ : *x * negcoef_;
  }
};

template <typename T>
struct ELUupdateGradInput_functor
{
  const T poscoef_;
  const T negcoef_;

  ELUupdateGradInput_functor(T poscoef, T negcoef)
    : poscoef_(poscoef)
    , negcoef_(negcoef)
  {}

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + poscoef_)) : (*gradOutput * negcoef_);
  }
};

#include "generic/ELU.cu"
#include "THCGenerateFloatTypes.h"
