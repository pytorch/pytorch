#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct ELUupdateOutput_functor
{
  const T negcoef_;
  const T poscoef_;

  ELUupdateOutput_functor(T negcoef, T poscoef)
    : negcoef_(negcoef)
    , poscoef_(poscoef)
  {}

  __device__ void operator()(T *output, const T *input) const
  {
    *output = *input <= 0 ? (exp(*input) - 1) * negcoef_ : *input * poscoef_;
  }
};

// in-place variant
template <typename T>
struct ELUupdateOutputIP_functor
{
  const T negcoef_;
  const T poscoef_;

  ELUupdateOutputIP_functor(T negcoef, T poscoef)
    : negcoef_(negcoef)
    , poscoef_(poscoef)
  {}

  __device__ void operator()(T *x) const
  {
    *x = *x <= 0 ? (exp(*x) - 1) * negcoef_ : *x * poscoef_;
  }
};

template <typename T>
struct ELUupdateGradInput_functor
{
  const T negcoef_;
  const T poscoef_;

  ELUupdateGradInput_functor(T negcoef, T poscoef)
    : negcoef_(negcoef)
    , poscoef_(poscoef)
  {}

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + negcoef_)) : (*gradOutput * poscoef_);
  }
};

#include "generic/ELU.cu"
#include "THCGenerateFloatTypes.h"
