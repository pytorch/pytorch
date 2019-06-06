#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCApply.cuh>

template <typename T>
struct ELUupdateOutput_functor
{
  const T negcoef_;
  const T poscoef_;
  const T negiptcoef_;

  ELUupdateOutput_functor(T negcoef, T poscoef, T negiptcoef)
    : negcoef_(negcoef)
    , poscoef_(poscoef)
    , negiptcoef_(negiptcoef)
  {}

  __device__ void operator()(T *output, const T *input) const
  {
    *output = *input <= 0 ? (exp(*input * negiptcoef_) - 1) * negcoef_ : *input * poscoef_;
  }
};

// in-place variant
template <typename T>
struct ELUupdateOutputIP_functor
{
  const T negcoef_;
  const T poscoef_;
  const T negiptcoef_;

  ELUupdateOutputIP_functor(T negcoef, T poscoef, T negiptcoef)
    : negcoef_(negcoef)
    , poscoef_(poscoef)
    , negiptcoef_(negiptcoef)
  {}

  __device__ void operator()(T *x) const
  {
    *x = *x <= 0 ? (exp(*x * negiptcoef_) - 1) * negcoef_ : *x * poscoef_;
  }
};

template <typename T>
struct ELUupdateGradInput_functor
{
  const T negcoef_;
  const T poscoef_;
  const T negiptcoef_;

  ELUupdateGradInput_functor(T negcoef, T poscoef, T negiptcoef)
    : negcoef_(negcoef)
    , poscoef_(poscoef)
    , negiptcoef_(negiptcoef)
  {}

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * negiptcoef_ * (*output + negcoef_)) : (*gradOutput * poscoef_);
  }
};

#include <THCUNN/generic/ELU.cu>
#include <THC/THCGenerateFloatTypes.h>
