#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>
#include <ATen/core/optional.h>
#include <vector>

int f() {
  // test trivially and non-trivially copyable.
  at::optional<int> x;
  at::optional<std::vector<int>> y;
  // reference these
  if (x.has_value()) {
    return 0;
  } else {
    if (y.has_value()) {
      return 1;
    }
  }
  return -1;
}

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

#include "generic/Abs.cu"
#include "THCGenerateFloatTypes.h"
