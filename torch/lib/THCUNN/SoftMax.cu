#include "THCUNN.h"
#include "THCHalf.h"

#include "SoftMaxCommon.cuh"

template<typename T, typename AccumT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(T max_input, AccumT sum)
    : max_input(max_input)
    , sum(ScalarConvert<AccumT, T>::to(sum)) {}

  __device__ __forceinline__ T operator()(T input) const {
    return THCNumerics<T>::exp(input - max_input) / sum;
  }

  const T max_input;
  const T sum;
};

template<typename T, typename AccumT>
struct SoftMaxBackwardEpilogue {
  __device__ __forceinline__ SoftMaxBackwardEpilogue(AccumT sum)
    : sum(ScalarConvert<AccumT, T>::to(sum)) {}

  // XXX: gradOutput that we get here is really gradOutput * output
  // Look for cmul in SoftMax_updateGradInput
  __device__ __forceinline__ T operator()(T gradOutput, T output) const {
    return gradOutput - output * sum;
  }

  const T sum;
};

#include "generic/SoftMax.cu"
#include "THCGenerateFloatTypes.h"
