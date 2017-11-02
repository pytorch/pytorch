#include "THCUNN.h"
#include "THCHalf.h"

#include "SoftMaxCommon.cuh"

template<typename T, typename AccumT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(T max_input, AccumT sum)
    : logsum(max_input + ScalarConvert<AccumT, T>::to(THCNumerics<AccumT>::log(sum))) {}

  __device__ __forceinline__ T operator()(T input) const {
    return input - logsum;
  }

  const T logsum;
};

template<typename T, typename AccumT>
struct LogSoftMaxBackwardEpilogue {
  __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
    : sum(ScalarConvert<AccumT, T>::to(sum)) {}

  __device__ __forceinline__ T operator()(T gradOutput, T output) const {
    return gradOutput - THCNumerics<T>::exp(output) * sum;
  }

  const T sum;
};

#include "generic/LogSoftMax.cu"
#include "THCGenerateFloatTypes.h"
