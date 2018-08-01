#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct TensorSigmoidOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    T one = (T) 1.0;
    *out = one / (one + THCNumerics<T>::exp(- *in));
  }

  __device__ __forceinline__ void operator()(T* v) const {
    T one = (T) 1.0;
    *v = one / (one + THCNumerics<T>::exp(- *v));
  }
};

template <>
struct TensorSigmoidOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
    float fin = ScalarConvert<half, float>::to(*in);
    *out = ScalarConvert<float, half>::to(1.0f / (1.0f + expf(- fin)));
  }

  __device__ __forceinline__ void operator()(half* v) const {
    float fv = ScalarConvert<half, float>::to(*v);
    *v = ScalarConvert<float, half>::to(1.0f / (1.0f + expf(- fv)));
  }
};

#include "generic/FusedRNNKernel.cu"
#include "THCGenerateFloatTypes.h"
