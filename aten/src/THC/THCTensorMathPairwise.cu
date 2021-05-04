#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <TH/THHalf.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorMathCompareT.cuh>
#include <THC/THCTensor.hpp>

template <typename T>
struct TensorMulConstantOp {
  TensorMulConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v *= val;
  }

  const T val;
};

#include <THC/generic/THCTensorMathPairwise.cu>
#include <THC/THCGenerateBoolType.h>
