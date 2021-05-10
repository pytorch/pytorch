#include <TH/THHalf.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensor.hpp>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorMath.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorMathCompareT.cuh>

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

// clang-format off
#include <THC/generic/THCTensorMathPairwise.cu>
#include <THC/THCGenerateBoolType.h>
// clang-format on
