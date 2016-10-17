#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

#include <cfloat>

template <typename T>
struct TensorFillOp {
  TensorFillOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* v) { *v = val; }

  const T val;
};

#include "generic/THCTensorMath.cu"
#include "THCGenerateAllTypes.h"
