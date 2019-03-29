#ifndef THC_TENSOR_TYPE_UTILS_INC
#define THC_TENSOR_TYPE_UTILS_INC

#include <cuda.h>
#include <assert.h>
#include <THC/THCGeneral.h>
#include <TH/THHalf.h>
#include <THC/THCTensor.hpp>
#include <THC/THCTensorInfo.cuh>
#include <THC/THCTensor.hpp>

/// A utility for accessing THCuda*Tensor types in a generic manner

/// Equivalent to C++11's type_traits std::is_same; used for comparing
/// equality of types. Don't assume the existence of C++11
template <typename T, typename U>
struct SameType {
  static const bool same = false;
};

template <typename T>
struct SameType<T, T> {
  static const bool same = true;
};

template <typename T, typename U>
bool isSameType() {
  return SameType<T, U>::same;
}

// Utility function for constructing TensorInfo structs. In this case, the
// two template parameters are:
//
// 1. The TensorType, e.g. THCTensor in generic functions, or THCudaTensor,
// THCudaLongTensor etc.
//
// 2. The IndexType. This is always going to be an unsigned integral value,
// but depending on the size of the Tensor you may select uint16_t
// uint32_t, uint64_t etc.
//
// Internally we use the TensorUtils static functions to get the necessary
// dims, sizes, stride etc.
//
// For example, suppose we have a THCudaTensor t, with dim = 2, size = [3, 4],
// stride = [4, 1], offset = 8, and we set our index type to be unsigned int.
// Then we yield a TensorInfo struct templatized with float, unsigned int and
// the following fields:
//
// data is a float* to the underlying storage at position 8
// dims is 2
// sizes is a MAX_CUTORCH_DIMS element array with [3, 4] in its first two positions
// strides is a MAX_CUTORCH_DIMS element array with [4, 1] in its first two positions
//
// TensorInfos can then be passed to CUDA kernels, but we can use the static functions
// defined above to perform Tensor Operations that are appropriate for each
// TensorType.
template <typename ScalarType, typename TensorType, typename IndexType>
TensorInfo<ScalarType, IndexType>
getTensorInfo(THCState* state, TensorType* t) {
  IndexType sz[MAX_CUTORCH_DIMS];
  IndexType st[MAX_CUTORCH_DIMS];

  int dims = THCTensor_nDimensionLegacyNoScalars(state, t);
  for (int i = 0; i < dims; ++i) {
    sz[i] = THTensor_sizeLegacyNoScalars(t, i);
    st[i] = THTensor_strideLegacyNoScalars(t, i);
  }

  return TensorInfo<ScalarType, IndexType>(
    t->template data<ScalarType>(), dims, sz, st);
}

template <typename T>
struct ScalarNegate {
  static __host__ __device__ T to(const T v) { return -v; }
};

template <typename T>
struct ScalarInv {
  static __host__ __device__ T to(const T v) { return ((T) 1) / v; }
};

#endif // THC_TENSOR_TYPE_UTILS_INC
