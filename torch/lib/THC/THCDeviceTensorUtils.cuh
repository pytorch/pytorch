#ifndef THC_DEVICE_TENSOR_UTILS_INC
#define THC_DEVICE_TENSOR_UTILS_INC

#include "THCDeviceTensor.cuh"
#include "THCTensor.h"

/// Constructs a THCDeviceTensor initialized from a THCudaTensor. Will
/// error if the dimensionality does not match exactly.
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensor(THCState* state, THCudaTensor* t);

template <typename T, int Dim, typename IndexT>
THCDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>
toDeviceTensor(THCState* state, THCudaTensor* t) {
  return toDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
THCDeviceTensor<T, Dim, int, DefaultPtrTraits>
toDeviceTensor(THCState* state, THCudaTensor* t) {
  return toDeviceTensor<T, Dim, int, DefaultPtrTraits>(state, t);
}

/// Constructs a DeviceTensor initialized from a THCudaTensor by
/// upcasting or downcasting the tensor to that of a different
/// dimension.
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensorCast(THCState* state, THCudaTensor* t);

template <typename T, int Dim, typename IndexT>
THCDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>
toDeviceTensorCast(THCState* state, THCudaTensor* t) {
  return toDeviceTensorCast<T, Dim, IndexT, DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
THCDeviceTensor<T, Dim, int, DefaultPtrTraits>
toDeviceTensorCast(THCState* state, THCudaTensor* t) {
  return toDeviceTensorCast<T, Dim, int, DefaultPtrTraits>(state, t);
}

#include "THCDeviceTensorUtils-inl.cuh"

#endif // THC_DEVICE_TENSOR_UTILS_INC
