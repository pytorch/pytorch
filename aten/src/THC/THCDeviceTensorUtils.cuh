#ifndef THC_DEVICE_TENSOR_UTILS_INC
#define THC_DEVICE_TENSOR_UTILS_INC

#include <THC/THCDeviceTensor.cuh>
#include <THC/THCTensor.hpp>
#include <limits>

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

/// Constructs a THCDeviceTensor initialized from a THCudaTensor. Will
/// error if the dimensionality does not match exactly.
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensor(THCState* state, THCTensor* t);

template <typename T, int Dim, typename IndexT>
THCDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>
toDeviceTensor(THCState* state, THCTensor* t) {
  return toDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
THCDeviceTensor<T, Dim, int, DefaultPtrTraits>
toDeviceTensor(THCState* state, THCTensor* t) {
  return toDeviceTensor<T, Dim, int, DefaultPtrTraits>(state, t);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensor(THCState* state, THCTensor* t) {
  if (Dim != THCTensor_nDimensionLegacyAll(state, t)) {
    THError("THCudaTensor dimension mismatch");
  }
  // Determine the maximum offset into the tensor achievable; `IndexT`
  // must be smaller than this type in order to use it.
  ptrdiff_t maxOffset = 0;
  IndexT sizes[Dim];
  IndexT strides[Dim];

  for (int i = 0; i < Dim; ++i) {
    int64_t size = THTensor_sizeLegacyNoScalars(t, i);
    int64_t stride = THTensor_strideLegacyNoScalars(t, i);

    maxOffset += (size - 1) * stride;

    sizes[i] = (IndexT) size;
    strides[i] = (IndexT) stride;
  }

  if (maxOffset > std::numeric_limits<IndexT>::max()) {
    THError("THCudaTensor sizes too large for THCDeviceTensor conversion");
  }

  return THCDeviceTensor<T, Dim, IndexT, PtrTraits>(
    t->data<T>(), sizes, strides);
}

#include <THC/THCDeviceTensorUtils-inl.cuh>

#endif // THC_DEVICE_TENSOR_UTILS_INC
