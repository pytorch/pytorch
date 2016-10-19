#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCDeviceTensorUtils.cu"
#else

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
  if (Dim != THCTensor_(nDimension)(state, t)) {
    THError("THCudaTensor dimension mismatch");
  }
  // Determine the maximum offset into the tensor achievable; `IndexT`
  // must be smaller than this type in order to use it.
  ptrdiff_t maxOffset = 0;
  IndexT sizes[Dim];
  IndexT strides[Dim];

  for (int i = 0; i < Dim; ++i) {
    long size = THCTensor_(size)(state, t, i);
    long stride = THCTensor_(stride)(state, t, i);

    maxOffset += (size - 1) * stride;

    sizes[i] = (IndexT) size;
    strides[i] = (IndexT) stride;
  }

  if (maxOffset > std::numeric_limits<IndexT>::max()) {
    THError("THCudaTensor sizes too large for THCDeviceTensor conversion");
  }

  return THCDeviceTensor<T, Dim, IndexT, PtrTraits>(
    THCTensor_(data)(state, t), sizes, strides);
}

#endif
