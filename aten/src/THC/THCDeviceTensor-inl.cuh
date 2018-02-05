#include <assert.h>

namespace detail {

template <typename T, int N>
__host__ __device__ void copy(T to[N], T from[N]) {
  for (int i = 0; i < N; ++i) {
    to[i] = from[i];
  }
}

} // namespace detail

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::THCDeviceTensor()
    : data_(NULL) {
  thc_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = 0;
    stride_[i] = (IndexT) 1;
  }
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::
#ifdef _MSC_VER
THCDeviceTensor(DataPtrType data, const IndexT (&sizes)[Dim])
#else
THCDeviceTensor(DataPtrType data, const IndexT sizes[Dim])
#endif
    : data_(data) {
  thc_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = sizes[i];
  }

  stride_[Dim - 1] = (IndexT) 1;
  for (int i = Dim - 2; i >= 0; --i) {
    stride_[i] = stride_[i + 1] * sizes[i + 1];
  }
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::THCDeviceTensor(
#ifdef _MSC_VER
  DataPtrType data, const IndexT (&sizes)[Dim], const IndexT (&strides)[Dim])
#else
  DataPtrType data, const IndexT sizes[Dim], const IndexT strides[Dim])
#endif
    : data_(data) {
  thc_static_assert(Dim > 0);

  for (int i = 0; i < Dim; ++i) {
    size_[i] = sizes[i];
    stride_[i] = strides[i];
  }
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int OtherDim>
__host__ __device__ bool
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::isSameSizeAndStride(
  const THCDeviceTensor<T, OtherDim, IndexT, PtrTraits>& rhs) const {
  if (Dim != OtherDim) {
    return false;
  }

  for (int i = 0; i < Dim; ++i) {
    if (size_[i] != rhs.size_[i]) {
      return false;
    }

    if (stride_[i] != rhs.stride_[i]) {
      return false;
    }
  }

  return true;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <typename U>
__host__ __device__ THCDeviceTensor<U, Dim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::cast() {
  thc_static_assert(sizeof(U) == sizeof(T));

  return THCDeviceTensor<U, Dim, IndexT, PtrTraits>(
    reinterpret_cast<U*>(data_), size_, stride_);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <typename U>
__host__ __device__ const THCDeviceTensor<U, Dim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::cast() const {
  thc_static_assert(sizeof(U) == sizeof(T));

  return THCDeviceTensor<U, Dim, IndexT, PtrTraits>(
    reinterpret_cast<U*>(data_), size_, stride_);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ ptrdiff_t
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::numElements() const {
  ptrdiff_t size = getSize(0);

  for (int i = 1; i < Dim; ++i) {
    size *= getSize(i);
  }

  return size;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::isContiguous() const {
  return isContiguousRange(0, Dim);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::isConsistentlySized(int i) const {
  if (i == 0 && getStride(i) > 0 && getSize(i) > 0) {
    return true;
  } else if ((i > 0) && (i < Dim) && (getStride(i) > 0) &&
             ((getStride(i - 1) / getStride(i)) >= getSize(i))) {
    return true;
  }

  return false;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::isConsistentlySized() const {
  for (int i = 0; i < Dim; ++i) {
    if (!isConsistentlySized(i)) {
      return false;
    }
  }

  return true;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::isContiguousRange(
  int first, int last) const {

  // We're testing if dimensions [first, last) are a part of a contiguous range
  // Call a dimension untrivial if it has size > 1 and trivial if it has size 1
  //
  // There is an edge case when the sizes of the last dimensions of
  // [first, last) are all 1's. The following needs to happen:
  // 1. If [first, last) ends with a trivial dim, find the next untrivial dimension,
  //    and get its size. If it doesn't exist, use 1 as the size.
  // 2. Find the dim newLast such that newLast is the largest dim < last
  //    such that newLast is untrivial.
  //
  // We now test that [first, newLast] is a contiguous range, using the size of
  // the next untrivial dimension that follows newLast to compute prevSize,
  // or prevSize = 1 if all the dimensions following newLast are all trivial.
  // (newLast, last) are ignored because they are trivial and don't matter.

  int newLast = last;
  int next_untrivial_dim = last;
  if (next_untrivial_dim < Dim && getSize(next_untrivial_dim) == 1) {
    // Find the next untrivial dim that is > last
    while (next_untrivial_dim < Dim && getSize(next_untrivial_dim) == 1) {
      ++next_untrivial_dim;
    }

    // Find the first untrivial dim that is < last
    int newLast = last;
    while (newLast >= first && getSize(newLast) == 1) {
      --newLast;
    }

    // Our entire range [first, last) was trivial
    if (newLast == first) {
      return true;
    }
  }

  int64_t prevSize = next_untrivial_dim < Dim ?
    getStride(next_untrivial_dim) * getSize(next_untrivial_dim) : 1;

  for (int i = newLast - 1; i >= first; --i) {
    if (getSize(i) != (IndexT) 1) {
      if (getStride(i) == prevSize) {
        prevSize *= getSize(i);
      } else {
        return false;
      }
    }
  }

  return true;
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ THCDeviceTensor<T, Dim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::transpose(int dim1,
                                                      int dim2) const {
#ifdef __CUDA_ARCH__
  // Device code
  assert(dim1 >= 0 && dim1 < Dim);
  assert(dim1 >= 0 && dim2 < Dim);
#else
  // Host code
  if (dim1 < 0 || dim1 >= Dim) {
    THError("dim1 out of bounds");
  }

  if (dim2 < 0 || dim2 >= Dim) {
    THError("dim2 out of bounds");
  }
#endif

  IndexT newSize[Dim];
  IndexT newStride[Dim];

  for (int i = 0; i < Dim; ++i) {
    newSize[i] = size_[i];
    newStride[i] = stride_[i];
  }

  IndexT tmp = newSize[dim1];
  newSize[dim1] = newSize[dim2];
  newSize[dim2] = tmp;

  tmp = newStride[dim1];
  newStride[dim1] = newStride[dim2];
  newStride[dim2] = tmp;

  return THCDeviceTensor<T, Dim, IndexT, PtrTraits>(data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::upcastOuter() {
  // Can only create tensors of greater dimension
  thc_static_assert(NewDim > Dim);

  IndexT newSize[NewDim];
  IndexT newStride[NewDim];

  int shift = NewDim - Dim;

  for (int i = 0; i < NewDim; ++i) {
    if (i < shift) {
      // These are the extended dimensions
      newSize[i] = (IndexT) 1;
      newStride[i] = size_[0] * stride_[0];
    } else {
      // Shift the remaining dimensions
      newSize[i] = size_[i - shift];
      newStride[i] = stride_[i - shift];
    }
  }

  return THCDeviceTensor<T, NewDim, IndexT, PtrTraits>(
    data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::upcastInner() {
  // Can only create tensors of greater dimension
  thc_static_assert(NewDim > Dim);

  IndexT newSize[NewDim];
  IndexT newStride[NewDim];

  for (int i = 0; i < NewDim; ++i) {
    if (i < Dim) {
      // Existing dimensions get copied over
      newSize[i] = size_[i];
      newStride[i] = stride_[i];
    } else {
      // Extended dimensions
      newSize[i] = (IndexT) 1;
      newStride[i] = (IndexT) 1;
    }
  }

  return THCDeviceTensor<T, NewDim, IndexT, PtrTraits>(
    data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::downcastOuter() {
  // Can only create tensors of lesser dimension
  thc_static_assert(NewDim < Dim);

  // We can't downcast non-contiguous tensors, since it leaves
  // garbage data in the tensor. The tensor needs to be contiguous
  // in all of the dimensions we are collapsing (no padding in
  // them).
  bool cont = isContiguousRange(0, Dim - NewDim);
#ifdef __CUDA_ARCH__
  // Device code
  assert(cont);
#else
  // Host code
  if (!cont) {
    THError("Can only downcast contiguous tensors");
  }
#endif

  IndexT newSize[NewDim];
  IndexT newStride[NewDim];

  int ignoredDims = Dim - NewDim;
  IndexT collapsedSize = 1;

  for (int i = 0; i < Dim; ++i) {
    if (i < ignoredDims) {
      // Collapse these dimensions
      collapsedSize *= getSize(i);
      continue;
    }

    // Non-collapsed dimensions

    if (i == ignoredDims) {
      // This is the first non-collapsed dimension
      newSize[i - ignoredDims] = collapsedSize * getSize(i);

      // If the size of this dimension is 1, the stride could
      // be anything. Recompute a reasonable stride based
      // on the assumption that the outer dimensions are
      // all contiguous.
      if (getSize(i) == 1) {
        int innerSize = 1;
        for (int j = ignoredDims + 1; j < Dim; ++j) {
          innerSize *= getSize(j);
        }
        newStride[i - ignoredDims] = innerSize;
        continue;
      }

      // If the size of this dimension wasn't 1, then
      // use the stride information
      newStride[i - ignoredDims] = getStride(i);
      continue;
    }

    // Subsequent non-collapsed dimensions
    newSize[i - ignoredDims] = getSize(i);
    newStride[i - ignoredDims] = getStride(i);
  }

  return THCDeviceTensor<T, NewDim, IndexT, PtrTraits>(
    data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::downcastInner() {
  // Can only create tensors of lesser dimension
  thc_static_assert(NewDim < Dim);

  // We can't downcast non-contiguous tensors, since it leaves
  // garbage data in the tensor. The tensor needs to be contiguous
  // in all of the dimensions we are collapsing (no padding in
  // them).
  bool cont = isContiguousRange(NewDim, Dim);
#ifdef __CUDA_ARCH__
  // Device code
  assert(cont);
#else
  // Host code
  if (!cont) {
    THError("Can only downcast contiguous tensors");
  }
#endif

  IndexT newSize[NewDim];
  IndexT newStride[NewDim];

  IndexT collapsedSize = 1;

  for (int i = Dim - 1; i >= 0; --i) {
    if (i >= NewDim) {
      // Collapse these dimensions
      collapsedSize *= getSize(i);
    } else {
      // Non-collapsed dimensions
      if (i == NewDim - 1) {
        // This is the first non-collapsed dimension
        newSize[i] = collapsedSize * getSize(i);
        newStride[i] = getStride(Dim - 1);
      } else {
        // Subsequent non-collapsed dimensions
        newSize[i] = getSize(i);
        newStride[i] = getStride(i);
      }
    }
  }

  return THCDeviceTensor<T, NewDim, IndexT, PtrTraits>(
    data_, newSize, newStride);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int SubDim>
__host__ __device__ THCDeviceTensor<T, SubDim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::view(DataPtrType at) {
  thc_static_assert(SubDim >= 1 && SubDim < Dim);

  IndexT viewSizes[SubDim];
  IndexT viewStrides[SubDim];

  for (int i = 0; i < SubDim; ++i) {
    viewSizes[i] = size_[Dim - SubDim + i];
    viewStrides[i] = stride_[Dim - SubDim + i];
  }

  return THCDeviceTensor<T, SubDim, IndexT, PtrTraits>(
    at, viewSizes, viewStrides);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
template <int SubDim>
__host__ __device__ THCDeviceTensor<T, SubDim, IndexT, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::view() {
  return view<SubDim>(data_);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
void
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::zero(cudaStream_t stream) {
#ifdef __CUDA_ARCH__
  assert(isContiguous());
#else
  if (!isContiguous()) {
    THError("fillAsync only works on contiguous data");
  }
#endif

  cudaMemsetAsync(data(), 0, numElements() * sizeof(T), stream);
}
