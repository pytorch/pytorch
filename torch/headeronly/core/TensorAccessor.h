#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>

namespace torch::headeronly {

// The PtrTraits argument to the TensorAccessor/GenericPackedTensorAccessor
// is used to enable the __restrict__ keyword/modifier for the data
// passed to cuda.
template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};
#endif

typedef int64_t ArrayRefT;

namespace detail {

template <
    template <typename I> class ArrayRef,
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class TensorAccessorBase {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  using ArrayRefCls = ArrayRef<ArrayRefT>;

  C10_HOST_DEVICE TensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_), sizes_(sizes_), strides_(strides_) {}
  C10_HOST ArrayRefCls sizes() const {
    return ArrayRefCls(sizes_, N);
  }
  C10_HOST ArrayRefCls strides() const {
    return ArrayRefCls(strides_, N);
  }
  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }

 protected:
  PtrType data_;
  const index_t* sizes_;
  const index_t* strides_;
};

// The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
// `Tensor.accessor<T, N>()`.
// For CUDA `Tensor`s, `GenericPackedTensorAccessor` is used on the host and
// only indexing on the device uses `TensorAccessor`s.
template <
    template <typename I> class ArrayRef,
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class TensorAccessor
    : public TensorAccessorBase<ArrayRef, T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<ArrayRef, T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_) {}

  C10_HOST_DEVICE TensorAccessor<ArrayRef, T, N - 1, PtrTraits, index_t>
  operator[](index_t i) {
    return TensorAccessor<ArrayRef, T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }

  C10_HOST_DEVICE const TensorAccessor<ArrayRef, T, N - 1, PtrTraits, index_t>
  operator[](index_t i) const {
    return TensorAccessor<ArrayRef, T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }
};

template <
    template <typename I> class ArrayRef,
    typename T,
    template <typename U> class PtrTraits,
    typename index_t>
class TensorAccessor<ArrayRef, T, 1, PtrTraits, index_t>
    : public TensorAccessorBase<ArrayRef, T, 1, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<ArrayRef, T, 1, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_) {}
  C10_HOST_DEVICE T& operator[](index_t i) {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->data_[this->strides_[0] * i];
  }
  C10_HOST_DEVICE const T& operator[](index_t i) const {
    return this->data_[this->strides_[0] * i];
  }
};

// GenericPackedTensorAccessorBase and GenericPackedTensorAccessor are used on
// for CUDA `Tensor`s on the host and as In contrast to `TensorAccessor`s, they
// copy the strides and sizes on instantiation (on the host) in order to
// transfer them on the device when calling kernels. On the device, indexing of
// multidimensional tensors gives to `TensorAccessor`s. Use RestrictPtrTraits as
// PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
// Instantiation from data, sizes, strides is only needed on the host and
// std::copy isn't available on the device, so those functions are host only.
template <
    typename IndexBoundsCheck,
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST GenericPackedTensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_) {
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
  }

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <
      typename source_index_t,
      class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  C10_HOST GenericPackedTensorAccessorBase(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : data_(data_) {
    for (size_t i = 0; i < N; ++i) {
      this->sizes_[i] = sizes_[i];
      this->strides_[i] = strides_[i];
    }
  }

  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }

 protected:
  PtrType data_;
  // NOLINTNEXTLINE(*c-arrays*)
  index_t sizes_[N];
  // NOLINTNEXTLINE(*c-arrays*)
  index_t strides_[N];
  C10_HOST void bounds_check_(index_t i) const {
    IndexBoundsCheck _(i);
  }
};

template <
    template <typename I> class ArrayRef,
    typename IndexBoundsCheck,
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class GenericPackedTensorAccessor : public GenericPackedTensorAccessorBase<
                                        IndexBoundsCheck,
                                        T,
                                        N,
                                        PtrTraits,
                                        index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : GenericPackedTensorAccessorBase<
            IndexBoundsCheck,
            T,
            N,
            PtrTraits,
            index_t>(data_, sizes_, strides_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <
      typename source_index_t,
      class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : GenericPackedTensorAccessorBase<
            IndexBoundsCheck,
            T,
            N,
            PtrTraits,
            index_t>(data_, sizes_, strides_) {}

  C10_DEVICE TensorAccessor<ArrayRef, T, N - 1, PtrTraits, index_t> operator[](
      index_t i) {
    index_t* new_sizes = this->sizes_ + 1;
    index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<ArrayRef, T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i, new_sizes, new_strides);
  }

  C10_DEVICE const TensorAccessor<ArrayRef, T, N - 1, PtrTraits, index_t>
  operator[](index_t i) const {
    const index_t* new_sizes = this->sizes_ + 1;
    const index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<ArrayRef, T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i, new_sizes, new_strides);
  }

  /// Returns a PackedTensorAccessor of the same dimension after transposing the
  /// two dimensions given. Does not actually move elements; transposition is
  /// made by permuting the size/stride arrays. If the dimensions are not valid,
  /// asserts.
  C10_HOST GenericPackedTensorAccessor<
      ArrayRef,
      IndexBoundsCheck,
      T,
      N,
      PtrTraits,
      index_t>
  transpose(index_t dim1, index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    GenericPackedTensorAccessor<
        ArrayRef,
        IndexBoundsCheck,
        T,
        N,
        PtrTraits,
        index_t>
        result(this->data_, this->sizes_, this->strides_);
    std::swap(result.strides_[dim1], result.strides_[dim2]);
    std::swap(result.sizes_[dim1], result.sizes_[dim2]);
    return result;
  }
};

template <
    template <typename I> class ArrayRef,
    typename IndexBoundsCheck,
    typename T,
    template <typename U> class PtrTraits,
    typename index_t>
class GenericPackedTensorAccessor<
    ArrayRef,
    IndexBoundsCheck,
    T,
    1,
    PtrTraits,
    index_t>
    : public GenericPackedTensorAccessorBase<
          IndexBoundsCheck,
          T,
          1,
          PtrTraits,
          index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : GenericPackedTensorAccessorBase<
            IndexBoundsCheck,
            T,
            1,
            PtrTraits,
            index_t>(data_, sizes_, strides_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <
      typename source_index_t,
      class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : GenericPackedTensorAccessorBase<
            IndexBoundsCheck,
            T,
            1,
            PtrTraits,
            index_t>(data_, sizes_, strides_) {}

  C10_DEVICE T& operator[](index_t i) {
    return this->data_[this->strides_[0] * i];
  }
  C10_DEVICE const T& operator[](index_t i) const {
    return this->data_[this->strides_[0] * i];
  }

  // Same as in the general N-dimensional case, but note that in the
  // 1-dimensional case the returned PackedTensorAccessor will always be an
  // identical copy of the original
  C10_HOST GenericPackedTensorAccessor<
      ArrayRef,
      IndexBoundsCheck,
      T,
      1,
      PtrTraits,
      index_t>
  transpose(index_t dim1, index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    return GenericPackedTensorAccessor<
        ArrayRef,
        IndexBoundsCheck,
        T,
        1,
        PtrTraits,
        index_t>(this->data_, this->sizes_, this->strides_);
  }
};

} // namespace detail

namespace {
template <size_t N, typename index_t>
struct IndexBoundsCheck {
  IndexBoundsCheck(index_t i) {
    STD_TORCH_CHECK(
        0 <= i && i < index_t{N},
        "Index ",
        i,
        " is not within bounds of a tensor of dimension ",
        N);
  }
};

// HeaderOnlyArrayRef here is a placeholder of
// torch::headeronly::HeaderOnlyArrayRef to be removed after
// https://github.com/pytorch/pytorch/pull/164991 lands
template <typename T>
class HeaderOnlyArrayRef {
 protected:
  const T* Data;
  size_t Length;
};
// using IntHeaderOnlyArrayRef = HeaderOnlyArrayRef<int64_t>;

} // anonymous namespace

// TensorAccessorBase is same as at::TensorAccessorBase except sizes() and
// strides() return IntHeaderOnlyArrayRef
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
using TensorAccessorBase =
    detail::TensorAccessorBase<HeaderOnlyArrayRef, T, N, PtrTraits, index_t>;

// TensorAccessor is same as at::TensorAccessor except sizes() and strides()
// return HeaderOnlyArrayRef<int64_t>
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
using TensorAccessor =
    detail::TensorAccessor<HeaderOnlyArrayRef, T, N, PtrTraits, index_t>;

// GenericPackedTensorAccessorBase is same as
// at::GenericPackedTensorAccessorBase except sizes() and strides() return
// IntHeaderOnlyArrayRef
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
using GenericPackedTensorAccessorBase = detail::GenericPackedTensorAccessorBase<
    IndexBoundsCheck<N, index_t>,
    T,
    N,
    PtrTraits,
    index_t>;

// GenericPackedTensorAccessor is same as at::GenericPackedTensorAccessor except
// sizes() and strides() return IntHeaderOnlyArrayRef, and bounds check uses
// STD_TORCH_CHECK instead of TORCH_CHECK_INDEX
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
using GenericPackedTensorAccessor = detail::GenericPackedTensorAccessor<
    HeaderOnlyArrayRef,
    IndexBoundsCheck<N, index_t>,
    T,
    N,
    PtrTraits,
    index_t>;

} // namespace torch::headeronly
