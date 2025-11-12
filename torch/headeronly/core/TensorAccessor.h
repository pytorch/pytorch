#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

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

namespace detail {
// Template classes in torch::headeronly::detail namespace are used
// to construct accessor template classes with custom ArrayRef and
// index bound check implementations. For instance,
// at::TensorAccessor and torch::headeronly::TensorAccessor template
// classes use c10::IntArrayRef and
// torch::headeronly::IntHeaderOnlyArrayRef classes, respectively,
// as return value types of sizes() and strides() methods.

// TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
// For CUDA tensors it is used in device code (only). This means that we
// restrict ourselves to functions and types available there (e.g. IntArrayRef
// isn't).

// The PtrTraits argument is only relevant to cuda to support `__restrict__`
// pointers.
template <
    class ArrayRefCls,
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class TensorAccessorBase {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

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
    class ArrayRefCls,
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class TensorAccessor
    : public TensorAccessorBase<ArrayRefCls, T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<ArrayRefCls, T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_) {}

  C10_HOST_DEVICE TensorAccessor<ArrayRefCls, T, N - 1, PtrTraits, index_t>
  operator[](index_t i) {
    return TensorAccessor<ArrayRefCls, T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }

  C10_HOST_DEVICE const TensorAccessor<
      ArrayRefCls,
      T,
      N - 1,
      PtrTraits,
      index_t>
  operator[](index_t i) const {
    return TensorAccessor<ArrayRefCls, T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }
};

template <
    class ArrayRefCls,
    typename T,
    template <typename U> class PtrTraits,
    typename index_t>
class TensorAccessor<ArrayRefCls, T, 1, PtrTraits, index_t>
    : public TensorAccessorBase<ArrayRefCls, T, 1, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<ArrayRefCls, T, 1, PtrTraits, index_t>(
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
// for CUDA `Tensor`s on the host and as in contrast to `TensorAccessor`s, they
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
    typename ItemAccessor,
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

  C10_DEVICE ItemAccessor operator[](index_t i) {
    index_t* new_sizes = this->sizes_ + 1;
    index_t* new_strides = this->strides_ + 1;
    return ItemAccessor(
        this->data_ + this->strides_[0] * i, new_sizes, new_strides);
  }

  C10_DEVICE const ItemAccessor operator[](index_t i) const {
    const index_t* new_sizes = this->sizes_ + 1;
    const index_t* new_strides = this->strides_ + 1;
    return ItemAccessor(
        this->data_ + this->strides_[0] * i, new_sizes, new_strides);
  }

  /// Returns a PackedTensorAccessor of the same dimension after transposing the
  /// two dimensions given. Does not actually move elements; transposition is
  /// made by permuting the size/stride arrays. If the dimensions are not valid,
  /// asserts.
  C10_HOST GenericPackedTensorAccessor<
      ItemAccessor,
      IndexBoundsCheck,
      T,
      N,
      PtrTraits,
      index_t>
  transpose(index_t dim1, index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    GenericPackedTensorAccessor<
        ItemAccessor,
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
    typename ItemAccessor,
    typename IndexBoundsCheck,
    typename T,
    template <typename U> class PtrTraits,
    typename index_t>
class GenericPackedTensorAccessor<
    ItemAccessor,
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
      ItemAccessor,
      IndexBoundsCheck,
      T,
      1,
      PtrTraits,
      index_t>
  transpose(index_t dim1, index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    return GenericPackedTensorAccessor<
        ItemAccessor,
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
struct HeaderOnlyIndexBoundsCheck {
  HeaderOnlyIndexBoundsCheck(index_t i) {
    STD_TORCH_CHECK(
        0 <= i && i < index_t{N},
        "Index ",
        i,
        " is not within bounds of a tensor of dimension ",
        N);
  }
};
} // anonymous namespace

// HeaderOnlyTensorAccessorBase is same as at::TensorAccessorBase
// except sizes() and strides() return IntHeaderOnlyArrayRef instead
// of IntArrayRef.
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
using HeaderOnlyTensorAccessorBase = detail::TensorAccessorBase<
    torch::headeronly::IntHeaderOnlyArrayRef,
    T,
    N,
    PtrTraits,
    index_t>;

// HeaderOnlyTensorAccessor is same as at::TensorAccessor except
// sizes() and strides() return IntHeaderOnlyArrayRef instead of
// IntArrayRef.
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
using HeaderOnlyTensorAccessor = detail::TensorAccessor<
    torch::headeronly::IntHeaderOnlyArrayRef,
    T,
    N,
    PtrTraits,
    index_t>;

// HeaderOnlyGenericPackedTensorAccessorBase is same as
// at::GenericPackedTensorAccessorBase except sizes() and strides()
// return IntHeaderOnlyArrayRef instead of IntArrayRef.
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
using HeaderOnlyGenericPackedTensorAccessorBase =
    detail::GenericPackedTensorAccessorBase<
        HeaderOnlyIndexBoundsCheck<N, index_t>,
        T,
        N,
        PtrTraits,
        index_t>;

// HeaderOnlyGenericPackedTensorAccessor is same as
// at::GenericPackedTensorAccessor except sizes() and strides() return
// IntHeaderOnlyArrayRef instead of IntArrayRef, and bounds check uses
// STD_TORCH_CHECK instead of TORCH_CHECK_INDEX.
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
using HeaderOnlyGenericPackedTensorAccessor =
    detail::GenericPackedTensorAccessor<
        HeaderOnlyTensorAccessor<T, N - 1, PtrTraits, index_t>,
        HeaderOnlyIndexBoundsCheck<N, index_t>,
        T,
        N,
        PtrTraits,
        index_t>;

} // namespace torch::headeronly
