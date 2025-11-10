#pragma once

#include <torch/headeronly/core/TensorAccessor.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace at {

using torch::headeronly::DefaultPtrTraits;
#if defined(__CUDACC__) || defined(__HIPCC__)
  using torch::headeronly::RestrictPtrTraits;
#endif

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
using TensorAccessorBase = torch::headeronly::TensorAccessorBaseDetail<c10::IntArrayRef, T, N, PtrTraits, index_t>;

template <
    class ArrayRefCls,
    typename T,
    size_t N,
    template <typename U> class PtrTraits,
    typename index_t>
class TensorAccessorDetail
    : public torch::headeronly::TensorAccessorBaseDetail<ArrayRefCls, T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessorDetail(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : torch::headeronly::TensorAccessorBaseDetail<ArrayRefCls, T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_) {}

  C10_HOST_DEVICE TensorAccessorDetail<
      ArrayRefCls,
      T,
      N - 1,
      PtrTraits,
      index_t>
  operator[](index_t i) {
    return TensorAccessorDetail<ArrayRefCls, T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }

  C10_HOST_DEVICE const TensorAccessorDetail<
      ArrayRefCls,
      T,
      N - 1,
      PtrTraits,
      index_t>
  operator[](index_t i) const {
    return TensorAccessorDetail<ArrayRefCls, T, N - 1, PtrTraits, index_t>(
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
class TensorAccessorDetail<ArrayRefCls, T, 1, PtrTraits, index_t>
    : public torch::headeronly::TensorAccessorBaseDetail<ArrayRefCls, T, 1, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessorDetail(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
    : torch::headeronly::TensorAccessorBaseDetail<ArrayRefCls, T, 1, PtrTraits, index_t>(
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

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
using TensorAccessor = torch::headeronly::TensorAccessorDetail<c10::IntArrayRef, T, N, PtrTraits, index_t>;

namespace {

template <size_t N, typename index_t>
struct IndexBoundsCheck {
    IndexBoundsCheck(index_t i) {
      TORCH_CHECK_INDEX(
        0 <= i && i < index_t{N},
        "Index ",
        i,
        " is not within bounds of a tensor of dimension ",
        N);
    }
};
}  // anonymous namespace

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
using GenericPackedTensorAccessorBase = torch::headeronly::detail::GenericPackedTensorAccessorBase<IndexBoundsCheck<N, index_t>, T, N, PtrTraits, index_t>;

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
using GenericPackedTensorAccessor = torch::headeronly::detail::GenericPackedTensorAccessor<TensorAccessor, IndexBoundsCheck<N, index_t>, T, N, PtrTraits, index_t>;

// Can't put this directly into the macro function args because of commas
#define AT_X GenericPackedTensorAccessor<T, N, PtrTraits, index_t>

// Old name for `GenericPackedTensorAccessor`
template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
C10_DEFINE_DEPRECATED_USING(PackedTensorAccessor, AT_X)

#undef AT_X

template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor32 = GenericPackedTensorAccessor<T, N, PtrTraits, int32_t>;

template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor64 = GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>;
} // namespace at
