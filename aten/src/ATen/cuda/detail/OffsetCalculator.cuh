#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/IntegerDivider.cuh>

// If element_sizes is nullptr, then the strides will be in bytes, otherwise
// the strides will be in # of elements.
// Operands that share the same shape, but may have different strides.
// OffsetCalculator iterates the tensor in a column-major order

#if defined(USE_ROCM)
constexpr int MAX_DIMS = 16;
#else
constexpr int MAX_DIMS = 25;
#endif

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
  // We allow having negative strides to implement some operations like torch.flip
  using stride_t = std::conditional_t<signed_strides,
                                      std::make_signed_t<index_t>,
                                      index_t>;
  // The offset for each argument. Wrapper around fixed-size array.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.
  using offset_type = at::detail::Array<stride_t, std::max<int>(NARGS, 1)>;

  // if element_sizes is nullptr, then the strides will be in bytes, otherwise
  // the strides will be in # of elements.
  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides, const int64_t* element_sizes=nullptr) : dims(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    for (int i=0; i < dims; i++){
      sizes_[i] = at::cuda::detail::IntDivider<index_t>(sizes[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = strides[arg][i] / element_size;
      }
    }
  }

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }

    }
    return offsets;
  }

  int dims;
  at::cuda::detail::IntDivider<index_t> sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};

template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // The offsets are in # of elements, not in bytes.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.
  using offset_type = at::detail::Array<index_t, std::max<int>(NARGS, 1)>;

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};

// Make an OffsetCalculator with byte offsets
template<int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_offset_calculator(const at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N, uint32_t, signed_strides>(iter.ndim(), iter.shape().data(), strides.data());
}

// Make an OffsetCalculator with element offsets
template<int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_element_offset_calculator(
    const at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  std::array<int64_t, N> element_sizes;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<N, uint32_t, signed_strides>(
      iter.ndim(), iter.shape().data(), strides.data(), element_sizes.data());
}
