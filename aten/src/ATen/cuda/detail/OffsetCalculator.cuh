#pragma once

#include <array>
#include <cstdint>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <THC/THCIntegerDivider.cuh>

/// OffsetCalculator calculates the offset in bytes of a linear index for NARGS
/// operands that share the same shape, but may have different strides.

#ifdef __HIP_PLATFORM_HCC__
constexpr int MAX_DIMS = 16;
#else
constexpr int MAX_DIMS = 25;
#endif

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.
  using offset_type = at::detail::Array<index_t, std::max<int>(NARGS, 1)>;

  // if element_sizes is nullptr, then the strides will be in bytes, otherwise
  // the strides will be in # of elements.
  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides, const int64_t* element_sizes=nullptr) : dims(dims) {
    std::cout << "entering OffsetCalculator" << std::endl;
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    std::cout << "element_sizes = " << element_sizes << std::endl;
    for (int i = 0; i < MAX_DIMS; ++i) {
      std::cout << "i = " << i << std::endl;
      if (i < dims) {
        sizes_[i] = IntDivider<index_t>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<index_t>(1);
      }
      for (int arg = 0; arg < NARGS; arg++) {
        std::cout << "arg = " << arg << std::endl;
        int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        std::cout << "element_size = " << element_size << std::endl;
        std::cout << "strides[arg][i] = " << strides[arg][i] << std::endl;
        std::cout << "strides_[i][arg] = " << strides_[i][arg] << std::endl;
        strides_[i][arg] =  i < dims ? strides[arg][i] / element_size : 0;
      }
    }
    std::cout << "leaving OffsetCalculator" << std::endl;
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
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS][NARGS];
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
