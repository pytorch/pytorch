#pragma once

#include <array>
#include <cstdint>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <THC/THCIntegerDivider.cuh>

/// OffsetCalculator calculates the offset in bytes of a linear index for NARGS
/// operands that share the same shape, but may have different strides.

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
  static constexpr int MAX_DIMS = 25;

  // The offset for each argument (in bytes). Wrapper around fixed-size array.
  using offset_type = at::detail::Array<index_t, NARGS>;

  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides) : dims(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>25) dims");
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = IntDivider<index_t>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<index_t>(1);
      }
      for (int arg = 0; arg < NARGS; arg++) {
        strides_[i][arg] =  i < dims ? strides[arg][i] : 0;
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
  IntDivider<index_t> sizes_[MAX_DIMS];
  index_t strides_[MAX_DIMS][NARGS];
};
