#include "ATen/native/cpu/ReduceOpsKernel.h"
#include "ATen/Dispatch.h"
#include <iostream>

namespace at {
namespace native {

using namespace vec256;

// This adds the content of arr to sum
template <class scalar_t, template <class> class OP, CPUCapability C>
inline scalar_t allreduce_kernel_(const scalar_t *arr, size_t start, size_t end,
                                  scalar_t sum) {
  Vec256<scalar_t> part_sum;
  // Use all 16 registers.
  Vec256<scalar_t> tmp_sum[4], tmp_sum1, tmp_sum2, tmp_sum3;
  Vec256<scalar_t> a[8];
  size_t width =
      256 / sizeof(scalar_t); // primitives per 256 bytes (two cache lines)
  size_t epr = 32 / sizeof(scalar_t); // primitives per Vec256
  size_t k = 0;
  for (; k < (end - start) / width; k++) {
    for (size_t i = 0; i < 8; i++) {
      a[i].load(arr + (k * width) + i * epr + start);
    }
    for (size_t i = 0; i < 8; i += 2) {
      tmp_sum[i / 2] = OP<Vec256<scalar_t>>()(a[i], a[i + 1]);
    }
    tmp_sum1 = OP<Vec256<scalar_t>>()(tmp_sum[0], tmp_sum[1]);
    tmp_sum2 = OP<Vec256<scalar_t>>()(tmp_sum[2], tmp_sum[3]);
    if (k == 0) {
      part_sum = OP<Vec256<scalar_t>>()(tmp_sum1, tmp_sum2);
    } else {
      tmp_sum3 = OP<Vec256<scalar_t>>()(tmp_sum1, tmp_sum2);
      part_sum = OP<Vec256<scalar_t>>()(part_sum, tmp_sum3);
    }
  }
  if (k > 0) {
    scalar_t sarr[32 / sizeof(scalar_t)];
    part_sum.store(sarr);
    for (size_t i = 0; i < part_sum.size(); i++) {
      sum = OP<scalar_t>()(sum, sarr[i]);
    }
  }
  k = k * width + start;
  for (; k < end; k++) {
    sum = OP<scalar_t>()(sum, arr[k]);
  }
  return sum;
}

// This overwrites the content of outarr
template <class scalar_t, template <class> class OP, CPUCapability C>
inline void dimreduce_kernel_(const scalar_t *arr, scalar_t *outarr,
                              size_t num_rows, size_t num_cols) {
  size_t width =
      256 / (sizeof(scalar_t)); // primitives per 256 bytes (two cache lines)
  Vec256<scalar_t> a[8];
  Vec256<scalar_t> b[8];
  constexpr size_t epr = 32 / sizeof(scalar_t); // primitives per Vec256
  size_t tile = 0;
  for (; tile < (num_cols) / width; tile++) {
    size_t row_ind = tile * width;
    for (size_t i = 0; i < num_rows; i += 1) {
      for (int ib = 0; ib < 8; ib++) {
        if (i == 0) {
          b[ib].load(arr + i * num_cols + tile * width + ib * epr);
        } else {
          a[ib].load(arr + i * num_cols + tile * width + ib * epr);
          b[ib] = OP<Vec256<scalar_t>>()(b[ib], a[ib]);
        }
      }
    }
    for (int ib = 0; ib < 8; ib++) {
      b[ib].store(outarr + row_ind + ib * epr);
    }
  }
  size_t k = tile * width;
  for (; k < num_cols; k++) {
    for (size_t i = 0; i < num_rows; i += 1) {
      if (i == 0) {
        outarr[k] = arr[i * num_cols + k];
      } else {
        outarr[k] = OP<scalar_t>()(outarr[k], arr[i * num_cols + k]);
      }
    }
  }
}

template <template <class> class OP, CPUCapability C>
inline void allImpl(Tensor & result, const Tensor & self, size_t dim, bool all, const char* name, int64_t init) {
  AT_DISPATCH_ALL_TYPES(self.type(), name, [&] {
    if (all) {
      result.fill_(parallel_reduce<scalar_t, OP>(
          &allreduce_kernel_<scalar_t, OP, CURRENT_CAPABILITY>, self.data<scalar_t>(),
          (size_t)0, (size_t)self.numel(), (scalar_t)init));
    } else {
      parallel_for_2d<scalar_t>(&dimreduce_kernel_<scalar_t, OP, CURRENT_CAPABILITY>,
                                self.sizes()[dim], self.strides()[dim],
                                self.numel(), self.data<scalar_t>(),
                                result.data<scalar_t>());
    }
    });
}

template <>
void sumImplC<CURRENT_CAPABILITY>::function(Tensor &result, const Tensor &self,
                                            size_t dim, bool all) {
  allImpl<std::plus, CURRENT_CAPABILITY>(result, self, dim, all, "sum", 0);
}

template <>
void prodImplC<CURRENT_CAPABILITY>::function(Tensor &result, const Tensor &self,
                                             size_t dim, bool all) {
  allImpl<std::multiplies, CURRENT_CAPABILITY>(result, self, dim, all, "prod", 1);
}
}
}
