#pragma once

#include <ATen/cuda/DeviceUtils.cuh>

namespace at::native {

enum class WarpReduceDirection { ASCENDING, DESCENDING };

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, WarpReduceDirection direction, typename BinaryOp>
__device__ __forceinline__ void warp_reduce(acc_t* vals, BinaryOp op) {
  #pragma unroll
  for (int offset = (direction == WarpReduceDirection::DESCENDING) ? WARP_SIZE / 2 : 1;
       (direction == WarpReduceDirection::DESCENDING) ? (offset > 0) : (offset < WARP_SIZE);
       offset = (direction == WarpReduceDirection::DESCENDING) ? (offset / 2) : (offset << 1)) {
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR(vals[i], offset, WARP_SIZE);
      vals[i] = op(vals[i], b);
    }
  }
}

} // namespace at::native
