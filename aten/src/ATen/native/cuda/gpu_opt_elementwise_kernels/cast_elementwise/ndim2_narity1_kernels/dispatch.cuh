// File: gpu_opt_elementwise_kernels/cast_elementwise/ndim2_narity1_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_cast_elementwise_2_1
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/log.h"
#include "broadcast_dim1.cuh"
#include "gap_copy.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_cast_elementwise_2_1(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;

  // Long dim1-broadcast copy). The functor in the cast path either takes
  // the output dtype directly (float(float) via LoadWithCast, e.g.
  // direct_copy_kernel_cuda) or the input dtype (float(int64_t), e.g.
  // quantized dequant kernels), so arg0_t is restricted to {float, int64_t}
  // to keep this kernel away from every other dtype mix. FillFunctor-style
  // arity-0 functors reach this function too, so arg0_t is extracted only
  // under the arity==1 guard.
  if constexpr (traits::arity == 1) {
    using arg0_t = typename traits::template arg<0>::type;
    if constexpr (std::is_same_v<res_t, float> &&
                  (std::is_same_v<arg0_t, float> || std::is_same_v<arg0_t, int64_t>)) {
      const int64_t numel = iter.numel();
      if (numel <= 0) {
        return false;
      }
      if (iter.ndim() != 2) {
        return false;
      }
      if (iter.dtype(0) != ScalarType::Float) {
        return false;
      }
      const ScalarType d1 = iter.dtype(1);
      if (d1 != ScalarType::Long && d1 != ScalarType::BFloat16) {
        return false;
      }
      const int64_t size0 = iter.shape()[0];
      const int64_t size1 = iter.shape()[1];
      const int64_t stride00 = iter.strides(0)[0];  // out d0 (bytes)
      const int64_t stride01 = iter.strides(1)[0];  // in  d0 (bytes)
      const int64_t stride10 = iter.strides(0)[1];  // out d1 (bytes)
      const int64_t stride11 = iter.strides(1)[1];  // in  d1 (bytes)


      // the dim1 strides may carry gaps (IN rows are slices of a larger
      // tensor). Vectorized per 16B chunk, row-major, y_t-tiled grid.
      // Row starts must stay 16B-aligned for the vector loads/stores, and
      // the y_t-tiled grid keeps gridDim.y within the hardware bound.
      if (d1 == ScalarType::BFloat16) {
        if constexpr (std::is_same_v<arg0_t, float>) {
        if (stride00 == static_cast<int64_t>(sizeof(float)) &&
            stride01 == static_cast<int64_t>(sizeof(c10::BFloat16)) &&
            stride10 % 16 == 0 && stride11 % 16 == 0 &&
            size0 % 8 == 0 && size1 <= gpu_grid_cap(1) * 8 &&
            reinterpret_cast<uintptr_t>(data[0]) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(data[1]) % 16 == 0) {
          log_elementwise_info(iter, "p_e_gpu_2_1_cast_gap", f);
          constexpr int nt = 128;
          launch_cast_elementwise_kernel_2_1_gap_copy<nt>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
          return true;
        }
        }
        return false;
      }

      // Long path (2_1_cb1): OUT fully contiguous, IN contiguous on dim0 and
      // broadcast on dim1.
      if (stride00 != static_cast<int64_t>(sizeof(float)) ||
          stride10 != static_cast<int64_t>(sizeof(float)) * size0 ||
          stride01 != static_cast<int64_t>(sizeof(int64_t)) ||
          stride11 != 0 || iter.is_cpu_scalar(1)) {
        return false;
      }
      constexpr int vt_max = 4;  // 16 bytes of float
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al_out = static_cast<int64_t>(sizeof(float)) * vt;
        const int64_t al_in = static_cast<int64_t>(sizeof(int64_t)) * vt;
        if (reinterpret_cast<uintptr_t>(data[0]) % al_out != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al_in != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_2_1_cb1", f);
        constexpr int nt = 128;
        if (vt == 4) {
          launch_cast_elementwise_kernel_2_1_broadcast_dim1<nt, 4>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        } else if (vt == 2) {
          launch_cast_elementwise_kernel_2_1_broadcast_dim1<nt, 2>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        } else {
          launch_cast_elementwise_kernel_2_1_broadcast_dim1<nt, 1>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        }
        return true;
      }
      return false;
    }
  }
  return false;
}
