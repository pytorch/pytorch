// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim4_narity2_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_4_2
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/common.cuh"
#include "../../utils/log.h"
#include "broadcast_in2_dim0.cuh"
#include "contiguous_alldim0.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_4_2(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;
  using arg1_t = typename traits::template arg<1>::type;

  if constexpr (!(((sizeof(res_t) == 2 && sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2)) ||
                  ((sizeof(res_t) == 4 && sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4)))) {
    return false;
  } else {
    const int64_t numel = iter.numel();
    // Small-workload threshold keeps this kernel off tiny broadcasts.
    if (numel < 5120) {
      return false;
    }
    const int64_t size0 = iter.shape()[0];
    const int64_t size1 = iter.shape()[1];
    const int64_t size2 = iter.shape()[2];
    const int64_t size3 = iter.shape()[3];
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride02 = iter.strides(2)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride11 = iter.strides(1)[1];
    const int64_t stride12 = iter.strides(2)[1];
    const int64_t stride20 = iter.strides(0)[2];
    const int64_t stride21 = iter.strides(1)[2];
    const int64_t stride22 = iter.strides(2)[2];
    const int64_t stride30 = iter.strides(0)[3];
    const int64_t stride31 = iter.strides(1)[3];
    const int64_t stride32 = iter.strides(2)[3];



    // (p_e_gpu_3_2_a1d0):
    // IN2 is only ever read with a scalar load, so its higher-dim strides are
    // only required to be multiples of the scalar width es (NOT the vector
    // width al), and its data pointer is only checked for natural alignment

    // Do not "restore" the % al checks here to match the vector-read operands:
    // the target layout has stride22 == es, which fails % al for any vt > 1
    // and would collapse vectorization to vt=1. Mutually exclusive with the

    // organizational.
    if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
        stride30 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 * size2 &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride02 == 0 && !iter.is_cpu_scalar(2)) {
      const int vt_max = numel <= 65536 ? 1 : vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        // IN1 is vector-read along dim0: its higher-dim strides must be
        // vector aligned. IN2 (scalar read) higher-dim strides only need es
        // alignment, per the rationale above.
        if (stride11 % al != 0 || stride21 % al != 0 || stride31 % al != 0 ||
            stride12 % static_cast<int64_t>(sizeof(arg1_t)) != 0 ||
            stride22 % static_cast<int64_t>(sizeof(arg1_t)) != 0 ||
            stride32 % static_cast<int64_t>(sizeof(arg1_t)) != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % static_cast<int64_t>(sizeof(arg1_t)) != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_4_2_a1d0", f);
        constexpr int nt = 128;
        if (vt == 8) {
          launch_elementwise_kernel_4_2_broadcast_in2_dim0<nt, 8, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2, size3,
              stride00, stride01, stride02, stride10, stride11, stride12,
              stride20, stride21, stride22, stride30, stride31, stride32, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_4_2_broadcast_in2_dim0<nt, 4, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2, size3,
              stride00, stride01, stride02, stride10, stride11, stride12,
              stride20, stride21, stride22, stride30, stride31, stride32, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_4_2_broadcast_in2_dim0<nt, 2, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2, size3,
              stride00, stride01, stride02, stride10, stride11, stride12,
              stride20, stride21, stride22, stride30, stride31, stride32, f);
        } else {
          launch_elementwise_kernel_4_2_broadcast_in2_dim0<nt, 1, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2, size3,
              stride00, stride01, stride02, stride10, stride11, stride12,
              stride20, stride21, stride22, stride30, stride31, stride32, f);
        }
        return true;
      }
    }

    if (stride00 != static_cast<int64_t>(sizeof(res_t)) ||
        stride10 != static_cast<int64_t>(sizeof(res_t)) * size0 ||
        stride20 != static_cast<int64_t>(sizeof(res_t)) * size0 * size1 ||
        stride30 != static_cast<int64_t>(sizeof(res_t)) * size0 * size1 * size2 ||
        stride01 != static_cast<int64_t>(sizeof(arg0_t)) ||
        stride02 != static_cast<int64_t>(sizeof(arg1_t))) {
      return false;
    }
    const int vt_max = numel <= 65536 ? 1 : vector_access_width<res_t>();
    for (int vt = vt_max; vt >= 1; vt /= 2) {
      if (size0 % vt != 0) {
        continue;
      }
      const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
      if (stride11 % al != 0 || stride21 % al != 0 || stride31 % al != 0 ||
          stride12 % al != 0 || stride22 % al != 0 || stride32 % al != 0) {
        continue;
      }
      if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
          reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
          reinterpret_cast<uintptr_t>(data[2]) % al != 0) {
        continue;
      }
      log_elementwise_info(iter, "p_e_gpu_4_2_d0c", f);
      constexpr int nt = 128;
      if (vt == 8) {
        launch_elementwise_kernel_4_2_contiguous_alldim0<nt, 8, res_t, arg0_t, arg1_t>(
            numel, data[0], data[1], data[2], size0, size1, size2, size3,
            stride00, stride01, stride02, stride10, stride11, stride12,
            stride20, stride21, stride22, stride30, stride31, stride32, f);
      } else if (vt == 4) {
        launch_elementwise_kernel_4_2_contiguous_alldim0<nt, 4, res_t, arg0_t, arg1_t>(
            numel, data[0], data[1], data[2], size0, size1, size2, size3,
            stride00, stride01, stride02, stride10, stride11, stride12,
            stride20, stride21, stride22, stride30, stride31, stride32, f);
      } else if (vt == 2) {
        launch_elementwise_kernel_4_2_contiguous_alldim0<nt, 2, res_t, arg0_t, arg1_t>(
            numel, data[0], data[1], data[2], size0, size1, size2, size3,
            stride00, stride01, stride02, stride10, stride11, stride12,
            stride20, stride21, stride22, stride30, stride31, stride32, f);
      } else {
        launch_elementwise_kernel_4_2_contiguous_alldim0<nt, 1, res_t, arg0_t, arg1_t>(
            numel, data[0], data[1], data[2], size0, size1, size2, size3,
            stride00, stride01, stride02, stride10, stride11, stride12,
            stride20, stride21, stride22, stride30, stride31, stride32, f);
      }
      return true;
    }
    return false;
  }
}
