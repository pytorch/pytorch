// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim1_narity3_kernels/dispatch.cuh
// Dispatch: launch_elementwise_kernel_1_3_broadcast_any1dim_vt_scan, try_launch_gpu_1_3
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/log.h"
#include "broadcast_any1dim.cuh"

template <int broadcast_pos, typename func_t>
static inline bool launch_elementwise_kernel_1_3_broadcast_any1dim_vt_scan(
    TensorIteratorBase& iter, int64_t numel,
    char* data0, char* data1, char* data2, char* data3,
    const func_t& f) {
  constexpr int nt = 128;
  for (int vt = 8; vt >= 1; vt /= 2) {
    if (numel % vt != 0) {
      continue;
    }
    const int64_t al = 4 * vt;
    if (reinterpret_cast<uintptr_t>(data0) % al != 0 ||
        reinterpret_cast<uintptr_t>(data1) % al != 0 ||
        reinterpret_cast<uintptr_t>(data2) % al != 0) {
      continue;
    }
    log_elementwise_info(iter, "p_e_gpu_1_3_v", f);
    if (vt == 8) {
      launch_elementwise_kernel_1_3_broadcast_any1dim<nt, 8, broadcast_pos, func_t>(numel, data0, data1, data2, data3, f);
    } else if (vt == 4) {
      launch_elementwise_kernel_1_3_broadcast_any1dim<nt, 4, broadcast_pos, func_t>(numel, data0, data1, data2, data3, f);
    } else if (vt == 2) {
      launch_elementwise_kernel_1_3_broadcast_any1dim<nt, 2, broadcast_pos, func_t>(numel, data0, data1, data2, data3, f);
    } else {
      launch_elementwise_kernel_1_3_broadcast_any1dim<nt, 1, broadcast_pos, func_t>(numel, data0, data1, data2, data3, f);
    }
    return true;
  }
  return false;
}
template <typename func_t, typename array_t>
static inline bool try_launch_gpu_1_3(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;
  using arg1_t = typename traits::template arg<1>::type;
  using arg2_t = typename traits::template arg<2>::type;
  if constexpr (!(std::is_same_v<res_t, float> && std::is_same_v<arg0_t, float> &&
                  std::is_same_v<arg1_t, float> && std::is_same_v<arg2_t, float>)) {
    return false;
  } else {
    const int64_t numel = iter.numel();
    if (numel <= 0 || numel > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    if (numel < 256) {
      return false;
    }
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride02 = iter.strides(2)[0];
    const int64_t stride03 = iter.strides(3)[0];
    if (stride00 != 4) {
      return false;
    }
    // Exactly one broadcast operand; two or more fall back to legacy.
    const int n_bcast = (stride01 == 0) + (stride02 == 0) + (stride03 == 0);
    if (n_bcast != 1) {
      return false;
    }
    const int bcast_idx = (stride01 == 0) ? 1 : (stride02 == 0) ? 2 : 3;
    if (iter.is_cpu_scalar(bcast_idx)) {
      return false;
    }
    // The broadcast operand is scalar-read: its pointer must be 4B aligned.
    if (reinterpret_cast<uintptr_t>(data[bcast_idx]) % 4 != 0) {
      return false;
    }
    // The two non-broadcast operands must be byte-stride == 4 (contiguous),
    // passed as data1/data2 in their original relative order; the broadcast
    // operand goes to data3. bpos = bcast_idx - 1:
    //   IN1 broadcast -> contiguous pair (IN2, IN3), bpos = 0
    //   IN2 broadcast -> contiguous pair (IN1, IN3), bpos = 1
    //   IN3 broadcast -> contiguous pair (IN1, IN2), bpos = 2 (former behavior)
    if (bcast_idx != 1 && stride01 != 4) {
      return false;
    }
    if (bcast_idx != 2 && stride02 != 4) {
      return false;
    }
    if (bcast_idx != 3 && stride03 != 4) {
      return false;
    }
    switch (bcast_idx) {
      case 1: return launch_elementwise_kernel_1_3_broadcast_any1dim_vt_scan<0>(iter, numel, data[0], data[2], data[3], data[1], f);
      case 2: return launch_elementwise_kernel_1_3_broadcast_any1dim_vt_scan<1>(iter, numel, data[0], data[1], data[3], data[2], f);
      case 3: return launch_elementwise_kernel_1_3_broadcast_any1dim_vt_scan<2>(iter, numel, data[0], data[1], data[2], data[3], f);
    }
    return false;
  }
}
