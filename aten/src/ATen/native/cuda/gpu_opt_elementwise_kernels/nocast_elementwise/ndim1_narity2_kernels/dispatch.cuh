// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim1_narity2_kernels/dispatch.cuh
// Dispatch: launch_elementwise_kernel_1_2_broadcast_in2_dim0_vt_scan, try_launch_gpu_1_2
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/log.h"
#include "broadcast_in2_dim0.cuh"

template <bool swap_tensor, typename func_t, typename array_t>
static inline bool launch_elementwise_kernel_1_2_broadcast_in2_dim0_vt_scan(int64_t numel, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;
  using arg1_t = typename traits::template arg<1>::type;
  constexpr int nt = 128;
  // Per-thread workload in elements: one 16-byte run, split into vt-wide
  // vectors. tws must stay a multiple of every vt instantiated below.
  constexpr int tws = 16 / static_cast<int>(sizeof(res_t));
  using cont_t = std::conditional_t<swap_tensor, arg1_t, arg0_t>;

  // Contiguous operand: swap=true (IN1 broadcast) -> data[2],
  // swap=false (IN2 broadcast) -> data[1].  Gate the result store and input
  // load independently by their actual single ISA access widths. For example,
  // bool <- float at vt=8 uses 8B stores and 16B float loads.
  char* cont_data = swap_tensor ? data[2] : data[1];
  const bool a8 = numel % 8 == 0 &&
      is_vector_access_aligned<res_t, 8>(data[0]) &&
      is_vector_access_aligned<cont_t, 8>(cont_data);
  const bool a4 = numel % 4 == 0 &&
      is_vector_access_aligned<res_t, 4>(data[0]) &&
      is_vector_access_aligned<cont_t, 4>(cont_data);
  const bool a2 = numel % 2 == 0 &&
      is_vector_access_aligned<res_t, 2>(data[0]) &&
      is_vector_access_aligned<cont_t, 2>(cont_data);
  if constexpr (tws % 8 == 0) {
    if (a8) {
      if constexpr (swap_tensor) {
        launch_elementwise_kernel_1_2_broadcast_in2_dim0<nt, 8, tws, res_t, arg1_t, arg0_t, func_t, true>(
            numel, data[0], data[2], data[1], f);
      } else {
        launch_elementwise_kernel_1_2_broadcast_in2_dim0<nt, 8, tws, res_t, arg0_t, arg1_t, func_t>(
            numel, data[0], data[1], data[2], f);
      }
      return true;
    }
  }
  if constexpr (tws % 4 == 0) {
    if (a4) {
      if constexpr (swap_tensor) {
        launch_elementwise_kernel_1_2_broadcast_in2_dim0<nt, 4, tws, res_t, arg1_t, arg0_t, func_t, true>(
            numel, data[0], data[2], data[1], f);
      } else {
        launch_elementwise_kernel_1_2_broadcast_in2_dim0<nt, 4, tws, res_t, arg0_t, arg1_t, func_t>(
            numel, data[0], data[1], data[2], f);
      }
      return true;
    }
  }
  if constexpr (tws % 2 == 0) {
    if (a2) {
      if constexpr (swap_tensor) {
        launch_elementwise_kernel_1_2_broadcast_in2_dim0<nt, 2, tws, res_t, arg1_t, arg0_t, func_t, true>(
            numel, data[0], data[2], data[1], f);
      } else {
        launch_elementwise_kernel_1_2_broadcast_in2_dim0<nt, 2, tws, res_t, arg0_t, arg1_t, func_t>(
            numel, data[0], data[1], data[2], f);
      }
      return true;
    }
  }
  if constexpr (swap_tensor) {
    launch_elementwise_kernel_1_2_broadcast_in2_dim0<nt, 1, tws, res_t, arg1_t, arg0_t, func_t, true>(
        numel, data[0], data[2], data[1], f);
  } else {
    launch_elementwise_kernel_1_2_broadcast_in2_dim0<nt, 1, tws, res_t, arg0_t, arg1_t, func_t>(
        numel, data[0], data[1], data[2], f);
  }
  return true;
}
template <typename func_t, typename array_t>
static inline bool try_launch_gpu_1_2(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;
  using arg1_t = typename traits::template arg<1>::type;

  const int64_t numel = iter.numel();
  if (numel <= 0) {
    return false;
  }
  // TODO: this threshold may be removable.
  // Min numel threshold: avoid launch overhead for tiny workloads.
  if (numel < 256) {
    return false;
  }
  // The vector kernel takes an int N (int32 grid indexing).
  if (numel > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  const int64_t stride00 = iter.strides(0)[0];
  const int64_t stride01 = iter.strides(1)[0];
  const int64_t stride02 = iter.strides(2)[0];
  if (stride00 != static_cast<int64_t>(sizeof(res_t))) {
    return false;
  }
  // IN1 is the broadcast operand: data2 is contiguous, data1 the stride-0
  // scalar. Swap the pointers so data1 holds the contiguous operand and set
  // swap_tensor=true to keep the functor argument order.
  if (stride01 == 0 && !iter.is_cpu_scalar(1) &&
      stride02 == static_cast<int64_t>(sizeof(arg1_t))) {
    log_elementwise_info(iter, "p_e_gpu_1_2_v", f);
    return launch_elementwise_kernel_1_2_broadcast_in2_dim0_vt_scan<true>(numel, f, data);
  }
  // IN2 is the broadcast operand: data1 contiguous, data2 the stride-0
  // scalar.
  if (stride02 == 0 && !iter.is_cpu_scalar(2) &&
      stride01 == static_cast<int64_t>(sizeof(arg0_t))) {
    log_elementwise_info(iter, "p_e_gpu_1_2_v", f);
    return launch_elementwise_kernel_1_2_broadcast_in2_dim0_vt_scan<false>(numel, f, data);
  }
  return false;
}
