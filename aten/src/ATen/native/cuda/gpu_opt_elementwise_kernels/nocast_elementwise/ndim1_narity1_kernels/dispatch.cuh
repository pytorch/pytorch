// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim1_narity1_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_1_1
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/common.cuh"
#include "../../utils/log.h"
#include "broadcast.cuh"
#include "strided.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_1_1(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;

  const int64_t numel = iter.numel();
  if (numel <= 0 || numel > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  if (iter.ndim() != 1) {
    return false;
  }
  if (iter.strides(0)[0] != iter.element_size(0)) {
    return false;
  }
  // IN1 strided gather (non-scalar, non-contiguous). One element per
  // thread with an element-wise load; the broadcast branch below requires
  // stride01 == 0, and the contiguous case stays on the legacy path.
  // tag p_e_gpu_1_1_s. Mutually exclusive with p_e_gpu_1_1_b (stride01 == 0).
  if (iter.strides(1)[0] != 0 && !iter.is_cpu_scalar(1) &&
      iter.strides(1)[0] != static_cast<int64_t>(iter.element_size(1))) {
    if (reinterpret_cast<uintptr_t>(data[0]) % static_cast<uintptr_t>(iter.element_size(0)) == 0 &&
        reinterpret_cast<uintptr_t>(data[1]) % static_cast<uintptr_t>(iter.element_size(1)) == 0) {
      log_elementwise_info(iter, "p_e_gpu_1_1_s", f);
      constexpr int nt = 256;
      using arg0_t = typename traits::template arg<0>::type;
      launch_gpu_1_1_strided<nt, res_t, arg0_t, func_t>(
          numel, data[0], data[1], iter.strides(0)[0], iter.strides(1)[0], f);
      return true;
    }
  }
  if (iter.strides(1)[0] != 0 || iter.is_cpu_scalar(1)) {
    return false;
  }
  constexpr int vt = vector_access_width<res_t>();
  if (!is_vector_access_aligned<res_t, vt>(data[0])) {
    return false;
  }

  log_elementwise_info(iter, "p_e_gpu_1_1_b", f);
  constexpr int nt = static_cast<int>(num_threads());
  constexpr int tws = elems_per_thread<calc_io_size<func_t>()>();
  using bcast_t = typename traits::template arg<0>::type;
  launch_gpu_1_1_broadcast<nt, vt, tws, res_t, bcast_t, func_t>(
      numel, data[0], data[1], f);
  return true;
}
