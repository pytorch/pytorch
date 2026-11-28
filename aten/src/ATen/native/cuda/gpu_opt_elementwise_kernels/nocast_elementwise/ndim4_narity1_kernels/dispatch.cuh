// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim4_narity1_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_4_1
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/common.cuh"
#include "../../utils/log.h"
#include "contiguous_alldim0.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_4_1(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;

  if constexpr (!(sizeof(res_t) == sizeof(arg0_t) &&
                  (sizeof(res_t) == 2 || sizeof(res_t) == 4))) {
    return false;
  } else {
    const int64_t numel = iter.numel();
    if (numel <= 0) {
      return false;
    }
    const int64_t size0 = iter.shape()[0];
    const int64_t size1 = iter.shape()[1];
    const int64_t size2 = iter.shape()[2];
    const int64_t size3 = iter.shape()[3];
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride11 = iter.strides(1)[1];
    const int64_t stride20 = iter.strides(0)[2];
    const int64_t stride21 = iter.strides(1)[2];
    const int64_t stride30 = iter.strides(0)[3];
    const int64_t stride31 = iter.strides(1)[3];


    // kernel computes both offsets from strides directly, so OUT row gaps
    // (sliced views of larger tensors) are safe; the vt loop below pins the

    // in try_launch_gpu_3_1).
    if (stride00 != static_cast<int64_t>(sizeof(res_t)) ||
        stride01 != static_cast<int64_t>(sizeof(arg0_t))) {
      return false;
    }
    const int vt_max = vector_access_width<res_t>();
    // vt < 4 (dim0-stride gaps of 4/8 bytes, e.g. 4-d copy layouts like
    // (1280,720,96,4) or (H,W,5,n)) explodes the grid and regresses below
    // legacy; require vt >= 4 alignment up front and let those shapes fall
    // through to legacy.
    if (size0 % 4 != 0 ||
        stride10 % (static_cast<int64_t>(sizeof(res_t)) * 4) != 0 ||
        stride20 % (static_cast<int64_t>(sizeof(res_t)) * 4) != 0 ||
        stride30 % (static_cast<int64_t>(sizeof(res_t)) * 4) != 0 ||
        stride11 % (static_cast<int64_t>(sizeof(res_t)) * 4) != 0 ||
        stride21 % (static_cast<int64_t>(sizeof(res_t)) * 4) != 0 ||
        stride31 % (static_cast<int64_t>(sizeof(res_t)) * 4) != 0) {
      return false;
    }
    for (int vt = vt_max; vt >= 4; vt /= 2) {
      if (size0 % vt != 0) {
        continue;
      }
      const int64_t al = static_cast<int64_t>(sizeof(arg0_t)) * vt;
      // OUT rows are written with vt-wide vector stores at idx1 * stride10 +
      // idx2 * stride20 + idx3 * stride30, so the OUT higher-dim strides must
      // stay vector aligned too (the entry precondition only pins stride00
      // and stride01).
      if (stride10 % al != 0 || stride20 % al != 0 || stride30 % al != 0 ||
          stride11 % al != 0 || stride21 % al != 0 || stride31 % al != 0) {
        continue;
      }
      if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
          reinterpret_cast<uintptr_t>(data[1]) % al != 0) {
        continue;
      }
      log_elementwise_info(iter, "p_e_gpu_4_1_d0c", f);
      constexpr int nt = 128;
      if (vt == 8) {
        launch_elementwise_kernel_4_1_contiguous_alldim0<nt, 8, res_t, arg0_t>(
            numel, data[0], data[1], size0, size1, size2, size3,
            stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, f);
      } else {
        launch_elementwise_kernel_4_1_contiguous_alldim0<nt, 4, res_t, arg0_t>(
            numel, data[0], data[1], size0, size1, size2, size3,
            stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, f);
      }
      return true;
    }
    return false;
  }
}
