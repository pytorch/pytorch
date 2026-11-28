// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim2_narity1_kernels/dispatch.cuh
// Dispatch: launch_elementwise_kernel_2_1_broadcast_any1dim_vt_scan, try_launch_gpu_2_1
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/common.cuh"
#include "../../utils/log.h"
#include "broadcast_any1dim.cuh"
#include "contiguous_alldim0.cuh"
#include "transpose_reverse.cuh"
#include "transpose_tiny.cuh"

template <template <int, typename, typename, typename> typename Policy,
          typename res_t, typename arg0_t, typename func_t>
static inline bool launch_elementwise_kernel_2_1_broadcast_any1dim_vt_scan(
    TensorIteratorBase& iter, int64_t numel,
    char* data0, char* data1,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01,
    int64_t stride10, int64_t stride11,
    const func_t& f) {
  const int vt_max = per_op_vector_width<res_t>();
  constexpr int nt = 128;
  for (int vt = vt_max; vt >= 1; vt /= 2) {
    if (size0 % vt != 0) {
      continue;
    }
    const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
    if (stride10 % al != 0) {
      continue;
    }
    if (reinterpret_cast<uintptr_t>(data0) % al != 0) {
      continue;
    }
    if constexpr (Policy<1, res_t, arg0_t, func_t>::arg_needs_align) {
      if (reinterpret_cast<uintptr_t>(data1) % al != 0) {
        continue;
      }
    }
    log_elementwise_info(iter, "p_e_gpu_2_1_1db", f);
    if constexpr (sizeof(res_t) == 1) {
      if (vt == 16) {
        launch_elementwise_kernel_2_1_broadcast_any1dim<Policy<16, res_t, arg0_t, func_t>, nt, 16, res_t, arg0_t>(
            numel, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
        return true;
      }
    }
    if (vt == 8) {
      launch_elementwise_kernel_2_1_broadcast_any1dim<Policy<8, res_t, arg0_t, func_t>, nt, 8, res_t, arg0_t>(
          numel, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
    } else if (vt == 4) {
      launch_elementwise_kernel_2_1_broadcast_any1dim<Policy<4, res_t, arg0_t, func_t>, nt, 4, res_t, arg0_t>(
          numel, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
    } else if (vt == 2) {
      launch_elementwise_kernel_2_1_broadcast_any1dim<Policy<2, res_t, arg0_t, func_t>, nt, 2, res_t, arg0_t>(
          numel, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
    } else {
      launch_elementwise_kernel_2_1_broadcast_any1dim<Policy<1, res_t, arg0_t, func_t>, nt, 1, res_t, arg0_t>(
          numel, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
    }
    return true;
  }
  return false;
}
template <typename func_t, typename array_t>
static inline bool try_launch_gpu_2_1(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;

  const int64_t numel = iter.numel();
  if (numel <= 0) {
    return false;
  }
  // TODO: Tiny patterns (2_1_tiny, numel ≤ 1024): launch overhead dominates, fall
  // through to legacy path instead of attempting GPU matching.
  // TODO: remove this threshold once its necessity is understood.

  // slice-view copies, and the two transpose families in general below the
  // shmem-tile size gates). The legacy path pays the full divmod offset
  // computation for these small tiles; the scalar tiny kernel beats it.
  if (numel <= 1024) {
    if constexpr (sizeof(res_t) == sizeof(arg0_t) &&
                  (sizeof(res_t) == 2 || sizeof(res_t) == 4 || sizeof(res_t) == 8)) {
      const int64_t t_size0 = iter.shape()[0];
      const int64_t t_size1 = iter.shape()[1];
      const int64_t t_stride00 = iter.strides(0)[0];
      const int64_t t_stride01 = iter.strides(1)[0];
      const int64_t t_stride10 = iter.strides(0)[1];
      const int64_t t_stride11 = iter.strides(1)[1];
      const int64_t t_es = static_cast<int64_t>(sizeof(res_t));
      if (numel == t_size0 * t_size1 && t_size0 <= 64 && t_size1 <= 64 &&
          t_stride00 == t_es) {

        // f(in[i][j]), both dim0 strides == es, dim1 strides may carry gaps
        // (rows are slices of larger tensors).
        if (t_stride01 == t_es &&
            t_stride10 >= t_es * t_size0 && t_stride10 % t_es == 0 &&
            t_stride11 >= t_es * t_size0 && t_stride11 % t_es == 0) {
          log_elementwise_info(iter, "p_e_gpu_2_1_tp_tiny", f);
          constexpr int t_nt = 128;
          launch_elementwise_kernel_2_1_transpose_tiny<t_nt, res_t, arg0_t>(
              numel, data[0], data[1], t_size0, t_size1,
              t_stride00, t_stride01, t_stride10, t_stride11, f);
          return true;
        }

        // OUT, i.e. IN dim1-contiguous): out[i][j] = f(in[i][j]) with
        // stride01 = row-length gap.
        if (t_stride11 == t_es &&
            t_stride10 >= t_es * t_size0 && t_stride10 % t_es == 0 &&
            t_stride01 >= t_es * t_size1 && t_stride01 % t_es == 0) {
          log_elementwise_info(iter, "p_e_gpu_2_1_tp_tiny", f);
          constexpr int t_nt = 128;
          launch_elementwise_kernel_2_1_transpose_tiny<t_nt, res_t, arg0_t>(
              numel, data[0], data[1], t_size0, t_size1,
              t_stride00, t_stride01, t_stride10, t_stride11, f);
          return true;
        }
      }
    }
    return false;
  }

  if constexpr (sizeof(res_t) == sizeof(arg0_t) &&
                (sizeof(res_t) == 2 || sizeof(res_t) == 4 || sizeof(res_t) == 8)) {
    // Both operands contiguous on dim0; dim1 strides and base pointers must
    // be multiples of the vector width, and size0 % vt == 0 keeps vector
    // accesses inside a row.
    const int64_t size0 = iter.shape()[0];
    const int64_t size1 = iter.shape()[1];
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride11 = iter.strides(1)[1];


    // forms share one guard and one helper; the strip Policy picks the read
    // shape (scalar row read + splat store vs hoisted vector read +

    // satisfies d0c (stride11 == 0 trivially passes its stride11 % al test,
    // and stride10 == es * size0 passes stride10 % al once size0 % vt == 0),
    // so d0c would capture it first and the row reuse would be lost. Any

    const bool bd0 = (stride01 == 0 && stride11 == static_cast<int64_t>(sizeof(arg0_t)));
    const bool bd1 = (stride01 == static_cast<int64_t>(sizeof(arg0_t)) && stride11 == 0);
    if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        !iter.is_cpu_scalar(1) && (bd0 || bd1)) {
      if (bd0) {
        if (launch_elementwise_kernel_2_1_broadcast_any1dim_vt_scan<strip_bd0, res_t, arg0_t>(
                iter, numel, data[0], data[1], size0, size1,
                stride00, stride01, stride10, stride11, f)) {
          return true;
        }
      } else if (launch_elementwise_kernel_2_1_broadcast_any1dim_vt_scan<strip_bd1, res_t, arg0_t>(
                     iter, numel, data[0], data[1], size0, size1,
                     stride00, stride01, stride10, stride11, f)) {
        return true;
      }
    }


    // Float32 row-gap tiles whose grid covers fewer than half the device's
    // SMs run faster on the legacy 1D kernel (more parallelism, no row-grid
    // launch overhead); preserve the specialized path for bf16 and for
    // larger grids. Mirror the launcher's grid: ceil(size0 / (vt * 128))
    // column blocks by min(ceil(size1 / 8), 2048) row blocks.
    if constexpr (sizeof(res_t) == 4 && sizeof(arg0_t) == 4) {
      const int vt = vector_access_width<res_t>();
      const int64_t grid_x =
          (size0 + static_cast<int64_t>(vt) * 128 - 1) / (static_cast<int64_t>(vt) * 128);
      const int64_t grid_y = std::min<int64_t>((size1 + 7) / 8, 2048);
      if (grid_x * grid_y < gpu_sm_count() / 2 &&
          stride00 == static_cast<int64_t>(sizeof(res_t)) &&
          stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
          stride10 > static_cast<int64_t>(sizeof(res_t)) * size0 &&
          stride11 == static_cast<int64_t>(sizeof(arg0_t)) * size0) {
        return false;
      }
    }

    if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t))) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (stride10 % al != 0 || stride11 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_2_1_d0c", f);

        if (vt == 8) {
          launch_elementwise_kernel_2_1_contiguous_alldim0<8, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_2_1_contiguous_alldim0<4, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_2_1_contiguous_alldim0<2, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        } else {
          launch_elementwise_kernel_2_1_contiguous_alldim0<1, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        }
        return true;
      }
    }



    // Bidirectional row strides allow gaps; element-size alignment only
    // (coalesced scalar access at element granularity).
    if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride11 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride10 >= static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride10 % static_cast<int64_t>(sizeof(res_t)) == 0 &&
        stride01 >= static_cast<int64_t>(sizeof(arg0_t)) * size1 &&
        stride01 % static_cast<int64_t>(sizeof(arg0_t)) == 0 &&
        size0 >= 128 && size1 >= 64 && sizeof(res_t) <= 4) {
      log_elementwise_info(iter, "p_e_gpu_2_1_tpr", f);
      if constexpr (sizeof(res_t) == 2) {
        // 32x16 keeps 512 threads on bf16 128/256 square matrices and shortens
        // the recomputation path.
        launch_elementwise_kernel_2_1_transpose_reverse<32, 16, res_t, arg0_t>(
            numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
      } else {
        launch_elementwise_kernel_2_1_transpose_reverse<32, 8, res_t, arg0_t>(
            numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
      }
      return true;
    }




    // x * stride00, which is only correct for stride00 == sizeof(res_t). All

    // here has stride00 > es and must use the scalar store (vt == 1).
    if (stride01 == static_cast<int64_t>(sizeof(arg0_t))) {
      const int vt_max = (stride00 == static_cast<int64_t>(sizeof(res_t)))
                             ? vector_access_width<res_t>()
                             : 1;
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (stride00 % al != 0 || stride10 % al != 0 || stride11 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_2_1_d0c", f);

        if (vt == 8) {
          launch_elementwise_kernel_2_1_contiguous_alldim0<8, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_2_1_contiguous_alldim0<4, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_2_1_contiguous_alldim0<2, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        } else {
          launch_elementwise_kernel_2_1_contiguous_alldim0<1, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, stride00, stride01, stride10, stride11, f);
        }
        return true;
      }
    }
    return false;
  } else if constexpr (sizeof(res_t) == 1 && sizeof(arg0_t) == 1) {
    // dim0-broadcast form (Bool): the input is stride-0 on dim0 and one
    // value feeds an entire row, so the functor is evaluated once per row.
    const int64_t size0 = iter.shape()[0];
    const int64_t size1 = iter.shape()[1];
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride11 = iter.strides(1)[1];

    if (stride00 != static_cast<int64_t>(sizeof(res_t)) || stride01 != 0 ||
        stride11 != static_cast<int64_t>(sizeof(arg0_t)) ||
        iter.is_cpu_scalar(1)) {
      return false;
    }

    // alignment and the helper's unconditional stride10 % al check supplies
    // the gap guard of the former inline scan.
    return launch_elementwise_kernel_2_1_broadcast_any1dim_vt_scan<strip_bd0, res_t, arg0_t>(
        iter, numel, data[0], data[1], size0, size1,
        stride00, stride01, stride10, stride11, f);
  } else {
    return false;
  }
}
