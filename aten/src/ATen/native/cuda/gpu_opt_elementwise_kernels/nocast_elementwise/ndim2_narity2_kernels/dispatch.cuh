// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim2_narity2_kernels/dispatch.cuh
// Dispatch: launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d_vt_scan, launch_elementwise_kernel_2_2_contiguous_in1_dim0_vt_scan, try_launch_gpu_2_2
// Tags: dispatched by the functions below.
//
// Numerical note: inlining the same functor into this family's kernel skeleton
// can lead the compiler to different fma-contraction decisions than the stock
// legacy/vectorized skeletons (all are valid evaluations under fmad=true), so
// outputs may differ at the bit level in a value-dependent, position-independent
// way. Functors with long formula chains such as erf/exp (e.g. gelu_backward)
// can amplify intermediate rounding differences in cancellation regions (tens of
// ULP). Correctness for such ops is checked with a per-op ULP tolerance rather
// than exact bit equality.

#pragma once

#include "../../utils/common.cuh"
#include "../../utils/log.h"
#include "broadcast_any1dim_grid2d.cuh"
#include "broadcast_in2_dim0_grid1d.cuh"
#include "contiguous_in1_dim0.cuh"
#include "crossbroadcast_in1_dim0_in2_dim1.cuh"
#include "crossbroadcast_in1_dim1_in2_dim0.cuh"

template <template <int, typename, typename, typename, typename> typename Policy,
          typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static inline bool launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d_vt_scan(
    TensorIteratorBase& iter, int64_t numel,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    const func_t& f) {
  using PolicyT = Policy<1, res_t, arg0_t, arg1_t, func_t>;
  constexpr int nt = PolicyT::nt;
  constexpr bool has_1d_fallback = PolicyT::has_1d_fallback;
  constexpr int64_t y_t = 8;
  // Keep the historical 65535 cap: above 65535 * y_t the 1D fallback kernel
  // wins (measured 1.3-2.0x at size1 524288, 2_2 1db 64M).
  const bool use_2d = has_1d_fallback ? (size1 <= gpu_grid_cap(1) * y_t) : true;
  const int vt_max = per_op_vector_width<res_t>();
  for (int vt = vt_max; vt >= 1; vt /= 2) {
    if (size0 % vt != 0) {
      continue;
    }
    const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
    if (reinterpret_cast<uintptr_t>(data0) % al != 0 ||
        reinterpret_cast<uintptr_t>(data1) % al != 0) {
      continue;
    }
    if constexpr (PolicyT::row_stride_is_s11) {
      if (stride11 % al != 0) {
        continue;
      }
    } else {
      if (stride12 % al != 0) {
        continue;
      }
    }
    if constexpr (PolicyT::data2_vec_read) {
      const int64_t al_arg1 = static_cast<int64_t>(sizeof(arg1_t)) * vt;
      if (reinterpret_cast<uintptr_t>(data2) % al_arg1 != 0) {
        continue;
      }
    }
    log_elementwise_info(iter, use_2d ? PolicyT::tag_2d : PolicyT::tag_1d, f);
    if constexpr (has_1d_fallback) {
      if (!use_2d) {
        if (vt == 8) {
          launch_elementwise_kernel_2_2_broadcast_in2_dim0_grid1d<nt, 8, res_t, arg0_t, arg1_t>(
              numel, data0, data1, data2, size0, size1,
              stride00, stride01, stride02, stride10, stride11, stride12, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_2_2_broadcast_in2_dim0_grid1d<nt, 4, res_t, arg0_t, arg1_t>(
              numel, data0, data1, data2, size0, size1,
              stride00, stride01, stride02, stride10, stride11, stride12, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_2_2_broadcast_in2_dim0_grid1d<nt, 2, res_t, arg0_t, arg1_t>(
              numel, data0, data1, data2, size0, size1,
              stride00, stride01, stride02, stride10, stride11, stride12, f);
        } else {
          launch_elementwise_kernel_2_2_broadcast_in2_dim0_grid1d<nt, 1, res_t, arg0_t, arg1_t>(
              numel, data0, data1, data2, size0, size1,
              stride00, stride01, stride02, stride10, stride11, stride12, f);
        }
        return true;
      }
    }
    if (vt == 8) {
      launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d<Policy<8, res_t, arg0_t, arg1_t, func_t>, nt, 8, res_t, arg0_t, arg1_t>(
          numel, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (vt == 4) {
      launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d<Policy<4, res_t, arg0_t, arg1_t, func_t>, nt, 4, res_t, arg0_t, arg1_t>(
          numel, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (vt == 2) {
      launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d<Policy<2, res_t, arg0_t, arg1_t, func_t>, nt, 2, res_t, arg0_t, arg1_t>(
          numel, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else {
      launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d<Policy<1, res_t, arg0_t, arg1_t, func_t>, nt, 1, res_t, arg0_t, arg1_t>(
          numel, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    }
    return true;
  }
  return false;
}
template <typename In2Read, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static inline bool launch_elementwise_kernel_2_2_contiguous_in1_dim0_vt_scan(
    TensorIteratorBase& iter, int64_t numel,
    char* data0, char* data1, char* data2,
    int64_t size0, int64_t size1,
    int64_t stride00, int64_t stride01, int64_t stride02,
    int64_t stride10, int64_t stride11, int64_t stride12,
    const func_t& f) {
  constexpr int nt = 128;
  // A float32 grid with fewer blocks than half the device's SMs is
  // latency-bound: halving the vector width doubles independent blocks
  // while retaining vector accesses for the whole layout family.
  const int vt_default = vector_access_width<res_t>();
  const int64_t blocks =
      (numel + static_cast<int64_t>(nt) * vt_default - 1) /
      (static_cast<int64_t>(nt) * vt_default);
  const int vt_max =
      (sizeof(res_t) == 4 && blocks < gpu_sm_count() / 2) ? vt_default / 2 : vt_default;
  for (int vt = vt_max; vt >= 1; vt /= 2) {
    if (size0 % vt != 0) {
      continue;
    }
    const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
    if (stride10 % al != 0 || stride11 % al != 0) {
      continue;
    }
    if (reinterpret_cast<uintptr_t>(data0) % al != 0 ||
        reinterpret_cast<uintptr_t>(data1) % al != 0) {
      continue;
    }
    if constexpr (In2Read::vec_read) {
      if (stride12 % al != 0) {
        continue;
      }
      if (reinterpret_cast<uintptr_t>(data2) % al != 0) {
        continue;
      }
    } else {
      if (reinterpret_cast<uintptr_t>(data2) % static_cast<int64_t>(sizeof(arg1_t)) != 0) {
        continue;
      }
    }
    log_elementwise_info(iter, "p_e_gpu_2_2_1dc", f);
    if (vt == 8) {
      launch_elementwise_kernel_2_2_contiguous_in1_dim0<In2Read, nt, 8, res_t, arg0_t, arg1_t>(
          numel, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (vt == 4) {
      launch_elementwise_kernel_2_2_contiguous_in1_dim0<In2Read, nt, 4, res_t, arg0_t, arg1_t>(
          numel, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (vt == 2) {
      launch_elementwise_kernel_2_2_contiguous_in1_dim0<In2Read, nt, 2, res_t, arg0_t, arg1_t>(
          numel, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else {
      launch_elementwise_kernel_2_2_contiguous_in1_dim0<In2Read, nt, 1, res_t, arg0_t, arg1_t>(
          numel, data0, data1, data2, size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    }
    return true;
  }
  return false;
}
template <typename func_t, typename array_t>
static inline bool try_launch_gpu_2_2(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;
  using arg1_t = typename traits::template arg<1>::type;

  const int64_t numel = iter.numel();
  if (numel <= 0) {
    return false;
  }
  const int64_t size0 = iter.shape()[0];
  const int64_t size1 = iter.shape()[1];
  const int64_t stride00 = iter.strides(0)[0];
  const int64_t stride01 = iter.strides(1)[0];
  const int64_t stride02 = iter.strides(2)[0];
  const int64_t stride10 = iter.strides(0)[1];
  const int64_t stride11 = iter.strides(1)[1];
  const int64_t stride12 = iter.strides(2)[1];

  // The whole-row specialization is removed: it regressed every measured
  // operation at 4K, and no sub-4K benefit was established. Tiny workloads
  // therefore fall through to the legacy implementation.
  if (numel <= 4096) {
    return false;
  }


  // forms share one guard and one helper; the strip Policy picks the read
  // shape (per-row scalar read vs hoisted vector read + cross-row reuse).

  // satisfy d0c's guard (which does not check dim1 strides), so d0c would
  // capture them first and the row reuse would be lost. dtype gating keeps
  // the per-form differences: the widest combination (d0a1/d1a1 allow
  // arg1 == Bool) wraps the guard, d1a0 (same-type only) is tightened with
  // a nested if constexpr.
  if constexpr (((sizeof(res_t) == 2 && sizeof(arg0_t) == 2 &&
                  (sizeof(arg1_t) == 2 || sizeof(arg1_t) == 1))) ||
                ((sizeof(res_t) == 4 && sizeof(arg0_t) == 4 &&
                  (sizeof(arg1_t) == 4 || sizeof(arg1_t) == 1)))) {
    const int64_t es_res = static_cast<int64_t>(sizeof(res_t));
    const int64_t es_arg0 = static_cast<int64_t>(sizeof(arg0_t));
    const int64_t es_arg1 = static_cast<int64_t>(sizeof(arg1_t));
    constexpr int64_t y_t = 8;
    // numel >= 131072 floors the d0a1 form. For size0 = 128 rows the
    // 256-thread strip keeps grid_x == 1 and only numel/32 threads stay
    // active (87.5% idle); measured div/add [128,512] (65536) 1.06~1.2x
    // slower than legacy while mul [128,48128] (6.16M) is 2.0x faster.
    // Fall-through lands in the 1dc family (reorder leg) with all lanes
    // active, ~ legacy parity.
    // size0 >= 128 floors the d0a1 form a second time. For size0 = 3
    // (add [3,100352] Float, numel 301056) the vt scan lands on
    // vt == 1 (3 % {8,4,2} != 0), grid_x == 1 and only 3/256 = 1.2% of
    // threads stay active; every block performs 3 * 8 = 24 scalar ops, and
    // the 12544-block launch cost dominates the 301056-element work
    // (measured 2~3x slower than legacy although numel > 131072). The
    // active-thread ratio is fixed by size0/vt: size0 >= 128 guarantees
    // >= 16 active lanes (vt == 8) / >= 1024 useful ops per block, i.e.
    // the measured win domain (mul [128,N], 3.08M 1.75x / 6.16M 2.0x).
    // Fall-through lands in the 1dc reorder leg (128 lanes all active,
    // ~ legacy parity), same as the numel-floor path above.
    const bool d0a1 = (numel >= 131072 && size0 >= 128 &&
                       stride00 == es_res && stride01 == es_arg0 &&
                       stride10 == es_res * size0 && stride02 == 0 && !iter.is_cpu_scalar(2));
    // The d1 forms have no 1D fallback and the 2D grid would overflow
    // gridDim.y for size1 > 65535 * y_t. Oversized rows fall through to the
    // d0c family's 1D kernel (Bool d1a1 to legacy).
    const bool d1_ok = size1 <= gpu_grid_cap(1) * y_t;
    const bool d1a1 = (stride00 == es_res && stride01 == es_arg0 && stride02 == es_arg1 &&
                       stride10 == es_res * size0 && stride12 == 0 && !iter.is_cpu_scalar(2) && d1_ok);
    bool d1a0 = false;
    if constexpr ((sizeof(res_t) == 2 && sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2) ||
                  (sizeof(res_t) == 4 && sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4)) {
      d1a0 = (stride00 == es_res && stride01 == es_arg0 && stride02 == es_arg1 &&
              stride10 == es_res * size0 && stride11 == 0 && !iter.is_cpu_scalar(1) && d1_ok);
    }
    // arg0 dim0-broadcast (a0d0) -- IN1 per-row scalar read (stride01==0),
    // IN2 dim0-contiguous vector read (stride02==es_arg1). Same-width only; the
    // grid2d skeleton has no 1D fallback, so rows must fit gridDim.y (d1_ok).
    // Mutually exclusive with d0a1 (stride02==0) and d1a1/d1a0 (stride01==es_arg0).
    // size0 >= 128 mirrors the d0a1 floor. The d0a0 strip uses the same
    // nt = 256 skeleton: add [3,100352] in1-dim0-broadcast form
    // keeps 3/256 = 1.2% of threads active and measures ~2x slower than
    // legacy. Unlike d0a1 there is no 1dc fall-through (stride01 == 0 fails
    // the 1dc guard), so the blocked form lands in legacy, which is faster
    // for this shape. The proven d0a0 win domain ([2048,2048]) has
    // size0 = 2048 >= 128 and is unaffected.
    bool d0a0 = false;
    if constexpr ((sizeof(res_t) == 2 && sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2) ||
                  (sizeof(res_t) == 4 && sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4)) {
      d0a0 = (stride00 == es_res && stride01 == 0 && !iter.is_cpu_scalar(1) &&
              stride02 == es_arg1 && stride10 == es_res * size0 && size0 >= 128 && d1_ok);
    }
    // The dim1-broadcast strip keeps only size0/vt of its 256 lanes
    // active. Below size0 = 128 (fewer than 16 active lanes at vt = 8) and
    // up to 1M elements both operand orders measured slower than legacy;
    // route these low-occupancy workloads to legacy and keep the later
    // 1D/reorder families available for other layouts.
    const bool p2_all_slow = size0 < 128 && numel <= 1048576;
    if ((d1a1 || d1a0) && p2_all_slow) {
      return false;
    }
    if (d0a1 || d1a1 || d1a0 || d0a0) {
      if (d0a1) {
        return launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d_vt_scan<strip_d0a1, res_t, arg0_t, arg1_t>(
            iter, numel, data[0], data[1], data[2], size0, size1,
            stride00, stride01, stride02, stride10, stride11, stride12, f);
      }
      if (d1a1) {
        return launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d_vt_scan<strip_d1a1, res_t, arg0_t, arg1_t>(
            iter, numel, data[0], data[1], data[2], size0, size1,
            stride00, stride01, stride02, stride10, stride11, stride12, f);
      }
      if (d0a0) {
        return launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d_vt_scan<strip_d0a0, res_t, arg0_t, arg1_t>(
            iter, numel, data[0], data[1], data[2], size0, size1,
            stride00, stride01, stride02, stride10, stride11, stride12, f);
      }
      return launch_elementwise_kernel_2_2_broadcast_any1dim_grid2d_vt_scan<strip_d1a0, res_t, arg0_t, arg1_t>(
          iter, numel, data[0], data[1], data[2], size0, size1,
          stride00, stride01, stride02, stride10, stride11, stride12, f);
    }
  }


  if constexpr (((sizeof(res_t) == 2 && sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2)) ||
                ((sizeof(res_t) == 4 && sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4)) ||
                ((sizeof(res_t) == 8 && sizeof(arg0_t) == 8 && sizeof(arg1_t) == 8))) {
    if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride01 == 0 && !iter.is_cpu_scalar(1) &&
        stride11 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride02 == static_cast<int64_t>(sizeof(arg1_t)) &&
        stride12 == 0 && !iter.is_cpu_scalar(2)) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % al != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_2_2_x", f);
        constexpr int nt = 128;
        if (vt == 8) {
          launch_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1<nt, 8, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1<nt, 4, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1<nt, 2, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
        } else {
          launch_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1<nt, 1, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
        }
        return true;
      }
    }
  }

  // Cross broadcast mirror -- arg0 dim1-broadcast (stride11 == 0) with
  // dim0-contiguous vector read hoisted out of the row loop, arg1
  // dim0-broadcast (stride02 == 0) scalar-read per row. 8-byte width only
  // (Long): the 2/4-byte forms are subsets of the 1db d0a1 guard above and
  // would be unreachable here. Mutually exclusive with 2_2_x above
  // (stride01==0 && stride11==es && stride02==es && stride12==0 there).
  // tag p_e_gpu_2_2_xl.
  if constexpr (sizeof(res_t) == 8 && sizeof(arg0_t) == 8 && sizeof(arg1_t) == 8) {
    if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t)) && !iter.is_cpu_scalar(1) &&
        stride11 == 0 &&
        stride02 == 0 && !iter.is_cpu_scalar(2) &&
        stride12 == static_cast<int64_t>(sizeof(arg1_t)) &&
        size1 <= gpu_grid_cap(1) * 8) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_2_2_xl", f);
        constexpr int nt = 128;
        if (vt == 2) {
          launch_elementwise_kernel_2_2_crossbroadcast_in1_dim1_in2_dim0<nt, 2, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
        } else {
          launch_elementwise_kernel_2_2_crossbroadcast_in1_dim1_in2_dim0<nt, 1, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
        }
        return true;
      }
    }
  }


  // one helper; the In2Read lane reader picks the IN2 read shape (vector
  // load vs per-lane scalar load). Kept after the 1db guard: the d0c leg
  // (stride02 == es) is mutually exclusive with 1db's stride02 == 0, so the
  // ordering is organizational only. The blow-up (b) overflow path of the
  // 1db d1 forms lands here: their layout also satisfies the d0c leg's
  // guard, and the 1D grid has no gridDim.y ceiling.
  if constexpr (((sizeof(res_t) == 2 && sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2)) ||
                ((sizeof(res_t) == 4 && sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4))) {
    if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
        !iter.is_cpu_scalar(1) && !iter.is_cpu_scalar(2)) {
      if (stride02 == static_cast<int64_t>(sizeof(arg1_t))) {
        // d0c leg: IN2 contiguous on dim0, vector read.
        if (launch_elementwise_kernel_2_2_contiguous_in1_dim0_vt_scan<in2_vec_read, res_t, arg0_t, arg1_t>(
                iter, numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f)) {
          return true;
        }
      }
      if (stride12 == static_cast<int64_t>(sizeof(arg1_t)) &&
          stride02 % static_cast<int64_t>(sizeof(arg1_t)) == 0) {
        // d0c-reorder leg: IN2 contiguous on dim1, per-lane scalar read.
        if (launch_elementwise_kernel_2_2_contiguous_in1_dim0_vt_scan<in2_lane_scalar_read, res_t, arg0_t, arg1_t>(
                iter, numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f)) {
          return true;
        }
      }
    }
  }

  return false;
}
