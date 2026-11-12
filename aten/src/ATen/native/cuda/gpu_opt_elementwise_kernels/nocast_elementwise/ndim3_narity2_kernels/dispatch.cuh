// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim3_narity2_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_3_2
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/common.cuh"
#include "../../utils/log.h"
#include "broadcast_in1_dim0_in2_dim2_grid1d.cuh"
#include "broadcast_in1_dim0_in2_dim2_grid3d.cuh"
#include "broadcast_in2_dim02_grid1d.cuh"
#include "broadcast_in2_dim02_grid3d.cuh"
#include "broadcast_in2_dim0.cuh"
#include "broadcast_in2_dim1.cuh"
#include "crossbroadcast_in1_dim1_in2_dim0.cuh"
#include "contiguous_alldim0_grid1d.cuh"
#include "contiguous_alldim0_grid2d.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_3_2(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;
  using arg1_t = typename traits::template arg<1>::type;

  if constexpr (!(sizeof(res_t) == sizeof(arg0_t) &&
                  (sizeof(res_t) == 2 || sizeof(res_t) == 4 ||
                   sizeof(res_t) == 8 || sizeof(res_t) == 16))) {
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
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride02 = iter.strides(2)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride11 = iter.strides(1)[1];
    const int64_t stride12 = iter.strides(2)[1];
    const int64_t stride20 = iter.strides(0)[2];
    const int64_t stride21 = iter.strides(1)[2];
    const int64_t stride22 = iter.strides(2)[2];


    // (stride02 == sizeof(arg1_t)); OUT and IN1 fully contiguous. IN2 is read
    // once per thread and reused across all size1 rows, while IN1/OUT are
    // touched per row; grid.z walks dim2.
    // arg1 may be any element size (Bool masks included): it is vector-read
    // when its size matches es and read element-wise otherwise, so only
    // natural alignment is required in that case (see al2 below).

    // dim1-broadcast IN2 would otherwise be captured by d0c first (it does
    // not test stride12) and lose the broadcast-row reuse.
    // 16B elements force vt == 1 (z_chunk/ub structure gives no win);
    // let them fall through to legacy instead.
    if (sizeof(res_t) != 16 &&
        stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride11 == static_cast<int64_t>(sizeof(arg0_t)) * size0 &&
        stride21 == static_cast<int64_t>(sizeof(arg0_t)) * size0 * size1 &&
        stride02 == static_cast<int64_t>(sizeof(arg1_t)) &&
        stride12 == 0 && !iter.is_cpu_scalar(2) &&
        size2 <= gpu_grid_cap(2) * (sizeof(res_t) == 16 ? 64 : 8) &&
        size1 <= gpu_grid_cap(1) * 8) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (stride22 % al != 0) {
          continue;
        }
        const int64_t al2 = static_cast<int64_t>(sizeof(arg1_t)) *
            ((sizeof(arg1_t) == sizeof(res_t)) ? vt : 1);
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % al2 != 0) {
          continue;
        }
        // The launcher maps x vectors and y rows directly to the block.
        // For small workloads, lower vt until that block has at least one
        // warp's worth of active lanes instead of launching an underfilled
        // vt=8 block (for example size0=16).
        const int64_t x_lanes = size0 / vt < 128 ? size0 / vt : 128;
        const int64_t y_lanes = size1 < 8 ? size1 : 8;
        if (numel <= 262144 && x_lanes * y_lanes < 32) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_3_2_bd1", f);


        // two rounds of four z segments (z_i = threadIdx.y * 4 + {0, 32} < 64)
        // and all 8 y-lanes stay active; each lane serializes 8 z rounds,


        constexpr int z_t = (sizeof(res_t) == 16) ? 64 : 8;
        if (vt == 8) {
          launch_elementwise_kernel_3_2_broadcast_in2_dim1<8, z_t, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_3_2_broadcast_in2_dim1<4, z_t, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_3_2_broadcast_in2_dim1<2, z_t, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else {
          launch_elementwise_kernel_3_2_broadcast_in2_dim1<1, z_t, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        }
        return true;
      }
    }


    // size with res (vector kernels assume one vector width for all operands).
    // 16B elements force vt == 1 here as well; dim1-broadcast 16B
    // layouts excluded from bd1 above would otherwise land in d0c.
    if (sizeof(res_t) != 16 &&
        sizeof(arg1_t) == sizeof(res_t) &&
        stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride02 == static_cast<int64_t>(sizeof(arg1_t))) {
      const int vt_max = vector_access_width<res_t>();

      // back to the 1D-grid kernel, which has no gridDim.y/z limit.
      // Keep the gridDim.z block count at the historical 65535 cap: beyond
      // that the 2D grid's per-block scheduling overhead dominates (measured
      // 3.4-3.6x slower, bf16 3_2_d0c 64M). Never exceed the device gridDim.z
      // limit.
      if (size2 > gpu_grid_cap(2)) {
        for (int vt = vt_max; vt >= 1; vt /= 2) {
          if (size0 % vt != 0) {
            continue;
          }
          const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
          if (stride11 % al != 0 || stride21 % al != 0 ||
              stride12 % al != 0 || stride22 % al != 0) {
            continue;
          }
          if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
              reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
              reinterpret_cast<uintptr_t>(data[2]) % al != 0) {
            continue;
          }
          log_elementwise_info(iter, "p_e_gpu_3_2_d0c_1d", f);
          constexpr int nt = 128;
          if (vt == 8) {
            launch_elementwise_kernel_3_2_contiguous_alldim0_grid1d<nt, 8, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else if (vt == 4) {
            launch_elementwise_kernel_3_2_contiguous_alldim0_grid1d<nt, 4, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else if (vt == 2) {
            launch_elementwise_kernel_3_2_contiguous_alldim0_grid1d<nt, 2, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_contiguous_alldim0_grid1d<nt, 1, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
          return true;
        }
        return false;
      }
      // 2D grid dimension limit; keep the historical 65535 cap on the y
      // block count for the same per-block scheduling reason above. Never
      // exceed the device gridDim.y limit.
      if (size1 > gpu_grid_cap(1) * 8) {
        return false;
      }
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (stride11 % al != 0 || stride21 % al != 0 ||
            stride12 % al != 0 || stride22 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % al != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_3_2_d0c_2d", f);

        if (vt == 8) {
          launch_elementwise_kernel_3_2_contiguous_alldim0_grid2d<8, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_3_2_contiguous_alldim0_grid2d<4, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_3_2_contiguous_alldim0_grid2d<2, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else {
          launch_elementwise_kernel_3_2_contiguous_alldim0_grid2d<1, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        }
        return true;
      }
    }


    // IN2 dim2-broadcast (same vector reused across dim2 rows).
    if (sizeof(arg1_t) == sizeof(res_t) &&
        stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
        stride01 == 0 && !iter.is_cpu_scalar(1) &&
        stride02 == static_cast<int64_t>(sizeof(arg1_t)) &&
        stride22 == 0 && !iter.is_cpu_scalar(2)) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (stride11 % al != 0 || stride21 % al != 0 ||
            stride12 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % al != 0) {
          continue;
        }
        // 3D-grid division-free variant: gridDim.y/z must stay within the
        // 65535 hardware limit, otherwise fall back to the 1D version.
        // C1: dim1 rows are chunked across blockDim.y lanes and batched
        // across gridDim.y in the kernel, so size1 is unbounded; only dim2
        // keeps the 65535 * z_t limit. C1b: nt_x = min(size0 / vt, 128)
        // inside the launcher (small size0 keeps a full warp via y_b).
        constexpr int nt = 128;  // 1D fallback block width
        constexpr int z_t = 8;
        const bool use_3d = size2 <= gpu_grid_cap(2) * z_t;
        log_elementwise_info(iter, use_3d ? "p_e_gpu_3_2_bd0_3d" : "p_e_gpu_3_2_bd0_1d", f);
        if (vt == 8) {
          if (use_3d) {
            launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid3d<8, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid1d<nt, 8, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
        } else if (vt == 4) {
          if (use_3d) {
            launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid3d<4, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid1d<nt, 4, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
        } else if (vt == 2) {
          if (use_3d) {
            launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid3d<2, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid1d<nt, 2, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
        } else {
          if (use_3d) {
            launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid3d<1, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_broadcast_in1_dim0_in2_dim2_grid1d<nt, 1, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
        }
        return true;
      }
    }



    // swapped; covers layouts like (8160,21,48) with a (21,) tensor
    // broadcast over dim0/dim2.
    if (sizeof(arg1_t) == sizeof(res_t) &&
        stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride02 == 0 && !iter.is_cpu_scalar(2) &&
        stride22 == 0) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (stride11 % al != 0 || stride21 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % static_cast<uintptr_t>(sizeof(arg1_t)) != 0) {
          continue;
        }
        constexpr int nt = 128;
        // 3D-grid division-free variant: gridDim.y/z must stay within the
        // historical 65535 cap, otherwise fall back to the 1D version.
        constexpr int z_t = 8;
        const int64_t y_cap = gpu_grid_cap(1);
        const int64_t z_cap = gpu_grid_cap(2);
        const bool use_3d = size1 <= y_cap && size2 <= z_cap * z_t;
        log_elementwise_info(iter, use_3d ? "p_e_gpu_3_2_a1d0_3d" : "p_e_gpu_3_2_a1d0_1d", f);
        if (vt == 8) {
          if (use_3d) {
            launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid3d<nt, 8, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid1d<nt, 8, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
        } else if (vt == 4) {
          if (use_3d) {
            launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid3d<nt, 4, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid1d<nt, 4, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
        } else if (vt == 2) {
          if (use_3d) {
            launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid3d<nt, 2, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid1d<nt, 2, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
        } else {
          if (use_3d) {
            launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid3d<nt, 1, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_broadcast_in2_dim02_grid1d<nt, 1, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
        }
        return true;
      }
    }

    // IN2 dim0-broadcast + dim2-contiguous (stride02==0, stride22==es_arg1);
    // OUT and IN1 fully contiguous. IN2 is invariant along dim0 only, so one
    // scalar read feeds all vt dim0 elements of a (y, z) row; IN1/OUT advance
    // per row. grid.z walks dim2 in z_t chunks (tag p_e_gpu_3_2_d0b).
    // Mutually exclusive with bd1/d0c (stride02==es), bd0 (stride01==0) and
    // bd02 (stride22==0).
    if (sizeof(arg1_t) == sizeof(res_t) &&
        stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride11 == static_cast<int64_t>(sizeof(arg0_t)) * size0 &&
        stride21 == static_cast<int64_t>(sizeof(arg0_t)) * size0 * size1 &&
        stride02 == 0 && !iter.is_cpu_scalar(2) &&
        stride22 == static_cast<int64_t>(sizeof(arg1_t)) &&
        size1 <= gpu_grid_cap(1) * 8 &&
        size2 <= gpu_grid_cap(2) * 8) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (stride10 % al != 0 || stride20 % al != 0 ||
            stride11 % al != 0 || stride21 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % static_cast<uintptr_t>(sizeof(arg1_t)) != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_3_2_d0b", f);
        constexpr int z_t = 8;
        if (vt == 8) {
          launch_elementwise_kernel_3_2_broadcast_in2_dim0<8, z_t, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_3_2_broadcast_in2_dim0<4, z_t, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_3_2_broadcast_in2_dim0<2, z_t, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        } else {
          launch_elementwise_kernel_3_2_broadcast_in2_dim0<1, z_t, res_t, arg0_t, arg1_t>(
              numel, data[0], data[1], data[2], size0, size1, size2,
              stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        }
        return true;
      }
    }

    // Cross-broadcast IN1 dim1 (stride11==0) + IN2 dim0 (stride02==0);
    // OUT fully contiguous. IN1's vt load hoists out of the dim1 row loop,
    // IN2 is one scalar per row. Same-width 2/4-byte operands only
    // (tag p_e_gpu_3_2_xb). Mutually exclusive with bd1/d0c/bd02
    // (stride11==es*size0), bd0 (stride01==0) and the IN2 dim0-broadcast form (stride11==es*size0).
    if constexpr ((sizeof(res_t) == 2 && sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2) ||
                  (sizeof(res_t) == 4 && sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4)) {
      if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
          stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
          stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
          stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
          stride11 == 0 && !iter.is_cpu_scalar(1) &&
          stride02 == 0 && !iter.is_cpu_scalar(2) &&
          size1 <= gpu_grid_cap(1) * 8 &&
          size2 <= gpu_grid_cap(2) * 8) {
        // A vt=8 row vector shorter than a warp (size0 / vt < 32 lanes)
        // leaves most of each block idle and serializes the functor chain.
        // For 2-byte tiles halving vt doubles active lanes without reducing
        // the block count; wider rows keep the full vector width.
        const int vt_default = vector_access_width<res_t>();
        const int vt_max =
            (sizeof(res_t) == 2 && size0 / vt_default < 32) ? vt_default / 2 : vt_default;
        for (int vt = vt_max; vt >= 1; vt /= 2) {
          if (size0 % vt != 0) {
            continue;
          }
          const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
          if (stride10 % al != 0 || stride20 % al != 0 ||
              stride21 % al != 0) {
            continue;
          }
          if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
              reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
              reinterpret_cast<uintptr_t>(data[2]) % static_cast<uintptr_t>(sizeof(arg1_t)) != 0) {
            continue;
          }
          log_elementwise_info(iter, "p_e_gpu_3_2_xb", f);
          constexpr int z_t = 8;
          if (vt == 8) {
            launch_elementwise_kernel_3_2_crossbroadcast_in1_dim1_in2_dim0<8, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else if (vt == 4) {
            launch_elementwise_kernel_3_2_crossbroadcast_in1_dim1_in2_dim0<4, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else if (vt == 2) {
            launch_elementwise_kernel_3_2_crossbroadcast_in1_dim1_in2_dim0<2, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          } else {
            launch_elementwise_kernel_3_2_crossbroadcast_in1_dim1_in2_dim0<1, z_t, res_t, arg0_t, arg1_t>(
                numel, data[0], data[1], data[2], size0, size1, size2,
                stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
          }
          return true;
        }
      }
    }

    return false;
  }
}
