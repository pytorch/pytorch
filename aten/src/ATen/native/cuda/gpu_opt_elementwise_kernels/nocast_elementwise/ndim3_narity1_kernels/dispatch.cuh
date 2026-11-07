// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim3_narity1_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_3_1
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/common.cuh"
#include "../../utils/log.h"
#include "broadcast_dim1.cuh"
#include "contiguous_alldim0_grid1d.cuh"
#include "contiguous_alldim0_grid2d.cuh"
#include "transpose.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_3_1(TensorIteratorBase& iter, const func_t& f, array_t data) {
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
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride11 = iter.strides(1)[1];
    const int64_t stride20 = iter.strides(0)[2];
    const int64_t stride21 = iter.strides(1)[2];


    // branches below carry their own full-contiguity checks, and the d0c
    // kernel computes both offsets from strides directly, so OUT row gaps
    // (sliced views of larger tensors) are safe for it.
    if (stride00 != static_cast<int64_t>(sizeof(res_t))) {
      return false;
    }


    // fully contiguous (dim0-fastest); IN is dim1-contiguous and its
    // (size1,size0) plane is the transpose view of OUT's (size0,size1) plane
    // (stride01 == es*size1), dim2 is an aligned batch axis. Tile kernel:
    // bf16 tile 64x64, float tile 32x32. tag p_e_gpu_3_1_tp.
    // NOTE: must be tested before the d0c/bd1 precondition stride01 == es;
    // stride01 == es*size1 with size1 >= 128 is mutually exclusive with it.
    if (stride01 != static_cast<int64_t>(sizeof(arg0_t))) {
      // Guard: below 256K the tile grid collapses to a few dozen blocks
      // and the smem staging + barrier cost is not amortized (256K measured
      // 0.822x vs legacy, 5/5 ops); 1M+ keeps the tile path and its speedup.
      // size2 rides gridDim.z directly, so bound it by the device's z limit
      // (gpu_max_grid_size) and, tighter in practice, the measured sweet cap.
      if (stride11 == static_cast<int64_t>(sizeof(arg0_t)) &&
          stride01 == static_cast<int64_t>(sizeof(arg0_t)) * size1 &&
          stride21 == static_cast<int64_t>(sizeof(arg0_t)) * size0 * size1 &&
          stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
          stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
          numel >= 262144 && size0 >= 128 && size1 >= 128 &&
          size2 <= std::min<int64_t>(gpu_max_grid_size(2), GPU_GRID_SWEET_CAP)) {
        log_elementwise_info(iter, "p_e_gpu_3_1_tp", f);
        if (sizeof(res_t) == 2 && numel > 1048576) {
          launch_elementwise_kernel_3_1_transpose<64, 8, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2,
              stride00, stride01, stride10, stride11, stride20, stride21, f);
        } else {
          launch_elementwise_kernel_3_1_transpose<32, 8, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2,
              stride00, stride01, stride10, stride11, stride20, stride21, f);
        }
        return true;
      }
      return false;
    }

    // Per thread, one vector load on dim0 feeds all size1 rows (per dim2
    // slice); grid.z walks dim2 in z_t chunks. Tag p_e_gpu_3_1_bd1.
    // Must be tested before the d0c loop below: this layout also satisfies
    // d0c (stride11 == 0 trivially passes its stride11 % al test), so d0c
    // would capture it first and the row reuse would be lost. Layouts that
    // miss here - including size2 beyond the gridDim.z limit - fall through
    // to d0c unchanged.
    if (stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride11 == 0 && !iter.is_cpu_scalar(1) &&
        // dim2 rides gridDim.z in z_t = 8 chunks: the device z limit scales
        // by 8, and the sweet cap (the tighter bound here) likewise.
        size2 <= std::min<int64_t>(gpu_max_grid_size(2), GPU_GRID_SWEET_CAP) * 8) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (stride21 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_3_1_bd1", f);
        constexpr int nt = 128;
        constexpr int z_t = 8;
        if (vt == 8 && size0 == 8) {
          launch_elementwise_kernel_3_1_broadcast_dim1_narrow_x<8, res_t, arg0_t>(
              data[0], data[1], size1, size2, stride00, stride01, stride10, stride20, stride21, f);
        } else if (vt == 8) {
          launch_elementwise_kernel_3_1_broadcast_dim1<nt, 8, z_t, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_3_1_broadcast_dim1<nt, 4, z_t, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_3_1_broadcast_dim1<nt, 2, z_t, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
        } else {
          launch_elementwise_kernel_3_1_broadcast_dim1<nt, 1, z_t, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
        }
        return true;
      }
    }


    // dim0-contiguous layouts with an arbitrary dim2 batch: the 2D-grid
    // kernel walks dim2 on gridDim.z (z_t >= 1 chunks, adaptive in the
    // launcher) and dim1 on gridDim.y in y_b >= 8 steps, so the launch must
    // satisfy size2 <= maxGridSize.z and size1 <= maxGridSize.y * 8. The
    // measured sweet cap (see env.cuh) is the tighter bound on both in
    // practice. The old kernel walked size2 on gridDim.y, which overflowed
    // for size2 > 65535. Out-of-range layouts fall through to legacy
    // unchanged.
    if (size1 <= std::min<int64_t>(gpu_max_grid_size(1), GPU_GRID_SWEET_CAP) * 8 &&
        size2 <= std::min<int64_t>(gpu_max_grid_size(2), GPU_GRID_SWEET_CAP)) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(arg0_t)) * vt;
        // OUT rows are written with vt-wide vector stores at idx1 * stride10
        // + idx2 * stride20, so the OUT higher-dim strides must stay vector
        // aligned too (the entry precondition only pins stride00).
        if (stride10 % al != 0 || stride20 % al != 0 ||
            stride11 % al != 0 || stride21 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0) {
          continue;
        }

        // leave more than a quarter of its x-threads idle in the tail block.
        // The v1 <=half-block rule misfired on shapes like 6240x13x4
        // (x_rem = 24 of nt_x = 128 spread over 13 blocks: 6% idle), where
        // the 1D kernel's 64-bit div/mod costs far more than the few wasted
        // threads. The y axis never wastes enough to matter (y_b <= 8 and
        // grid_y grows with size1), so it is left to the 2D kernel.
        const int64_t x_vec = size0 / vt;
        const int64_t nt_x = std::min<int64_t>(x_vec, 128);
        const int64_t grid_x = (x_vec + nt_x - 1) / nt_x;
        const int64_t x_rem = x_vec % nt_x;
        const bool x_waste = x_rem != 0 && (nt_x - x_rem) * 4 > grid_x * nt_x;
        if (x_waste) {
          log_elementwise_info(iter, "p_e_gpu_3_1_d0c_1d", f);
          constexpr int nt = 128;
          if (vt == 8) {
            launch_elementwise_kernel_3_1_contiguous_alldim0_grid1d<nt, 8, res_t, arg0_t>(
                numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
          } else if (vt == 4) {
            launch_elementwise_kernel_3_1_contiguous_alldim0_grid1d<nt, 4, res_t, arg0_t>(
                numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
          } else if (vt == 2) {
            launch_elementwise_kernel_3_1_contiguous_alldim0_grid1d<nt, 2, res_t, arg0_t>(
                numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
          } else {
            launch_elementwise_kernel_3_1_contiguous_alldim0_grid1d<nt, 1, res_t, arg0_t>(
                numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
          }
          return true;
        }
        log_elementwise_info(iter, "p_e_gpu_3_1_d0c_2d", f);

        if (vt == 8) {
          launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d<8, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d<4, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d<2, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
        } else {
          launch_elementwise_kernel_3_1_contiguous_alldim0_grid2d<1, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
        }
        return true;
      }
    }

    return false;
  }
}
