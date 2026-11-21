// File: gpu_opt_elementwise_kernels/nocast_elementwise/ndim5_narity1_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_n1_slice_shmem, try_launch_gpu_5_1
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/common.cuh"
#include "../../utils/log.h"
#include "contiguous_alldim0.cuh"
#include "contiguous_out.cuh"
#include "interleave_dim03.cuh"
#include "permuted_contiguous_block.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_n1_slice_shmem(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  using arg0_t = typename traits::template arg<0>::type;
  if constexpr (!(std::is_same_v<res_t, arg0_t> &&
                  (sizeof(res_t) == 2 || sizeof(res_t) == 4))) {
    return false;
  } else {
    const int64_t numel = iter.numel();
    if (numel <= 0 || iter.ndim() != 5) {
      return false;
    }
    const int64_t es = static_cast<int64_t>(sizeof(res_t));
    const int64_t size0 = iter.shape()[0];
    const int64_t size1 = iter.shape()[1];
    const int64_t size2 = iter.shape()[2];
    const int64_t size3 = iter.shape()[3];
    const int64_t size4 = iter.shape()[4];
    // OUT fully contiguous.
    const int64_t s00 = iter.strides(0)[0], s10 = iter.strides(0)[1],
                  s20 = iter.strides(0)[2], s30 = iter.strides(0)[3], s40 = iter.strides(0)[4];
    if (s00 != es || s10 != es * size0 || s20 != es * size0 * size1 ||
        s30 != es * size0 * size1 * size2 || s40 != es * size0 * size1 * size2 * size3) {
      return false;
    }
    // IN permuted-contiguous chain (4,0,2,1,3).
    const int64_t s01 = iter.strides(1)[0], s11 = iter.strides(1)[1],
                  s21 = iter.strides(1)[2], s31 = iter.strides(1)[3], s41 = iter.strides(1)[4];
    if (s41 != es || s01 != es * size4 || s21 != es * size4 * size0 ||
        s11 != es * size4 * size0 * size2 || s31 != es * size4 * size0 * size2 * size1) {
      return false;
    }
    const int64_t slice_bytes = es * size4 * size0 * size1 * size2;
    // 32KB is the dynamic-smem budget (chunk * slice_bytes); 16KB is the
    // measured threshold below which staging and synchronization regress below
    // legacy. In this accepted interval the quotient is always one.
    if (slice_bytes > 32768 || slice_bytes % 16 != 0 || slice_bytes <= 16384) {
      return false;
    }
    constexpr int64_t chunk = 1;
    const int64_t block_elems = size0 * size1 * size2;
    if (block_elems % 4 != 0) {
      return false;  // 8B write vectors
    }
    if (s30 % 8 != 0 || s40 % 8 != 0) {
      return false;  // 8B alignment of block bases
    }
    if (reinterpret_cast<uintptr_t>(data[0]) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(data[1]) % 16 != 0) {
      return false;
    }
    const int64_t max_grid_x = gpu_max_grid_size(0);
    if (size3 < 128 || max_grid_x <= 0 || size3 > max_grid_x) {
      return false;  // small PCB grids regress; otherwise respect runtime capability
    }
    // Keep shared-memory staging only above the measured per-dtype thresholds.
    if constexpr (sizeof(res_t) == 4) {
      if (numel < 16 * 1024 * 1024) {
        return false;
      }
    } else {
      if (numel < 64 * 1024 * 1024) {
        return false;
      }
    }
    log_elementwise_info(iter, "p_e_gpu_5_1_pcb", f);
    // nt=256 measured faster than nt=128 across the tested unary functors for
    // float32 B=128/512, with shared memory and the grid guard unchanged.
    constexpr int nt = 256;
    launch_elementwise_kernel_5_1_permuted_contiguous_block<nt, res_t, func_t>(
        data[0], data[1], size0, size1, size2, size3, size4,
        s30, s40, s31, slice_bytes, size4 * block_elems, f, chunk);
    return true;
  }
}
template <typename func_t, typename array_t>
static inline bool try_launch_gpu_5_1(TensorIteratorBase& iter, const func_t& f, array_t data) {
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
    const int64_t size4 = iter.shape()[4];
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride11 = iter.strides(1)[1];
    const int64_t stride20 = iter.strides(0)[2];
    const int64_t stride21 = iter.strides(1)[2];
    const int64_t stride30 = iter.strides(0)[3];
    const int64_t stride31 = iter.strides(1)[3];
    const int64_t stride40 = iter.strides(0)[4];
    const int64_t stride41 = iter.strides(1)[4];

    const bool out_contiguous =
        stride00 == static_cast<int64_t>(sizeof(res_t)) &&
        stride10 == static_cast<int64_t>(sizeof(res_t)) * size0 &&
        stride20 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 &&
        stride30 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 * size2 &&
        stride40 == static_cast<int64_t>(sizeof(res_t)) * size0 * size1 * size2 * size3;



    // chain (4,0,2,1,3) layout has in-dim0 stride es*size4, which d0s
    // (any positive es multiple) would otherwise capture first, losing

    // mismatch falls through to the branches below untouched.
    if (try_launch_gpu_n1_slice_shmem(iter, f, data)) {
      return true;
    }

    // The shared-memory PCB path rejects grids with insufficient parallelism.
    // For the same permuted-contiguous layout this direct tile keeps dynamic
    // dims/strides and avoids the generic path's repeated 5D index decode.
    if constexpr (std::is_same_v<res_t, c10::BFloat16> &&
                  std::is_same_v<arg0_t, c10::BFloat16>) {
      constexpr int64_t kVectorWidth = 4;
      constexpr int64_t kDim1Tile = 64;
      constexpr int64_t kDim2Tile = 2;
      const int64_t es = static_cast<int64_t>(sizeof(res_t));
      const bool pcb_permuted_input = out_contiguous &&
          stride41 == es && stride01 == es * size4 &&
          stride21 == es * size4 * size0 &&
          stride11 == es * size4 * size0 * size2 &&
          stride31 == es * size4 * size0 * size2 * size1;
      const int64_t dim2_tiles = (size2 + kDim2Tile - 1) / kDim2Tile;
      const int64_t tile_grid_y = dim2_tiles * size3;
      if (pcb_permuted_input && size0 == kVectorWidth && size1 >= kDim1Tile &&
          size2 > 0 && size2 <= 4 && numel >= 65536 && numel <= 262144 &&
          tile_grid_y > 0 && tile_grid_y <= gpu_max_grid_size(1) &&
          size4 > 0 && size4 <= gpu_max_grid_size(2) &&
          reinterpret_cast<uintptr_t>(data[0]) % (es * kVectorWidth) == 0) {
        log_elementwise_info(iter, "p_e_gpu_5_1_d0s", f);
        launch_elementwise_kernel_5_1_permuted_out_tile<res_t, arg0_t>(
            data[0], data[1], size0, size1, size2, size3, size4,
            stride00, stride01, stride10, stride11, stride20, stride21,
            stride30, stride31, stride40, stride41, f);
        return true;
      }
      // For this exact PCB layout the generic d0s gather path is slower than
      // legacy at 4M/16M. Stop this optimized dispatcher after the direct-tile
      // range so those sizes genuinely fall back instead of being recaptured.
      if (pcb_permuted_input && numel > 262144 &&
          numel < 64 * 1024 * 1024) {
        return false;
      }
    }

    // higher-dim strides (0 for a broadcast dim) must be multiples of the
    // vector width.
    if (out_contiguous &&
        stride01 == static_cast<int64_t>(sizeof(arg0_t))) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(arg0_t)) * vt;
        if (stride11 % al != 0 || stride21 % al != 0 || stride31 % al != 0 || stride41 % al != 0) {
          continue;
        }
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % al != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_5_1_d0c", f);
        constexpr int nt = 128;
        if (vt == 8) {
          launch_elementwise_kernel_5_1_contiguous_alldim0<nt, 8, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, size3, size4,
              stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
        } else if (vt == 4) {
          launch_elementwise_kernel_5_1_contiguous_alldim0<nt, 4, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, size3, size4,
              stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
        } else if (vt == 2) {
          launch_elementwise_kernel_5_1_contiguous_alldim0<nt, 2, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, size3, size4,
              stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
        } else {
          launch_elementwise_kernel_5_1_contiguous_alldim0<nt, 1, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, size3, size4,
              stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
        }
        return true;
      }
    }


    // IN dim3 contiguous (stride31 == sizeof(arg0_t)) and dim0 strides by
    // size3 elements, so IN's (d0, d3) plane is a contiguous vt*s3-element
    // run: one wide vector load. OUT is contiguous: s3 separate vt-wide
    // vector stores along dim0 (dim3 strides by size0 elements). Uses the
    // dedicated in_interleaved kernel - no data pointer swap, so the
    // functor direction stays res = f(arg0).
    if (out_contiguous &&
        stride31 == static_cast<int64_t>(sizeof(arg0_t)) &&
        stride01 == stride31 * size3 &&
        (size3 == 2 || size3 == 4)) {
      constexpr int nt = 128;
      if constexpr (sizeof(res_t) == sizeof(arg0_t)) {
        const int s3 = static_cast<int>(size3);
        const int vt_max = std::min(16 / (static_cast<int>(sizeof(res_t)) * s3), 8);
        for (int vt = vt_max; vt >= 1; vt /= 2) {
          if (size0 % vt != 0) {
            continue;
          }
          const int64_t al_in = static_cast<int64_t>(sizeof(arg0_t)) * vt * s3;
          const int64_t al_out = static_cast<int64_t>(sizeof(res_t)) * vt;
          // IN's wide vector load needs its higher-dim base offsets
          // al_in-aligned; OUT's vt-wide stores need each dim3 row
          // (k*stride30) and the higher-dim base al_out-aligned.
          if (stride11 % al_in != 0 || stride21 % al_in != 0 ||
              stride41 % al_in != 0) {
            continue;
          }
          if (stride10 % al_out != 0 || stride20 % al_out != 0 ||
              stride30 % al_out != 0 || stride40 % al_out != 0) {
            continue;
          }
          if (reinterpret_cast<uintptr_t>(data[0]) % al_out != 0 ||
              reinterpret_cast<uintptr_t>(data[1]) % al_in != 0) {
            continue;
          }
          log_elementwise_info(iter, "p_e_gpu_5_1_ilv_in", f);
          if (size3 == 2) {
            if (vt == 4) {
              launch_elementwise_kernel_5_1_interleave_dim03<nt, 4, 2, res_t, arg0_t>(
                  numel, data[0], data[1], size0, size1, size2, size3, size4,
                  stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
            } else if (vt == 2) {
              launch_elementwise_kernel_5_1_interleave_dim03<nt, 2, 2, res_t, arg0_t>(
                  numel, data[0], data[1], size0, size1, size2, size3, size4,
                  stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
            } else {
              launch_elementwise_kernel_5_1_interleave_dim03<nt, 1, 2, res_t, arg0_t>(
                  numel, data[0], data[1], size0, size1, size2, size3, size4,
                  stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
            }
            return true;
          }
          // size3 == 4
          if (vt == 2) {
            launch_elementwise_kernel_5_1_interleave_dim03<nt, 2, 4, res_t, arg0_t>(
                numel, data[0], data[1], size0, size1, size2, size3, size4,
                stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
          } else {
            launch_elementwise_kernel_5_1_interleave_dim03<nt, 1, 4, res_t, arg0_t>(
                numel, data[0], data[1], size0, size1, size2, size3, size4,
                stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
          }
          return true;
        }
      }
    }


    // multiple of the element size (more than one element): the input is
    // gather-loaded element-wise, the output is vector-stored. Covers
    // direct-copy layouts like (960,44064,2,2,3) whose input dim0 stride
    // is 2 elements.
    // vt < 4 gather-copy (size0 2/50 layouts) regresses below legacy;
    // only vector stores of vt >= 4 stay specialized.
    if (out_contiguous &&
        stride01 % static_cast<int64_t>(sizeof(arg0_t)) == 0 &&
        stride01 > static_cast<int64_t>(sizeof(arg0_t)) &&
        size0 % 4 == 0) {
      const int vt_max = vector_access_width<res_t>();
      for (int vt = vt_max; vt >= 4; vt /= 2) {  // only vt >= 4
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(res_t)) * vt;
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % static_cast<uintptr_t>(sizeof(arg0_t)) != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_5_1_d0s", f);
        constexpr int nt = 128;
        if (vt == 8) {
          launch_elementwise_kernel_5_1_contiguous_out<nt, 8, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, size3, size4,
              stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
        } else {
          launch_elementwise_kernel_5_1_contiguous_out<nt, 4, res_t, arg0_t>(
              numel, data[0], data[1], size0, size1, size2, size3, size4,
              stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, stride40, stride41, f);
        }
        return true;
      }
    }

    return false;
  }
}
