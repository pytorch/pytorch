// File: gpu_opt_elementwise_kernels/cast_elementwise/ndim2_narity2_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_cast_elementwise_2_2
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/log.h"
#include "broadcast_in2_dim0_grid2d.cuh"
#include "broadcast_in2_dim0_grid2d_ytiled.cuh"
#include "broadcast_in1_dim1.cuh"
#include "broadcast_in2_dim1.cuh"
#include "castin1_in2_dim0.cuh"
#include "crossbroadcast_in1_dim0_in2_dim1.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_cast_elementwise_2_2(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  const int64_t numel = iter.numel();
  if (numel <= 0) {
    return false;
  }
  const int64_t ndim = iter.ndim();  // as in cast_1_2: referenced by in-block if (ndim == N)

  if constexpr (std::is_same_v<res_t, float> &&
                std::is_same_v<typename traits::template arg<0>::type, float> &&
                std::is_same_v<typename traits::template arg<1>::type, float>) {
    if (ndim == 2) {

      if (iter.dtype(0) != ScalarType::Float) {
        return false;
      }
      const int64_t size0 = iter.shape()[0];
      const int64_t size1 = iter.shape()[1];
      const int64_t stride00 = iter.strides(0)[0];
      const int64_t stride10 = iter.strides(0)[1];
      const int64_t stride01 = iter.strides(1)[0];
      const int64_t stride11 = iter.strides(1)[1];
      const int64_t stride02 = iter.strides(2)[0];
      const int64_t stride12 = iter.strides(2)[1];

      // Sub-variant 1: IN1 bf16 fully contiguous, IN2 float dim0-broadcast.
      // size1 <= 65535 keeps the 1-row-per-block grid; larger size1 uses the

      if (iter.dtype(1) == ScalarType::BFloat16 && iter.dtype(2) == ScalarType::Float &&
          stride00 == static_cast<int64_t>(sizeof(float)) &&
          stride10 == static_cast<int64_t>(sizeof(float)) * size0 &&
          stride01 == static_cast<int64_t>(sizeof(c10::BFloat16)) &&
          stride11 == static_cast<int64_t>(sizeof(c10::BFloat16)) * size0 &&
          stride02 == 0 && !iter.is_cpu_scalar(2) &&
          stride12 == static_cast<int64_t>(sizeof(float)) &&
          size1 <= gpu_grid_cap(1) * 8) {
        // Historical 65535 threshold: above it the y-tiled variant wins
        // (measured 2.2-2.8x over the 2D variant, 2_2 cbd0 4M/16M).
        const bool use_yt = size1 > gpu_grid_cap(1);
        for (int vt = 4; vt >= 1; vt /= 2) {
          if (size0 % vt != 0) {
            continue;
          }
          const int64_t al_out = static_cast<int64_t>(sizeof(float)) * vt;
          const int64_t al_in = static_cast<int64_t>(sizeof(c10::BFloat16)) * vt;
          if (reinterpret_cast<uintptr_t>(data[0]) % al_out != 0 ||
              reinterpret_cast<uintptr_t>(data[1]) % al_in != 0 ||
              reinterpret_cast<uintptr_t>(data[2]) % static_cast<uintptr_t>(sizeof(float)) != 0) {
            continue;
          }
          log_elementwise_info(iter, use_yt ? "p_e_gpu_2_2_cbd0_yt" : "p_e_gpu_2_2_cbd0_2d", f);
          constexpr int nt = 128;
          if (use_yt) {
            if (vt == 4) {
              launch_cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d_ytiled<nt, 4>(
                  numel, data[0], data[1], data[2], size0, size1,
                  stride00, stride01, stride02, stride10, stride11, stride12, f);
            } else if (vt == 2) {
              launch_cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d_ytiled<nt, 2>(
                  numel, data[0], data[1], data[2], size0, size1,
                  stride00, stride01, stride02, stride10, stride11, stride12, f);
            } else {
              launch_cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d_ytiled<nt, 1>(
                  numel, data[0], data[1], data[2], size0, size1,
                  stride00, stride01, stride02, stride10, stride11, stride12, f);
            }
          } else {
            if (vt == 4) {
              launch_cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d<nt, 4>(
                  numel, data[0], data[1], data[2], size0, size1,
                  stride00, stride01, stride02, stride10, stride11, stride12, f);
            } else if (vt == 2) {
              launch_cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d<nt, 2>(
                  numel, data[0], data[1], data[2], size0, size1,
                  stride00, stride01, stride02, stride10, stride11, stride12, f);
            } else {
              launch_cast_elementwise_kernel_2_2_broadcast_in2_dim0_grid2d<nt, 1>(
                  numel, data[0], data[1], data[2], size0, size1,
                  stride00, stride01, stride02, stride10, stride11, stride12, f);
            }
          }
          return true;
        }
        return false;
      }

      // Sub-variant 2: IN1 float fully contiguous, IN2 bf16 dim1-broadcast.
      if (iter.dtype(1) == ScalarType::Float && iter.dtype(2) == ScalarType::BFloat16 &&
          stride00 == static_cast<int64_t>(sizeof(float)) &&
          stride10 == static_cast<int64_t>(sizeof(float)) * size0 &&
          stride01 == static_cast<int64_t>(sizeof(float)) &&
          stride11 == static_cast<int64_t>(sizeof(float)) * size0 &&
          stride02 == static_cast<int64_t>(sizeof(c10::BFloat16)) &&
          stride12 == 0 && !iter.is_cpu_scalar(2) &&
          size1 <= gpu_grid_cap(1) * 8) {
        for (int vt = 4; vt >= 1; vt /= 2) {
          if (size0 % vt != 0) {
            continue;
          }
          const int64_t al_out = static_cast<int64_t>(sizeof(float)) * vt;
          const int64_t al_in = static_cast<int64_t>(sizeof(c10::BFloat16)) * vt;
          if (reinterpret_cast<uintptr_t>(data[0]) % al_out != 0 ||
              reinterpret_cast<uintptr_t>(data[1]) % al_out != 0 ||
              reinterpret_cast<uintptr_t>(data[2]) % al_in != 0) {
            continue;
          }
          log_elementwise_info(iter, "p_e_gpu_2_2_cbd1", f);
          constexpr int nt = 128;
          if (vt == 4) {
            launch_cast_elementwise_kernel_2_2_broadcast_in2_dim1<nt, 4>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          } else if (vt == 2) {
            launch_cast_elementwise_kernel_2_2_broadcast_in2_dim1<nt, 2>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          } else {
            launch_cast_elementwise_kernel_2_2_broadcast_in2_dim1<nt, 1>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          }
          return true;
        }
        return false;
      }

      // Sub-variant: IN1 bf16 dim1-broadcast (stride11==0), IN2 float fully
      // contiguous. IN1's vt-wide bf16 vector load hoists out of the row loop
      // (stride11==0 makes the hoist legal); IN2 is vector-read per row.
      // tag p_e_gpu_2_2_cbd1i. Mutually exclusive with sub1 (stride02==0) and
      // sub2 (d1==Float && d2==BFloat16).
      if (iter.dtype(1) == ScalarType::BFloat16 && iter.dtype(2) == ScalarType::Float &&
          stride00 == static_cast<int64_t>(sizeof(float)) &&
          stride10 == static_cast<int64_t>(sizeof(float)) * size0 &&
          stride01 == static_cast<int64_t>(sizeof(c10::BFloat16)) &&
          stride11 == 0 && !iter.is_cpu_scalar(1) &&
          stride02 == static_cast<int64_t>(sizeof(float)) &&
          stride12 == static_cast<int64_t>(sizeof(float)) * size0 &&
          size1 <= gpu_grid_cap(1) * 8) {
        for (int vt = 4; vt >= 1; vt /= 2) {
          if (size0 % vt != 0) {
            continue;
          }
          const int64_t al_out = static_cast<int64_t>(sizeof(float)) * vt;
          const int64_t al_in = static_cast<int64_t>(sizeof(c10::BFloat16)) * vt;
          if (reinterpret_cast<uintptr_t>(data[0]) % al_out != 0 ||
              reinterpret_cast<uintptr_t>(data[1]) % al_in != 0 ||
              reinterpret_cast<uintptr_t>(data[2]) % al_out != 0) {
            continue;
          }
          log_elementwise_info(iter, "p_e_gpu_2_2_cbd1i", f);
          constexpr int nt = 128;
          if (vt == 4) {
            launch_cast_elementwise_kernel_2_2_broadcast_in1_dim1<nt, 4>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          } else if (vt == 2) {
            launch_cast_elementwise_kernel_2_2_broadcast_in1_dim1<nt, 2>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          } else {
            launch_cast_elementwise_kernel_2_2_broadcast_in1_dim1<nt, 1>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          }
          return true;
        }
        return false;
      }

      // Sub-variant: IN1 bf16 fully contiguous (cast on load), IN2 float
      // dim0-contiguous with arbitrary dim1 stride (gap or dim1 broadcast).
      // Both operands vector-read per row; IN2's load stays in the row loop
      // because stride12 is not guaranteed 0. tag p_e_gpu_2_2_cbd1f.
      // Mutually exclusive with cbd1i (stride11==0 vs ==es*size0), cbd1
      // (d2==BFloat16 vs Float) and cbd0 (stride02==0 vs ==es).
      if (iter.dtype(1) == ScalarType::BFloat16 && iter.dtype(2) == ScalarType::Float &&
          stride00 == static_cast<int64_t>(sizeof(float)) &&
          stride10 == static_cast<int64_t>(sizeof(float)) * size0 &&
          stride01 == static_cast<int64_t>(sizeof(c10::BFloat16)) &&
          stride11 == static_cast<int64_t>(sizeof(c10::BFloat16)) * size0 && !iter.is_cpu_scalar(1) &&
          stride02 == static_cast<int64_t>(sizeof(float)) && !iter.is_cpu_scalar(2) &&
          size1 <= gpu_grid_cap(1) * 8) {
        for (int vt = 4; vt >= 1; vt /= 2) {
          if (size0 % vt != 0) {
            continue;
          }
          const int64_t al_out = static_cast<int64_t>(sizeof(float)) * vt;
          const int64_t al_in = static_cast<int64_t>(sizeof(c10::BFloat16)) * vt;
          if (stride12 % al_out != 0) {
            continue;
          }
          if (reinterpret_cast<uintptr_t>(data[0]) % al_out != 0 ||
              reinterpret_cast<uintptr_t>(data[1]) % al_in != 0 ||
              reinterpret_cast<uintptr_t>(data[2]) % al_out != 0) {
            continue;
          }
          log_elementwise_info(iter, "p_e_gpu_2_2_cbd1f", f);
          constexpr int nt = 128;
          if (vt == 4) {
            launch_cast_elementwise_kernel_2_2_castin1_in2_dim0<nt, 4>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          } else if (vt == 2) {
            launch_cast_elementwise_kernel_2_2_castin1_in2_dim0<nt, 2>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          } else {
            launch_cast_elementwise_kernel_2_2_castin1_in2_dim0<nt, 1>(
                numel, data[0], data[1], data[2], size0, size1,
                stride00, stride01, stride02, stride10, stride11, stride12, f);
          }
          return true;
        }
        return false;
      }
    }
  } else if constexpr (std::is_same_v<res_t, double> &&
                         std::is_same_v<typename traits::template arg<0>::type, double> &&
                         std::is_same_v<typename traits::template arg<1>::type, double>) {

    if (ndim == 2) {

      if (iter.dtype(0) != ScalarType::Double ||
          iter.dtype(1) != ScalarType::Long || iter.dtype(2) != ScalarType::Double) {
        return false;
      }
      const int64_t size0 = iter.shape()[0];
      const int64_t size1 = iter.shape()[1];
      const int64_t stride00 = iter.strides(0)[0];
      const int64_t stride10 = iter.strides(0)[1];
      const int64_t stride01 = iter.strides(1)[0];
      const int64_t stride11 = iter.strides(1)[1];
      const int64_t stride02 = iter.strides(2)[0];
      const int64_t stride12 = iter.strides(2)[1];
      if (stride00 != static_cast<int64_t>(sizeof(double)) ||
          stride10 != static_cast<int64_t>(sizeof(double)) * size0 ||
          stride01 != 0 || iter.is_cpu_scalar(1) ||
          stride11 != static_cast<int64_t>(sizeof(int64_t)) ||
          stride02 != static_cast<int64_t>(sizeof(double)) ||
          stride12 != 0 || iter.is_cpu_scalar(2) ||
          size1 <= 0 || size1 > gpu_grid_cap(1)) {
        return false;
      }
      for (int vt = 2; vt >= 1; vt /= 2) {
        if (size0 % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(double)) * vt;
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % 8 != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % al != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_2_2_cbx", f);
        constexpr int nt = 128;
        if (vt == 2) {
          launch_cast_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1<nt, 2>(
              numel, data[0], data[1], data[2], size0, size1,
              stride00, stride01, stride02, stride10, stride11, stride12, f);
        } else {
          launch_cast_elementwise_kernel_2_2_crossbroadcast_in1_dim0_in2_dim1<nt, 1>(
              numel, data[0], data[1], data[2], size0, size1,
              stride00, stride01, stride02, stride10, stride11, stride12, f);
        }
        return true;
      }
      return false;
    }
  }
  return false;
}
