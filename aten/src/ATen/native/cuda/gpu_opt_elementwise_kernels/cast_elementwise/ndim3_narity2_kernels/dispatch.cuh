// File: gpu_opt_elementwise_kernels/cast_elementwise/ndim3_narity2_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_cast_elementwise_3_2
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/log.h"
#include "broadcast_in1_dim2_in2_dim1_castin1.cuh"
#include "broadcast_in2_dim1.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_cast_elementwise_3_2(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;

  // OUT float fully contiguous, IN1 bf16 dim0+dim1 contiguous +
  // dim2-broadcast (cast on load), IN2 float dim0-contiguous +
  // dim1-broadcast + dim2-contiguous. tag p_e_gpu_3_2_cbd1f. Mutually
  // exclusive with the complex cbd1 leg below (functor type differs).
  if constexpr (std::is_same_v<res_t, float> &&
                std::is_same_v<typename traits::template arg<0>::type, float> &&
                std::is_same_v<typename traits::template arg<1>::type, float>) {
    const int64_t numel = iter.numel();
    if (numel <= 0) {
      return false;
    }
    if (iter.ndim() != 3) {
      return false;
    }
    if (iter.dtype(0) != ScalarType::Float ||
        iter.dtype(1) != ScalarType::BFloat16 ||
        iter.dtype(2) != ScalarType::Float) {
      return false;
    }
    const int64_t size0 = iter.shape()[0];
    const int64_t size1 = iter.shape()[1];
    const int64_t size2 = iter.shape()[2];
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride20 = iter.strides(0)[2];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride11 = iter.strides(1)[1];
    const int64_t stride21 = iter.strides(1)[2];
    const int64_t stride02 = iter.strides(2)[0];
    const int64_t stride12 = iter.strides(2)[1];
    const int64_t stride22 = iter.strides(2)[2];
    if (stride00 != 4 || stride10 != 4 * size0 || stride20 != 4 * size0 * size1 ||
        stride01 != 2 || stride11 != 2 * size0 || stride21 != 0 || iter.is_cpu_scalar(1) ||
        stride02 != 4 || stride12 != 0 || iter.is_cpu_scalar(2) ||
        size1 <= 0 || size1 > gpu_grid_cap(1) * 8 ||
        size2 <= 0 || size2 > gpu_grid_cap(2) * 8) {
      return false;
    }
    for (int vt = 4; vt >= 1; vt /= 2) {
      if (size0 % vt != 0) {
        continue;
      }
      const int64_t al_out = static_cast<int64_t>(sizeof(float)) * vt;
      const int64_t al_in = static_cast<int64_t>(sizeof(c10::BFloat16)) * vt;
      if (stride22 % al_out != 0 || stride11 % al_in != 0) {
        continue;
      }
      if (reinterpret_cast<uintptr_t>(data[0]) % al_out != 0 ||
          reinterpret_cast<uintptr_t>(data[1]) % al_in != 0 ||
          reinterpret_cast<uintptr_t>(data[2]) % al_out != 0) {
        continue;
      }
      log_elementwise_info(iter, "p_e_gpu_3_2_cbd1f", f);
      constexpr int nt = 128;
      constexpr int z_t = 8;
      if (vt == 4) {
        launch_cast_elementwise_kernel_3_2_broadcast_in1_dim2_in2_dim1_castin1<nt, z_t, 4>(
            numel, data[0], data[1], data[2], size0, size1, size2,
            stride00, stride01, stride02, stride10, stride11, stride12,
            stride20, stride21, stride22, f);
      } else if (vt == 2) {
        launch_cast_elementwise_kernel_3_2_broadcast_in1_dim2_in2_dim1_castin1<nt, z_t, 2>(
            numel, data[0], data[1], data[2], size0, size1, size2,
            stride00, stride01, stride02, stride10, stride11, stride12,
            stride20, stride21, stride22, f);
      } else {
        launch_cast_elementwise_kernel_3_2_broadcast_in1_dim2_in2_dim1_castin1<nt, z_t, 1>(
            numel, data[0], data[1], data[2], size0, size1, size2,
            stride00, stride01, stride02, stride10, stride11, stride12,
            stride20, stride21, stride22, f);
      }
      return true;
    }
    return false;
  }

  // ComplexFloat IN1 dim1-broadcast).
  if constexpr (std::is_same_v<res_t, c10::complex<double>> &&
                std::is_same_v<typename traits::template arg<0>::type, c10::complex<double>> &&
                std::is_same_v<typename traits::template arg<1>::type, c10::complex<double>>) {
    const int64_t numel = iter.numel();
    if (numel <= 0) {
      return false;
    }
    // This ComplexFloat -> ComplexDouble specialization is below legacy
    // through 1M. Preserve the measured 4M+ win and use cast legacy below it.
    if (numel < 4 * 1024 * 1024) {
      return false;
    }
    if (iter.ndim() != 3) {
      return false;
    }
    if (iter.dtype(0) != ScalarType::ComplexDouble ||
        iter.dtype(1) != ScalarType::ComplexFloat ||
        iter.dtype(2) != ScalarType::ComplexDouble) {
      return false;
    }
    const int64_t size0 = iter.shape()[0];
    const int64_t size1 = iter.shape()[1];
    const int64_t size2 = iter.shape()[2];
    const int64_t stride00 = iter.strides(0)[0];
    const int64_t stride10 = iter.strides(0)[1];
    const int64_t stride20 = iter.strides(0)[2];
    const int64_t stride01 = iter.strides(1)[0];
    const int64_t stride11 = iter.strides(1)[1];
    const int64_t stride21 = iter.strides(1)[2];
    const int64_t stride02 = iter.strides(2)[0];
    const int64_t stride12 = iter.strides(2)[1];
    const int64_t stride22 = iter.strides(2)[2];
    if (stride00 != 16 || stride10 != 16 * size0 || stride20 != 16 * size0 * size1 ||
        stride01 != 8 || stride11 != 8 * size0 || stride21 != 8 * size0 * size1 ||
        stride02 != 16 || stride12 != 0 || iter.is_cpu_scalar(2) ||
        stride22 % 16 != 0 || size2 <= 0 || size2 > gpu_grid_cap(2) * 8) {
      return false;
    }
    if (reinterpret_cast<uintptr_t>(data[0]) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(data[1]) % 8 != 0 ||
        reinterpret_cast<uintptr_t>(data[2]) % 16 != 0) {
      return false;
    }
    log_elementwise_info(iter, "p_e_gpu_3_2_cbd1", f);
    constexpr int nt = 128;
    constexpr int z_t = 8;
    launch_cast_elementwise_kernel_3_2_broadcast_in2_dim1<nt, z_t>(
        numel, data[0], data[1], data[2], size0, size1, size2,
        stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    return true;
  }
  return false;
}
