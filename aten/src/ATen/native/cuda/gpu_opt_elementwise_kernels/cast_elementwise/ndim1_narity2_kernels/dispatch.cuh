// File: gpu_opt_elementwise_kernels/cast_elementwise/ndim1_narity2_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_cast_elementwise_1_2
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/log.h"
#include "broadcast_in1_dim0_castin1.cuh"
#include "broadcast_in1_dim0_castin2.cuh"
#include "broadcast_in2_dim0_castin2.cuh"
#include "broadcast_in2_dim0_bf16out.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_cast_elementwise_1_2(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  const int64_t numel = iter.numel();
  if (numel <= 0) {
    return false;
  }
  // Function-level variable from the original cast_float/cast_double; the in-block
  // if (ndim == N) check references it (the entry already routes by iter.ndim(), so
  // it is always true and kept only to preserve the original semantics).
  const int64_t ndim = iter.ndim();

  if constexpr (std::is_same_v<res_t, float> &&
                std::is_same_v<typename traits::template arg<0>::type, float> &&
                std::is_same_v<typename traits::template arg<1>::type, float>) {
    if (ndim == 1) {
      if (iter.dtype(0) != ScalarType::Float) {
        return false;
      }
      const ScalarType d1 = iter.dtype(1);
      const ScalarType d2 = iter.dtype(2);
      if (d2 != ScalarType::Float) {

        // contiguous with a different storage dtype (Long) cast on load.

        // family to the mul (128,) Long-exponent patterns.
        if (d1 == ScalarType::Float && d2 == ScalarType::Long &&
            iter.strides(1)[0] == 0 && !iter.is_cpu_scalar(1) &&
            iter.strides(0)[0] == static_cast<int64_t>(sizeof(float)) &&
            iter.strides(2)[0] == static_cast<int64_t>(sizeof(int64_t))) {
          constexpr int vt0 = 4;  // 16 bytes of float
          constexpr int vt2 = 2;  // 16 bytes of int64
          constexpr int vt = vt0 > vt2 ? vt0 : vt2;
          if (numel % vt == 0 &&
              reinterpret_cast<uintptr_t>(data[0]) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(data[1]) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(data[2]) % 16 == 0) {
            log_elementwise_info(iter, "p_e_gpu_1_2_cast_cb", f);
            constexpr int nt = 128;
            launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin2<nt, float, float, int64_t>(
                numel, data[0], data[1], data[2], sizeof(float), sizeof(int64_t), f);
            return true;
          }
        }

        // IN2 Long dim0-broadcast (stride02==0), IN1 float contiguous;
        // IN2 is cast on load once per thread. tag p_e_gpu_1_2_cast_cb2.
        // Mutually exclusive with cast_cb above (stride01==0 there).
        if (d1 == ScalarType::Float && d2 == ScalarType::Long &&
            iter.strides(0)[0] == static_cast<int64_t>(sizeof(float)) &&
            iter.strides(1)[0] == static_cast<int64_t>(sizeof(float)) &&
            iter.strides(2)[0] == 0 && !iter.is_cpu_scalar(2)) {
          for (int vt = 4; vt >= 1; vt /= 2) {
            if (numel % vt != 0) {
              continue;
            }
            const int64_t al = static_cast<int64_t>(sizeof(float)) * vt;
            if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
                reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
                reinterpret_cast<uintptr_t>(data[2]) % 8 != 0) {
              continue;
            }
            log_elementwise_info(iter, "p_e_gpu_1_2_cast_cb2", f);
            constexpr int nt = 128;
            if (vt == 4) {
              launch_cast_elementwise_kernel_1_2_broadcast_in2_dim0_castin2<nt, 4>(
                  numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
            } else if (vt == 2) {
              launch_cast_elementwise_kernel_1_2_broadcast_in2_dim0_castin2<nt, 2>(
                  numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
            } else {
              launch_cast_elementwise_kernel_1_2_broadcast_in2_dim0_castin2<nt, 1>(
                  numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
            }
            return true;
          }
        }
        return false;
      }

      if (d1 != ScalarType::Long && d1 != ScalarType::Double) {
        return false;
      }
      if (iter.strides(1)[0] != 0 || iter.is_cpu_scalar(1) ||
          iter.strides(0)[0] != static_cast<int64_t>(sizeof(float)) ||
          iter.strides(2)[0] != static_cast<int64_t>(sizeof(float))) {
        return false;
      }
      // Tiny cb workloads (numel < 256) are launch-bound; pow
      // [32]/[64]/[128] Float x Long scalar measured 2.7x/3.2x slower than
      // legacy. The double twin below stays positive (pow [128] Double x
      // Long scalar measured 1.21x) and is left untouched.
      if (numel < 256) {
        return false;
      }
      for (int vt = 4; vt >= 1; vt /= 2) {
        if (numel % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(float)) * vt;
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % 8 != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_1_2_cb", f);
        constexpr int nt = 128;
        if (d1 == ScalarType::Long) {
          if (vt == 4) {
            launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, 4, float, int64_t>(
                numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
          } else if (vt == 2) {
            launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, 2, float, int64_t>(
                numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
          } else {
            launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, 1, float, int64_t>(
                numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
          }
        } else {
          if (vt == 4) {
            launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, 4, float, double>(
                numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
          } else if (vt == 2) {
            launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, 2, float, double>(
                numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
          } else {
            launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, 1, float, double>(
                numel, data[0], data[1], data[2], sizeof(float), sizeof(float), f);
          }
        }
        return true;
      }
      return false;
    }
  } else if constexpr (std::is_same_v<res_t, double> &&
                         std::is_same_v<typename traits::template arg<0>::type, double> &&
                         std::is_same_v<typename traits::template arg<1>::type, double>) {

    if (ndim == 1) {

      if (iter.dtype(0) != ScalarType::Double || iter.dtype(2) != ScalarType::Double ||
          iter.dtype(1) != ScalarType::Long) {
        return false;
      }
      if (iter.strides(1)[0] != 0 || iter.is_cpu_scalar(1) ||
          iter.strides(0)[0] != static_cast<int64_t>(sizeof(double)) ||
          iter.strides(2)[0] != static_cast<int64_t>(sizeof(double))) {
        return false;
      }
      for (int vt = 2; vt >= 1; vt /= 2) {
        if (numel % vt != 0) {
          continue;
        }
        const int64_t al = static_cast<int64_t>(sizeof(double)) * vt;
        if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[2]) % al != 0 ||
            reinterpret_cast<uintptr_t>(data[1]) % 8 != 0) {
          continue;
        }
        log_elementwise_info(iter, "p_e_gpu_1_2_cb", f);
        constexpr int nt = 128;
        if (vt == 2) {
          launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, 2, double, int64_t>(
              numel, data[0], data[1], data[2], sizeof(double), sizeof(double), f);
        } else {
          launch_cast_elementwise_kernel_1_2_broadcast_in1_dim0_castin1<nt, 1, double, int64_t>(
              numel, data[0], data[1], data[2], sizeof(double), sizeof(double), f);
        }
        return true;
      }
      return false;
    }
  } else if constexpr (std::is_same_v<res_t, c10::BFloat16> &&
                       std::is_same_v<typename traits::template arg<0>::type, c10::BFloat16> &&
                       std::is_same_v<typename traits::template arg<1>::type, c10::BFloat16>) {
    // OUT BFloat16, IN1 BFloat16 contiguous, IN2 Float dim0-broadcast
    // scalar (mul bf16 x float scalar, e.g. [1152]/[524288]/[4194304]). The
    // functor is BinaryFunctor<BFloat16,BFloat16,BFloat16,opmath>; IN2 is
    // read once per thread and converted to the functor arg type. numel >=
    // 512 keeps tiny launch-bound shapes on legacy (mirrors the
    // rationale). tag p_e_gpu_1_2_cast_cb2_bf16.
    if (ndim == 1 &&
        iter.dtype(0) == ScalarType::BFloat16 &&
        iter.dtype(1) == ScalarType::BFloat16 &&
        iter.dtype(2) == ScalarType::Float) {
      const int64_t es = static_cast<int64_t>(sizeof(c10::BFloat16));
      if (numel >= 512 &&
          iter.strides(0)[0] == es &&
          iter.strides(1)[0] == es &&
          iter.strides(2)[0] == 0 && !iter.is_cpu_scalar(2)) {
        for (int vt = 4; vt >= 1; vt /= 2) {
          if (numel % vt != 0) {
            continue;
          }
          const int64_t al = es * vt;
          if (reinterpret_cast<uintptr_t>(data[0]) % al != 0 ||
              reinterpret_cast<uintptr_t>(data[1]) % al != 0 ||
              reinterpret_cast<uintptr_t>(data[2]) % 4 != 0) {
            continue;
          }
          log_elementwise_info(iter, "p_e_gpu_1_2_cast_cb2_bf16", f);
          constexpr int nt = 128;
          if (vt == 4) {
            launch_cast_elementwise_kernel_1_2_broadcast_in2_dim0_bf16out<nt, 4, c10::BFloat16, c10::BFloat16, float>(
                numel, data[0], data[1], data[2], es, es, f);
          } else if (vt == 2) {
            launch_cast_elementwise_kernel_1_2_broadcast_in2_dim0_bf16out<nt, 2, c10::BFloat16, c10::BFloat16, float>(
                numel, data[0], data[1], data[2], es, es, f);
          } else {
            launch_cast_elementwise_kernel_1_2_broadcast_in2_dim0_bf16out<nt, 1, c10::BFloat16, c10::BFloat16, float>(
                numel, data[0], data[1], data[2], es, es, f);
          }
          return true;
        }
      }
    }
  }
  return false;
}
