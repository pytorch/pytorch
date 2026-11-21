// File: gpu_opt_elementwise_kernels/cast_contiguous_unrolled/ndim1_narity2_kernels/dispatch.cuh
// Dispatch: try_launch_gpu_cast_contiguous_1_2
// Tags: dispatched by the functions below.

#pragma once

#include "../../utils/log.h"
#include "contiguous_castin2.cuh"

template <typename func_t, typename array_t>
static inline bool try_launch_gpu_cast_contiguous_1_2(TensorIteratorBase& iter, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;

  // contiguous arity-2 cast with a float functor; IN1 float contiguous, IN2
  // BFloat16/Long/Int cast on load. Any mismatch falls through to the
  // unrolled path.
  // This entry is reached by functors of every arity (including arity-0
  // FillFunctor and arity-1 lambdas), so the arg<> types must be extracted

  // instantiated, while types in its condition are.
  if constexpr (traits::arity == 2) {
    if constexpr (std::is_same_v<res_t, float> &&
                  std::is_same_v<typename traits::template arg<0>::type, float> &&
                  std::is_same_v<typename traits::template arg<1>::type, float>) {
      const int64_t numel = iter.numel();
      if (numel <= 0 || iter.ndim() != 1 || !iter.is_contiguous()) {
        return false;
      }
      const ScalarType d0 = iter.dtype(0);
      const ScalarType d1 = iter.dtype(1);
      const ScalarType d2 = iter.dtype(2);
      if (d0 != ScalarType::Float) {
        return false;
      }
      // Sub-variant: IN1 bf16 cast on load, IN2 float contiguous.
      // Mutually exclusive with the castin2 legs below (d1 differs); placed
      // first only for code organization.
      if (d1 == ScalarType::BFloat16 && d2 == ScalarType::Float) {
        const bool launched = launch_unrolled_elementwise_kernel_1_2_contiguous_castin1<c10::BFloat16>(iter, f, data, numel);
        if (launched) log_elementwise_info(iter, "p_e_gpu_1_2_cast_v", f);
        return launched;
      }
      if (d1 != ScalarType::Float) {
        return false;
      }
      if (d2 == ScalarType::BFloat16) {
        const bool launched = launch_unrolled_elementwise_kernel_1_2_contiguous_castin2<c10::BFloat16>(iter, f, data, numel);
        if (launched) log_elementwise_info(iter, "p_e_gpu_1_2_cast_v", f);
        return launched;
      } else if (d2 == ScalarType::Long) {
        const bool launched = launch_unrolled_elementwise_kernel_1_2_contiguous_castin2<int64_t>(iter, f, data, numel);
        if (launched) log_elementwise_info(iter, "p_e_gpu_1_2_cast_v", f);
        return launched;
      } else if (d2 == ScalarType::Int) {
        const bool launched = launch_unrolled_elementwise_kernel_1_2_contiguous_castin2<int32_t>(iter, f, data, numel);
        if (launched) log_elementwise_info(iter, "p_e_gpu_1_2_cast_v", f);
        return launched;
      }
    }
  }
  return false;
}
