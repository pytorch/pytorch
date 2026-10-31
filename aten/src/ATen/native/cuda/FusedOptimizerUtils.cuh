#pragma once

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <algorithm>

namespace at::native {

inline void check_fused_optimizer_step_dtypes(TensorList state_steps) {
  TORCH_CHECK(!state_steps.empty(), "Tensor list must have at least one tensor.");
  const auto step_dtype = state_steps[0].scalar_type();
  TORCH_CHECK(
      std::all_of(
          state_steps.begin(),
          state_steps.end(),
          [step_dtype](const Tensor& step) {
            return step.scalar_type() == step_dtype;
          }),
      "Fused optimizer step tensors must have the same dtype");
}

} // namespace at::native

#define AT_DISPATCH_FUSED_OPTIMIZER_STEP_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE,                                                 \
      NAME,                                                 \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                       \
          at::ScalarType::Float, step_t, __VA_ARGS__)         \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                   \
              at::ScalarType::Double, step_t, __VA_ARGS__))
