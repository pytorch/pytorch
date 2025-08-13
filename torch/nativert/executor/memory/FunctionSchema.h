#pragma once

#include <ATen/core/function_schema.h>
#include <torch/nativert/executor/OpKernelKind.h>

namespace torch::nativert {

struct InputOutputIdxPair {
  size_t input_idx;
  size_t output_idx;
};

using AliasingSpec = std::vector<InputOutputIdxPair>;

class FunctionSchema {
 public:
  explicit FunctionSchema(
      const c10::FunctionSchema& schema,
      AliasingSpec&& aliasing_spec = {},
      torch::nativert::OpKernelKind kernel_kind =
          torch::nativert::OpKernelKind::kInterpreterFallbackKernel)
      : aliasing_spec_(std::move(aliasing_spec)),
        kernel_kind_(kernel_kind),
        c10_fn_schema_(schema) {}

  c10::FunctionSchema& base_schema() {
    return c10_fn_schema_;
  }

  const c10::FunctionSchema& base_schema() const {
    return c10_fn_schema_;
  }

  bool alias(size_t input_idx, size_t output_idx) const;

  C10_ALWAYS_INLINE torch::nativert::OpKernelKind kernel_kind() const {
    return kernel_kind_;
  }

 private:
  AliasingSpec aliasing_spec_;
  torch::nativert::OpKernelKind kernel_kind_;
  c10::FunctionSchema c10_fn_schema_;
};

} // namespace torch::nativert
