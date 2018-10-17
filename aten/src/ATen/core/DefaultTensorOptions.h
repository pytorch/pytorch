#pragma once

#include <ATen/core/Backend.h>
#include <ATen/core/Device.h>
#include <ATen/core/Layout.h>
#include <ATen/core/ScalarType.h>

namespace at {

struct TensorOptions;

/// Like TensorOptions, but all fields are guaranteed to be filled.
struct DefaultTensorOptions {
  ScalarType dtype()    const noexcept { return dtype_; }
  Device device()       const noexcept { return device_; }
  Layout layout()       const noexcept { return layout_; }
  bool requires_grad()  const noexcept { return requires_grad_; }
  bool is_variable()    const noexcept { return is_variable_; }

  // Defined in TensorOptions.h
  inline DefaultTensorOptions& merge(const TensorOptions& options);

 private:
  Device device_      = at::kCPU;     // 64-bit
  ScalarType dtype_   = at::kFloat;   // 8-bit
  Layout layout_      = at::kStrided; // 8-bit
  bool requires_grad_ = false;        // 8-bit
  bool is_variable_   = false;        // 8-bit
};

// TODO: Even better would be < sizeof(int64_t)
static_assert(sizeof(DefaultTensorOptions) < sizeof(int64_t) * 2,
              "DefaultTensorOptions must fit in 128 bits");

} // namespace at
