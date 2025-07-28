#pragma once

#include <cstdint>

namespace torch::nativert {

enum class OpKernelKind : uint8_t {
  kPrimKernel,
  kStaticDispatchKernel,
  kInterpreterFallbackKernel,
  // static dispatch kernels that don't reuse
  // out TensorImpl
  kNativeStaticDispatchKernel,
};

} // namespace torch::nativert
