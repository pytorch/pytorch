// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>
#include <ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h>
#include <c10/core/TensorOptions.h>

namespace at {

namespace {

TORCH_LIBRARY_IMPL(_, DeviceGuard, m) {
  // TODO: proper fallback
  m.fallback(torch::CppFunction::makeFallthrough());
}

${device_guard_function_definitions}

TORCH_LIBRARY_IMPL(aten, DeviceGuard, m) {
  ${device_guard_function_registrations};
}

} // namespace
}
