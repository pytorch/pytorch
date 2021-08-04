#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <ATen/PythonMode.h>

namespace at {

TORCH_LIBRARY_IMPL(aten, PythonMode, m) {
  ${python_mode_registrations};
}

} // namespace at
