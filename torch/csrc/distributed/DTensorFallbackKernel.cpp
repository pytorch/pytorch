#include <ATen/core/PythonFallbackKernel.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/library.h>

namespace {
bool is_same_size(const at::Tensor& self, const at::Tensor& other) {
  return self.sizes() == other.sizes();
}

} // namespace

TORCH_LIBRARY_IMPL(aten, DTensor, m) {
  // For simple custom op handlers, we can port them to C++, dispatch
  // to them right here, and skip Python dispatch and map lookup.
  m.impl("aten::is_same_size", is_same_size);
  // For custom op handlers in Python, we of course need to dispatch
  // to Python.
  m.impl(
      "aten::convolution",
      torch::CppFunction::makeFromBoxedFunction<&callDTensorCustomOpHandler>());
  m.impl(
      "aten::convolution_backward",
      torch::CppFunction::makeFromBoxedFunction<&callDTensorCustomOpHandler>());
  m.impl(
      "aten::_amp_foreach_non_finite_check_and_unscale_",
      torch::CppFunction::makeFromBoxedFunction<&callDTensorCustomOpHandler>());
}

TORCH_LIBRARY_IMPL(_, DTensor, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&callDTensorOpDispatch>());
}
