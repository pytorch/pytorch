#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_backward_native.h>
#include <ATen/ops/_fw_primal_native.h>
#include <ATen/ops/_version_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/data_native.h>
#include <ATen/ops/is_leaf_native.h>
#include <ATen/ops/output_nr_native.h>
#include <ATen/ops/requires_grad_native.h>
#include <ATen/ops/retain_grad_native.h>
#include <ATen/ops/retains_grad_native.h>
#include <ATen/ops/set_data_native.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

// The stubs in here are used by dynamic dispatch. It just redirects everything
// to the Tensor method we manually bind in TensorBody.h.

namespace at::native {

void _backward(const Tensor& self, TensorList inputs, const std::optional<Tensor>& gradient_opt, std::optional<bool> keep_graph, bool create_graph) {
  return self._backward(inputs, gradient_opt, keep_graph, create_graph);
}

void set_data(Tensor& self, const Tensor& new_data) {
  return self.set_data(new_data);
}

Tensor data(const Tensor& self) {
  return self.data();
}

bool is_leaf(const Tensor& self) {
  return self.is_leaf();
}

int64_t output_nr(const Tensor& self) {
  return self.output_nr();
}

int64_t _version(const Tensor& self) {
  return self._version();
}

Tensor& requires_grad_(Tensor& self, bool _requires_grad) {
  self.requires_grad_(_requires_grad);
  return self;
}

void retain_grad(Tensor& self) {
  return self.retain_grad();
}

bool retains_grad(const Tensor& self) {
  return self.retains_grad();
}

// We expect this code to only be reached in inference mode and when all inputs are inference tensors
Tensor _fw_primal(const Tensor& self, int64_t level) {
  TORCH_INTERNAL_ASSERT(
    InferenceMode::is_enabled() && self.is_inference(),
    "Expected this method to only be reached in inference mode and when all the "
    "inputs are inference tensors. You should NOT call this method directly as "
    "native::_fw_primal. Please use the dispatcher, i.e., at::_fw_primal. Please "
    "file an issue if you come across this error otherwise.");
  return at::alias(self);
}

} // namespace at::native
