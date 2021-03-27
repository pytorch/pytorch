#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

// The stubs in here are used by dynamic dispatch. It just redirects everything
// to the Tensor method we manually bind in TensorBody.h.

namespace at {
namespace native {

void _backward(const Tensor& self, TensorList inputs, const c10::optional<Tensor>& gradient_opt, c10::optional<bool> keep_graph, bool create_graph) {
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
  return self.requires_grad_(_requires_grad);
}

void retain_grad(Tensor& self) {
  return self.retain_grad();
}

Tensor _fw_primal(const Tensor& self, int64_t level) {
  AT_ERROR("_fw_primal is not implemented for Tensor");
}

} // namespace native
} // namespace at
