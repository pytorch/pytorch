#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

// The stubs in here are only used for static dispatch.
// Dynamic dispatch doesn't ever see those but directly goes to the kernels in VariableTypeManual.cpp.
// TODO Since the plan is to remove static dispatch, we should then also remove this file here.

namespace at {
namespace native {

void _backward(const Tensor& self, TensorList inputs, const Tensor& gradient, c10::optional<bool> keep_graph, bool create_graph) {
  AT_ERROR("backward is not implemented for Tensor");
}

void set_data(Tensor& self, const Tensor& new_data) {
  AT_ERROR("set_data is not implemented for Tensor");
}

Tensor data(const Tensor& self) {
  AT_ERROR("data is not implemented for Tensor");
}

bool is_leaf(const Tensor& self) {
  AT_ERROR("is_leaf is not implemented for Tensor");
}

int64_t output_nr(const Tensor& self) {
  AT_ERROR("output_nr is not implemented for Tensor");
}

int64_t _version(const Tensor& self) {
  AT_ERROR("version is not implemented for Tensor");
}

Tensor& requires_grad_(Tensor& self, bool _requires_grad) {
  AT_ERROR("requires_grad_ is not implemented for Tensor");
}

void retain_grad(Tensor& self) {
  AT_ERROR("retain_grad is not implemented for Tensor");
}

Tensor _fw_primal(const Tensor& self, int64_t level) {
  AT_ERROR("_fw_primal is not implemented for Tensor");
}

} // namespace native
} // namespace at
