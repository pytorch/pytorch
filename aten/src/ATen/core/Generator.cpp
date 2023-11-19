#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace at {

void Generator::set_state(const at::Tensor& new_state) {
  TORCH_CHECK(new_state.defined(), "Undefined tensor is not allowed");
  this->impl_->set_state(*new_state.unsafeGetTensorImpl());
}

at::Tensor Generator::get_state() const {
  return at::Tensor::wrap_tensor_impl(this->impl_->get_state());
}

size_t Generator::register_state_with_index(const at::Tensor& new_state) {
  TORCH_CHECK(new_state.defined(), "Undefined tensor is not allowed");
  return this->impl_->register_state_with_index(*new_state.unsafeGetTensorImpl());
}

} // namespace at
