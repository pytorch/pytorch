#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace at {

void Generator::set_state(at::Tensor& new_state) {
  this->impl_->set_state(*new_state.unsafeGetTensorImpl());
}

at::Tensor Generator::state() const {
  return at::Tensor::wrap_tensor_impl(this->impl_->state());
}

} // namespace at
