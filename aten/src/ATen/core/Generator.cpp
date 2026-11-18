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

void Generator::graphsafe_set_state(const Generator& new_state) {
  this->impl_->graphsafe_set_state(new_state.getIntrusivePtr());
}

Generator Generator::graphsafe_get_state() const {
  return Generator(this->impl_->graphsafe_get_state());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> Generator::philox_state(
    uint64_t increment) {
  std::tuple<at::Tensor, at::Tensor, at::Tensor> state;
  this->impl_->philox_state(
      increment, std::get<0>(state), std::get<1>(state), std::get<2>(state));
  return state;
}

} // namespace at
