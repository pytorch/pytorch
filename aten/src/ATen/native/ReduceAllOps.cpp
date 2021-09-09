#include <ATen/native/ReduceAllOps.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace meta {

void check_not_empty(const char* name, const Tensor& self) {
  TORCH_CHECK(
      self.numel() > 0,
      name, ": Expected reduction dim to be specified for input.numel() == 0. ",
      "Specify the reduction dim with the 'dim' argument.");
}

TORCH_META_FUNC(min)(const Tensor& self) {
  check_not_empty("min()", self);
  set_output({}, self.options());
}

TORCH_META_FUNC(max)(const Tensor& self) {
  check_not_empty("max()", self);
  set_output({}, self.options());
}

} // namespace meta

namespace native {

DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);

TORCH_IMPL_FUNC(min_all_out)(const Tensor& self, const Tensor& result) {
  min_all_stub(self.device().type(), result, self.contiguous());
}

TORCH_IMPL_FUNC(max_all_out)(const Tensor& self, const Tensor& result) {
  max_all_stub(self.device().type(), result, self.contiguous());
}

// DEPRECATED: Use at::aminmax instead
std::tuple<Tensor, Tensor> _aminmax_all(const Tensor &self) {
  return at::aminmax(self);
}

}} // namespace at::native
