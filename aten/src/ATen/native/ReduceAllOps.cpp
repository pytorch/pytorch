#include <ATen/native/ReduceAllOps.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);
DEFINE_DISPATCH(_aminmax_all_stub);

Tensor min(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor result = at::empty({}, self.options());
  min_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

Tensor max(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor result = at::empty({}, self.options());
  max_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

std::tuple<Tensor, Tensor> _aminmax_all(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor min_result = at::empty({}, self.options());
  Tensor max_result = at::empty({}, self.options());
  _aminmax_all_stub(self.device().type(), min_result, max_result, self.contiguous());
  return std::tuple<Tensor&, Tensor&>(min_result, max_result);
}

}} // namesapce at::native
