#include <ATen/native/ReduceAllOps.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);
DEFINE_DISPATCH(min_and_max_all_stub);

Tensor min(const Tensor &self) {
  TORCH_CHECK(!self.is_complex(), "min is not yet implemented for complex tensors.");
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor result = at::empty({}, self.options());
  min_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

Tensor max(const Tensor &self) {
  TORCH_CHECK(!self.is_complex(), "max is not yet implemented for complex tensors.");
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor result = at::empty({}, self.options());
  max_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

Tensor min_and_max(const Tensor &self) {
  TORCH_CHECK(!self.is_complex(), "max is not yet implemented for complex tensors.");
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor min_result = at::empty({}, self.options());
  Tensor max_result = at::empty({}, self.options());
  min_and_max_all_stub(self.device().type(), min_result, max_result, self.contiguous());
  Tensor result = at::empty({2}, self.options());
  result[0] = min_result.item();
  result[1] = max_result.item();
  return result;
}

}} // namesapce at::native
