#include <ATen/native/ReduceAllOps.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);

Tensor min(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor result = at::empty({}, self.options());
  min_all_stub(kCPU, result, self.contiguous());
  return result;
}

Tensor max(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor result = at::empty({}, self.options());
  max_all_stub(kCPU, result, self.contiguous());
  return result;
}

}} // namesapce at::native
