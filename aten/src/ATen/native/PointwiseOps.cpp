// Ternary and higher-order pointwise operations
#include <ATen/native/PointwiseOps.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/NamedTensorUtils.h>

namespace at {
namespace native {

Tensor addcmul(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  Tensor result = at::empty({0}, self.options());
  return at::addcmul_out(result, self, tensor1, tensor2, value);
}

Tensor& addcmul_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  return at::addcmul_out(self, self, tensor1, tensor2, value);
}

Tensor& addcmul_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  checkBackend("addcmul_cpu", result, self.options().backend());
  auto iter = at::TensorIteratorConfig()
    .add_output(result)
    .add_input(self)
    .add_input(tensor1)
    .add_input(tensor2)
    .build();
  addcmul_stub(iter.device_type(), iter, value);
  return result;
}

Tensor addcdiv(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  Tensor result = at::empty({0}, self.options());
  return at::addcdiv_out(result, self, tensor1, tensor2, value);
}

Tensor& addcdiv_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  return at::addcdiv_out(self, self, tensor1, tensor2, value);
}

Tensor& addcdiv_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  if (isIntegralType(tensor1.scalar_type(), /*includeBool=*/ true)
      && isIntegralType(tensor2.scalar_type(), /*includeBool=*/ true)) {
    TORCH_CHECK(false,
      "Integer division with addcdiv is no longer supported, and in a future  ",
      "release addcdiv will perform a true division of tensor1 and tensor2. ",
      "The historic addcdiv behavior can be implemented as ",
      "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) ",
      "for integer inputs and as ",
      "(input + value * tensor1 / tensor2) for float inputs. ",
      "The future addcdiv behavior is just the latter implementation: ",
      "(input + value * tensor1 / tensor2), for all dtypes.");
  }
  checkBackend("addcdiv_cpu", result, self.options().backend());
  auto iter = at::TensorIteratorConfig()
    .add_output(result)
    .add_input(self)
    .add_input(tensor1)
    .add_input(tensor2)
    .build();
  addcdiv_stub(iter.device_type(), iter, value);
  return result;
}

DEFINE_DISPATCH(addcmul_stub);
DEFINE_DISPATCH(addcdiv_stub);

} // namespace native
} // namespace at
