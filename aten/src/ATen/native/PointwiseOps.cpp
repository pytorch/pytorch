// Ternary and higher-order pointwise operations
#include <ATen/native/PointwiseOps.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/core/EnableNamedTensor.h>

#include <ATen/NamedTensorUtils.h>

namespace at {
namespace native {

Tensor addcmul(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  Tensor result = at::empty({0}, self.options());
  return at::addcmul_out(result, self, tensor1, tensor2, value);
}

Tensor& addcmul_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  return at::addcmul_out(self, self, tensor1, tensor2, value);
}

Tensor& addcmul_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  checkBackend("addcmul_cpu", result, self.type().backend());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(self);
  iter.add_input(tensor1);
  iter.add_input(tensor2);
  iter.build();
  addcmul_stub(iter.device_type(), iter, value);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(result, self);
#endif
  return result;
}

Tensor addcdiv(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  Tensor result = at::empty({0}, self.options());
  return at::addcdiv_out(result, self, tensor1, tensor2, value);
}

Tensor& addcdiv_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  return at::addcdiv_out(self, self, tensor1, tensor2, value);
}

Tensor& addcdiv_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  checkBackend("addcdiv_cpu", result, self.type().backend());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(self);
  iter.add_input(tensor1);
  iter.add_input(tensor2);
  iter.build();
  addcdiv_stub(iter.device_type(), iter, value);
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(result, self);
#endif
  return result;
}

DEFINE_DISPATCH(addcmul_stub);
DEFINE_DISPATCH(addcdiv_stub);

} // namespace native
} // namespace at
