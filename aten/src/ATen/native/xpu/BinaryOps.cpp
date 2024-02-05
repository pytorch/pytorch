#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/xpu/sycl/BinaryKernels.h>
#include <ATen/ScalarOps.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace xpu {

Tensor add_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self_arg, other_arg);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return out;
}

Tensor& add_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  auto iter = TensorIterator::binary_op(self, self, other_arg);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return self;
}

Tensor add_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::add_tensor(self_arg, wrapper, alpha);
}

Tensor& add_scalar_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::add_tensor_(self, wrapper, alpha);
}

Tensor sub_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  Tensor out;
  native::sub_check(self_arg, other_arg);
  auto iter = TensorIterator::binary_op(out, self_arg, other_arg);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::sub_kernel(iter, alpha);
  return out;
}

Tensor& sub_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  native::sub_check(self, other_arg);
  auto iter = TensorIterator::binary_op(self, self, other_arg);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::sub_kernel(iter, alpha);
  return self;
}

Tensor sub_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::sub_tensor(self_arg, wrapper, alpha);
}

Tensor& sub_scalar_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::sub_tensor_(self, wrapper, alpha);
}

Tensor mul_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self_arg, other_arg);
  native::xpu::mul_kernel(iter);
  return out;
}

Tensor& mul_tensor_(Tensor& self, const Tensor& other_arg) {
  auto iter = TensorIterator::binary_op(self, self, other_arg);
  native::xpu::mul_kernel(iter);
  return self;
}

Tensor mul_scalar(const Tensor& self_arg, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::mul_tensor(self_arg, wrapper);
}

Tensor& mul_scalar_(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::mul_tensor_(self, wrapper);
}

Tensor div_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self_arg, other_arg);
  native::xpu::div_kernel(iter);
  return out;
}

Tensor& div_tensor_(Tensor& self, const Tensor& other_arg) {
  auto iter = TensorIterator::binary_op(self, self, other_arg);
  native::xpu::div_kernel(iter);
  return self;
}

Tensor div_scalar(const Tensor& self_arg, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::div_tensor(self_arg, wrapper);
}

Tensor& div_scalar_(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::div_tensor_(self, wrapper);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Scalar"), TORCH_FN(add_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Scalar"), TORCH_FN(add_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"), TORCH_FN(add_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"), TORCH_FN(add_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Scalar"), TORCH_FN(sub_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Scalar"), TORCH_FN(sub_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Tensor"), TORCH_FN(sub_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Tensor"), TORCH_FN(sub_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Scalar"), TORCH_FN(mul_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Scalar"), TORCH_FN(mul_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Tensor"), TORCH_FN(mul_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Tensor"), TORCH_FN(mul_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Scalar"), TORCH_FN(div_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Scalar"), TORCH_FN(div_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Tensor"), TORCH_FN(div_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Tensor"), TORCH_FN(div_tensor_));
}

}}} // at::native::xpu
