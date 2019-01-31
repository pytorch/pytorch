#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>


namespace at {
namespace native{
DEFINE_DISPATCH(hypot_stub);
Tensor& hypot_out(Tensor& out, Tensor const& a, Tensor const& b) {
  auto iter = TensorIterator::binary_op(out, a, b);
  hypot_stub(iter->device_type(), *iter);
  return out;
}

Tensor hypot(Tensor const& a, Tensor const& b) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, a, b);
  hypot_stub(iter->device_type(), *iter);
  return iter->output();
}
}
}
