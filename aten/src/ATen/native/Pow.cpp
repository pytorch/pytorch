#include <ATen/native/Pow.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ScalarOps.h>

namespace at { namespace native {

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);

Tensor& pow_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto iter = TensorIterator::binary_op(result, base, exp,
                                        /*check_mem_overlap=*/true);
  pow_tensor_tensor_stub(iter.device_type(), iter);
  return result;
}

Tensor& pow_out(Tensor& result, const Tensor& base, Scalar exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              exp.isIntegral(true) && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");

  auto common_dtype = at::result_type(base, exp);
  TORCH_CHECK(at::can_cast(common_dtype, result.scalar_type()),
           "result type ", common_dtype, "can't be cast to the desired output type ",
           result.scalar_type());
  // Avoid runtime error when typecasting
  if (!exp.isComplex() && (exp.toDouble() == 0.0)) {
    result.resize_as_(base).fill_(1);
  } else if (!exp.isComplex() && (exp.toDouble() == 1.0)) {
    result.resize_as_(base).copy_(base);
  } else {
    auto iter = TensorIterator::unary_op(result, base.to(common_dtype),
                                         /*check_mem_overlap=*/true);
    pow_tensor_scalar_stub(iter.device_type(), iter, exp);
  }
  return result;
}

Tensor& pow_out(Tensor& result, Scalar base, const Tensor& exp) {
  if (base.toDouble() == 1.0) {
    result.resize_as_(exp).fill_(1);
  } else {
    native::pow_out(result, c10::scalar_to_tensor(base, exp.device()), exp);
  }
  return result;
}

Tensor& pow_(Tensor& base, const Tensor& other) {
  return native::pow_out(base, base, other);
}

Tensor& pow_(Tensor& base, Scalar alpha) {
  return native::pow_out(base, base, alpha);
}

Tensor pow(const Tensor& base, const Tensor& exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result = at::empty({0}, base.options().dtype(dtype));
  return native::pow_out(result, base, exp);
}

Tensor pow(const Tensor& base, Scalar exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result = at::empty_like(base, base.options().dtype(dtype), MemoryFormat::Preserve);
  return native::pow_out(result, base, exp);
}

Tensor pow(Scalar base, const Tensor& exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result = at::empty_like(exp, exp.options().dtype(dtype), MemoryFormat::Preserve);
  return native::pow_out(result, base, exp);
}

} // namespace native

} // namespace at
