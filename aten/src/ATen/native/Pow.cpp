#include <ATen/native/Pow.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/Resize.h>

namespace at { namespace native {

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);

Tensor& pow_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  if (exp.dim() == 0 && exp.device().is_cpu() && base.is_cuda()) {
    return native::pow_out(result, base, exp.item());
  }
  auto iter = TensorIterator::binary_op(result, base, exp);
  pow_tensor_tensor_stub(iter.device_type(), iter);
  return result;
}

Tensor& pow_out(Tensor& result, const Tensor& base, const Scalar& exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              exp.isIntegral(true) && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");

  auto common_dtype = at::result_type(base, exp);
  TORCH_CHECK(at::can_cast(common_dtype, result.scalar_type()),
           "result type ", common_dtype, " can't be cast to the desired output type ",
           result.scalar_type());

  if (exp.equal(0.0)) {
    resize_output(result, base.sizes());
    result.fill_(1);
    namedinference::propagate_names(result, base);
  } else if (exp.equal(1.0)) {
    resize_output(result, base.sizes());
    result.copy_(base);
    namedinference::propagate_names(result, base);
  } else {
    auto iter = TensorIterator::unary_op(result, base.to(common_dtype));
    pow_tensor_scalar_stub(iter.device_type(), iter, exp);
  }
  return result;
}

Tensor& pow_out(Tensor& result, const Scalar& base, const Tensor& exp) {
  if (base.isComplex() && base.toComplexDouble() == 1.0) {
    resize_output(result, exp.sizes());
    result.fill_(1);
    namedinference::propagate_names(result, exp);
  } else if (!base.isComplex() && base.toDouble() == 1.0) {
    resize_output(result, exp.sizes());
    result.fill_(1);
    namedinference::propagate_names(result, exp);
  } else {
    native::pow_out(result, c10::scalar_to_tensor(base, exp.device()), exp);
  }
  return result;
}

Tensor& pow_(Tensor& base, const Tensor& other) {
  return native::pow_out(base, base, other);
}

Tensor& pow_(Tensor& base, const Scalar& alpha) {
  return native::pow_out(base, base, alpha);
}

Tensor pow(const Tensor& base, const Tensor& exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result = at::empty({0}, base.options().dtype(dtype));
  return native::pow_out(result, base, exp);
}

Tensor pow(const Tensor& base, const Scalar& exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result = at::empty_like(base, base.options().dtype(dtype), MemoryFormat::Preserve);
  return native::pow_out(result, base, exp);
}

Tensor pow(const Scalar& base, const Tensor& exp) {
  auto dtype = at::result_type(base, exp);
  Tensor result = at::empty_like(exp, exp.options().dtype(dtype), MemoryFormat::Preserve);
  return native::pow_out(result, base, exp);
}

Tensor& float_power_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ?
                at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  return at::pow_out(result, base.to(dtype), exp.to(dtype));
}

Tensor& float_power_out(Tensor& result, const Tensor& base, const Scalar& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  // Note: need the casts inside the ternary because conversion functions return e.g. c10::complex,
  // which causes a complex scalar to always be returned.
  auto casted_exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return at::pow_out(result, base.to(dtype), casted_exp);
}

Tensor& float_power_out(Tensor& result, const Scalar& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  auto casted_base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  return at::pow_out(result, casted_base, exp.to(dtype));
}

Tensor float_power(const Tensor& base, const Scalar& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  auto casted_exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return at::pow(base.to(dtype), casted_exp);
}

Tensor float_power(const Scalar& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  auto casted_base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  return at::pow(casted_base, exp.to(dtype));
}

Tensor float_power(const Tensor& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ? at::kComplexDouble : at::kDouble;
  return at::pow(base.to(dtype), exp.to(dtype));
}

Tensor& float_power_(Tensor& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(base.scalar_type() == dtype,
              "the base given to float_power_ has dtype ", base.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  return base.pow_(exp.to(dtype));
}

Tensor& float_power_(Tensor& base, const Scalar& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(base.scalar_type() == dtype,
              "the base given to float_power_ has dtype ", base.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  auto casted_exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return base.pow_(casted_exp);
}

} // namespace native

} // namespace at
