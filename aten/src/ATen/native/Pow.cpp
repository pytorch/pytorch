#include <ATen/native/Pow.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/Resize.h>

namespace at {
namespace meta {

TORCH_META_FUNC2(pow, Tensor_Tensor) (const Tensor& base, const Tensor& exp) {
  build_binary_op(maybe_get_output(), base, exp);
}

TORCH_META_FUNC2(pow, Tensor_Scalar) (const Tensor& base, const Scalar exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              exp.isIntegral(true) && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");

  auto common_dtype = at::result_type(base, exp);
  build_unary_op(maybe_get_output(), base.to(common_dtype));
}

} // namespace meta

namespace native {

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);

TORCH_IMPL_FUNC(pow_Tensor_Tensor_out) (const Tensor& base, const Tensor& exp, const Tensor& out) {
  if (exp.dim() == 0 && exp.device().is_cpu() && base.is_cuda()) {
    at::pow_out(const_cast<Tensor&>(out), base, exp.item()); // redispatch!
  }
  pow_tensor_tensor_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(pow_Tensor_Scalar_out) (const Tensor& base, const Scalar exp, const Tensor& out) {
  auto common_dtype = at::result_type(base, exp);
  TORCH_CHECK(at::can_cast(common_dtype, out.scalar_type()),
           "result type ", common_dtype, " can't be cast to the desired output type ",
           out.scalar_type());
  if (exp.equal(0.0)) {
    out.fill_(1);
  } else if (exp.equal(1.0)) {
    out.copy_(base);
  } else {
    pow_tensor_scalar_stub(device_type(), *this, exp);
  }
}

//Tensor& pow_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  //if (exp.dim() == 0 && exp.device().type() == DeviceType::CPU
    //&& base.device().type() == DeviceType::CUDA) {
    //return native::pow_out(result, base, exp.item());
  //}
  //auto iter = TensorIterator::binary_op(result, base, exp);
  //pow_tensor_tensor_stub(iter.device_type(), iter);
  //return result;
//}

//Tensor& pow_out(Tensor& result, const Tensor& base, Scalar exp) {
  //// Numpy compatibility check:
  //TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              //exp.isIntegral(true) && exp.toLong() < 0),
              //"Integers to negative integer powers are not allowed.");

  //auto common_dtype = at::result_type(base, exp);
  //TORCH_CHECK(at::can_cast(common_dtype, result.scalar_type()),
           //"result type ", common_dtype, " can't be cast to the desired output type ",
           //result.scalar_type());

  //if (exp.equal(0.0)) {
    //resize_output(result, base.sizes());
    //result.fill_(1);
    //namedinference::propagate_names(result, base);
  //} else if (exp.equal(1.0)) {
    //resize_output(result, base.sizes());
    //result.copy_(base);
    //namedinference::propagate_names(result, base);
  //} else {
    //auto iter = TensorIterator::unary_op(result, base.to(common_dtype));
    //pow_tensor_scalar_stub(iter.device_type(), iter, exp);
  //}
  //return result;
//}

Tensor& pow_out(Scalar base, const Tensor& exp, Tensor& result) {
  auto& result_ = const_cast<Tensor&>(result);
  if (base.isComplex() && base.toComplexDouble() == 1.0) {
    resize_output(result_, exp.sizes());
    result_.fill_(1);
    namedinference::propagate_names(result_, exp);
  } else if (!base.isComplex() && base.toDouble() == 1.0) {
    resize_output(result_, exp.sizes());
    result_.fill_(1);
    namedinference::propagate_names(result_, exp);
  } else {
    at::pow_out(result_, c10::scalar_to_tensor(base, exp.device()), exp); // redispatch!
  }
  return result_;
}

Tensor& pow_(const Tensor& base, const Tensor& exp) {
  return at::pow_out(const_cast<Tensor&>(base), base, exp); // redispatch!
}

Tensor& pow_(const Tensor& base, const Scalar exp) {
  return at::pow_out(const_cast<Tensor&>(base), base, exp); // redispatch!
}

//Tensor pow(const Tensor& base, const Tensor& exp) {
  //auto dtype = at::result_type(base, exp);
  //Tensor result = at::empty({0}, base.options().dtype(dtype));
  //return native::pow_out(result, base, exp);
//}

//Tensor pow(const Tensor& base, Scalar exp) {
  //auto dtype = at::result_type(base, exp);
  //Tensor result = at::empty_like(base, base.options().dtype(dtype), MemoryFormat::Preserve);
  //return native::pow_out(result, base, exp);
//}

Tensor pow(Scalar base, const Tensor& exp) {
  return at::pow(c10::scalar_to_tensor(base, exp.device()), exp); // redispatch!
  //auto dtype = at::result_type(base, exp);
  //Tensor result = at::empty_like(exp, exp.options().dtype(dtype), MemoryFormat::Preserve);
  //return native::pow_out(result, base, exp);
}

Tensor& float_power_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ?
                at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  return at::pow_out(result, base.to(dtype), exp.to(dtype));
}

Tensor& float_power_out(Tensor& result, const Tensor& base, Scalar exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  // Note: need the casts inside the ternary because conversion functions return e.g. c10::complex,
  // which causes a complex scalar to always be returned.
  exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return at::pow_out(result, base.to(dtype), exp);
}

Tensor& float_power_out(Tensor& result, Scalar base, const Tensor& exp) {
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  return at::pow_out(result, base, exp.to(dtype));
}

Tensor float_power(const Tensor& base, Scalar exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return at::pow(base.to(dtype), exp);
}

Tensor float_power(Scalar base, const Tensor& exp) {
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  return at::pow(base, exp.to(dtype));
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

Tensor& float_power_(Tensor& base, Scalar exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(base.scalar_type() == dtype,
              "the base given to float_power_ has dtype ", base.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return base.pow_(exp);
}

} // namespace native

} // namespace at
