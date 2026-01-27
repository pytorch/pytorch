#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Pow.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/UnaryKernel_metallib.h>
#endif

// KURT: call site of `exec_unary_kernel`
#define REGISTER_UNARY_TI_DISPATCH(NAME)                    \
  static void NAME##_kernel_mps(TensorIteratorBase& iter) { \
    lib.exec_unary_kernel(iter, #NAME);                     \
  }                                                         \
  REGISTER_DISPATCH(NAME##_stub, NAME##_kernel_mps)

static void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals) {
  lib.exec_unary_kernel(iter, "round_decimals", Scalar(decimals), ScalarType::Long);
}

static void pow_tensor_scalar_kernel(TensorIteratorBase& iter, const Scalar& exp_scalar) {
  if (!exp_scalar.isComplex() && exp_scalar.to<float>() == 2.0) {
    return lib.exec_unary_kernel(iter, "sqr");
  }
  if (c10::isIntegralType(iter.common_dtype(), true)) {
    return lib.exec_unary_kernel(iter, "pow_scalar", exp_scalar, kInt);
  }
  if (!exp_scalar.isComplex() && exp_scalar.to<float>() == -1.0) {
    return lib.exec_unary_kernel(iter, "reciprocal");
  }
  if (!exp_scalar.isComplex() && exp_scalar.to<float>() == -.5) {
    return lib.exec_unary_kernel(iter, "rsqrt");
  }
  if (!exp_scalar.isComplex() && exp_scalar.to<float>() == .5) {
    return lib.exec_unary_kernel(iter, "sqrt");
  }
  if (exp_scalar.isComplex()) {
    return lib.exec_unary_kernel(iter, "pow_scalar", exp_scalar, ScalarType::ComplexFloat);
  }
  lib.exec_unary_kernel(iter, "pow_scalar", exp_scalar, ScalarType::Float);
}

static void erfcx_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "erfcx");
}

REGISTER_UNARY_TI_DISPATCH(exp);
REGISTER_UNARY_TI_DISPATCH(expm1);
REGISTER_UNARY_TI_DISPATCH(erf);
REGISTER_UNARY_TI_DISPATCH(erfc);
REGISTER_UNARY_TI_DISPATCH(erfinv);
REGISTER_UNARY_TI_DISPATCH(sinc);
REGISTER_UNARY_TI_DISPATCH(sinh);
REGISTER_UNARY_TI_DISPATCH(cosh);
REGISTER_UNARY_TI_DISPATCH(tanh);
REGISTER_UNARY_TI_DISPATCH(angle);
REGISTER_UNARY_TI_DISPATCH(abs);
REGISTER_UNARY_TI_DISPATCH(sin);
REGISTER_UNARY_TI_DISPATCH(cos);
REGISTER_UNARY_TI_DISPATCH(tan);
REGISTER_UNARY_TI_DISPATCH(asin);
REGISTER_UNARY_TI_DISPATCH(acos);
REGISTER_UNARY_TI_DISPATCH(atan);
REGISTER_UNARY_TI_DISPATCH(sqrt);
REGISTER_UNARY_TI_DISPATCH(reciprocal);
REGISTER_UNARY_TI_DISPATCH(rsqrt);
REGISTER_UNARY_TI_DISPATCH(neg);
REGISTER_UNARY_TI_DISPATCH(exp2);
REGISTER_UNARY_TI_DISPATCH(log10);
REGISTER_UNARY_TI_DISPATCH(log2);
REGISTER_UNARY_TI_DISPATCH(log);
REGISTER_UNARY_TI_DISPATCH(log1p);
REGISTER_UNARY_TI_DISPATCH(bitwise_not);
REGISTER_UNARY_TI_DISPATCH(round);
REGISTER_UNARY_TI_DISPATCH(sigmoid);
REGISTER_DISPATCH(special_erfcx_stub, erfcx_kernel);
REGISTER_DISPATCH(round_decimals_stub, round_decimals_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, pow_tensor_scalar_kernel);
} // namespace at::native
