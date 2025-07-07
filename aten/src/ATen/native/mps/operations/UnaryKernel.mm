#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
// #include <ATen/native/Activation.h>
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

REGISTER_UNARY_TI_DISPATCH(exp);
REGISTER_UNARY_TI_DISPATCH(expm1);
REGISTER_UNARY_TI_DISPATCH(erf);
REGISTER_UNARY_TI_DISPATCH(erfc);
REGISTER_UNARY_TI_DISPATCH(erfinv);
REGISTER_UNARY_TI_DISPATCH(sinc);
REGISTER_UNARY_TI_DISPATCH(sinh);
REGISTER_UNARY_TI_DISPATCH(cosh);
REGISTER_UNARY_TI_DISPATCH(tanh);
REGISTER_UNARY_TI_DISPATCH(abs);
REGISTER_UNARY_TI_DISPATCH(sin);
REGISTER_UNARY_TI_DISPATCH(cos);
REGISTER_UNARY_TI_DISPATCH(tan);
REGISTER_UNARY_TI_DISPATCH(asin);
REGISTER_UNARY_TI_DISPATCH(acos);
REGISTER_UNARY_TI_DISPATCH(atan);
REGISTER_UNARY_TI_DISPATCH(sqrt);
REGISTER_UNARY_TI_DISPATCH(rsqrt);
REGISTER_UNARY_TI_DISPATCH(neg);
REGISTER_UNARY_TI_DISPATCH(exp2);
REGISTER_UNARY_TI_DISPATCH(log10);
REGISTER_UNARY_TI_DISPATCH(log2);
REGISTER_UNARY_TI_DISPATCH(log);
REGISTER_UNARY_TI_DISPATCH(log1p);
REGISTER_UNARY_TI_DISPATCH(bitwise_not);
REGISTER_UNARY_TI_DISPATCH(sigmoid);
REGISTER_DISPATCH(round_decimals_stub, round_decimals_kernel);
} // namespace at::native
