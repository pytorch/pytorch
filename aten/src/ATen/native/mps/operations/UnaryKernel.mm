#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/OperationUtils.h>

#include <fmt/format.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/UnaryKernel_metallib.h>
#endif

static void erfinv_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "erfinv");
}

static void exp_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "exp");
}

static void sinc_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "sinc");
}

static void tanh_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "tanh");
}

static void sin_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "sin");
}

static void cos_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "cos");
}

static void tan_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "tan");
}

static void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals) {
  lib.exec_unary_kernel(iter, "round_decimals", decimals);
}

static void exp2_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "exp2");
}

static void sqrt_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "sqrt");
}

static void rsqrt_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "rsqrt");
}

static void neg_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "neg");
}

static void bitwise_not_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "bitwise_not");
}

REGISTER_DISPATCH(exp_stub, exp_kernel);
REGISTER_DISPATCH(erfinv_stub, erfinv_kernel);
REGISTER_DISPATCH(sinc_stub, sinc_kernel);
REGISTER_DISPATCH(tanh_stub, tanh_kernel);
REGISTER_DISPATCH(sin_stub, sin_kernel);
REGISTER_DISPATCH(cos_stub, cos_kernel);
REGISTER_DISPATCH(tan_stub, tan_kernel);
REGISTER_DISPATCH(round_decimals_stub, round_decimals_kernel);
REGISTER_DISPATCH(sqrt_stub, sqrt_kernel_mps);
REGISTER_DISPATCH(rsqrt_stub, rsqrt_kernel_mps);
REGISTER_DISPATCH(exp2_stub, exp2_kernel_mps);
REGISTER_DISPATCH(neg_stub, neg_kernel_mps);
REGISTER_DISPATCH(bitwise_not_stub, bitwise_not_kernel_mps);
} // namespace at::native
