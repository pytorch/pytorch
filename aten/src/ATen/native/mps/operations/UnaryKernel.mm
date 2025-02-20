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

static void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals) {
  lib.exec_unary_kernel(iter, "round_decimals", decimals);
}

REGISTER_DISPATCH(exp_stub, exp_kernel);
REGISTER_DISPATCH(erfinv_stub, erfinv_kernel);
REGISTER_DISPATCH(sinc_stub, sinc_kernel);
REGISTER_DISPATCH(tanh_stub, tanh_kernel);
REGISTER_DISPATCH(round_decimals_stub, round_decimals_kernel);
} // namespace at::native
