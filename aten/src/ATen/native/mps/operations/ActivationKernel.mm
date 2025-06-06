#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Scalar.h>
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
// #include <ATen/native/Activation.h>
#include <ATen/native/Activation.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ActivationKernel0_metallib.h>
#endif

static void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& lambd=0.5) {
  float lambda = lambd.toFloat();
  lib.exec_unary_kernel(iter, "hardshrink", lambda);
}

REGISTER_DISPATCH(hardshrink_stub, hardshrink_kernel);

} // namespace at::native
