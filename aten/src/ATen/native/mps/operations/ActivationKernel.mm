#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Activation.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ActivationKernel_metallib.h>
#endif

static void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& lambda = 0.5) {
  lib.exec_unary_kernel(iter, "hardshrink", lambda);
}

static void hardshrink_backward_kernel(TensorIteratorBase& iter, const Scalar& lambda = 0.5) {
  lib.exec_binary_kernel(iter, "hardshrink_backward", lambda);
}

static void hardsigmoid_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "hardsigmoid");
}

static void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "hardsigmoid_backward");
}

static void hardswish_kernel(at::TensorIterator& iter) {
  lib.exec_unary_kernel(iter, "hardswish");
}

static void hardswish_backward_kernel(at::TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "hardswish_backward");
}

static void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negative_slope) {
  lib.exec_unary_kernel(iter, "leaky_relu", negative_slope);
}

static void leaky_relu_backward_kernel(TensorIteratorBase& iter, const Scalar& negative_slope) {
  lib.exec_binary_kernel(iter, "leaky_relu_backward", negative_slope);
}

REGISTER_DISPATCH(hardshrink_stub, hardshrink_kernel);
REGISTER_DISPATCH(shrink_backward_stub, hardshrink_backward_kernel);
REGISTER_DISPATCH(hardsigmoid_stub, hardsigmoid_kernel);
REGISTER_DISPATCH(hardsigmoid_backward_stub, hardsigmoid_backward_kernel);
REGISTER_DISPATCH(hardswish_stub, hardswish_kernel);
REGISTER_DISPATCH(hardswish_backward_stub, hardswish_backward_kernel);
REGISTER_DISPATCH(leaky_relu_stub, leaky_relu_kernel);
REGISTER_DISPATCH(leaky_relu_backward_stub, leaky_relu_backward_kernel);

} // namespace at::native
