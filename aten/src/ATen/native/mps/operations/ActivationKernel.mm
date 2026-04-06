#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Activation.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/rsub.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sigmoid_native.h>
#endif
#include <ATen/native/mps/kernels/Activation.h>
#include <fmt/format.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ActivationKernel_metallib.h>
#endif

Tensor relu_mps(const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "relu is not supported for complex types");
  auto output = at::empty_like(self);
  if (output.numel() == 0)
    return output;
  auto iter = at::TensorIteratorConfig().add_output(output).add_input(self).build();
  lib.exec_unary_kernel(iter, "relu", /*alpha=*/std::nullopt, /*scalar_arg_type=*/std::nullopt, /*supports_vec4=*/true);
  return output;
}

Tensor& relu_mps_(Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "relu is not supported for complex types");
  if (self.numel() == 0)
    return self;
  auto iter = at::TensorIteratorConfig().add_output(self).add_input(self).set_check_mem_overlap(false).build();
  lib.exec_unary_kernel(iter, "relu", /*alpha=*/std::nullopt, /*scalar_arg_type=*/std::nullopt, /*supports_vec4=*/true);
  return self;
}

static void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& lambda = 0.5) {
  lib.exec_unary_kernel(iter, "hardshrink", lambda);
}

static void softshrink_kernel(TensorIteratorBase& iter, const Scalar& lambda = 0.5) {
  lib.exec_unary_kernel(iter, "softshrink", lambda);
}

static void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& lambda = 0.5) {
  lib.exec_binary_kernel(iter, "shrink_backward", lambda);
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

static void elu_kernel(TensorIteratorBase& iter, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, iter.common_dtype(), "elu_mps", [&]() {
    ELUParams<scalar_t> params{alpha.to<scalar_t>(), scale.to<scalar_t>(), input_scale.to<scalar_t>()};
    lib.exec_unary_kernel_with_params(
        iter, "elu", params, fmt::format("ELUParams_{}", mps::scalarToMetalTypeString(iter.common_dtype())));
  });
}

static void elu_backward_kernel(TensorIteratorBase& iter,
                                const Scalar& alpha,
                                const Scalar& scale,
                                const Scalar& input_scale,
                                bool is_result) {
  AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, iter.common_dtype(), "elu_backward_mps", [&]() {
    ELUBackwardParams<scalar_t> params{
        alpha.to<scalar_t>(), scale.to<scalar_t>(), input_scale.to<scalar_t>(), is_result};
    lib.exec_binary_kernel_with_params(
        iter,
        "elu_backward",
        params,
        fmt::format("ELUBackwardParams_{}", mps::scalarToMetalTypeString(iter.common_dtype())));
  });
}

static void silu_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.common_dtype())) {
    auto out = iter.output(0);
    auto self = iter.input(0);
    at::mul_out(out, self, at::sigmoid(self));
    return;
  }
  lib.exec_unary_kernel(iter, "silu", /*alpha=*/std::nullopt, /*scalar_arg_type=*/std::nullopt, /*supports_vec4=*/true);
}

static void silu_backward_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.common_dtype())) {
    auto grad_input = iter.output(0);
    auto grad_output = iter.input(0);
    auto self = iter.input(1);
    auto sig = at::sigmoid(self);
    auto one_minus_sig = at::rsub(sig, 1);
    auto inner = at::add(at::mul(self, one_minus_sig), 1);
    grad_input.copy_(at::mul(grad_output, at::mul(sig, inner)));
    return;
  }
  lib.exec_binary_kernel(iter, "silu_backward");
}

static void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negative_slope) {
  lib.exec_unary_kernel(iter, "leaky_relu", negative_slope);
}

static void leaky_relu_backward_kernel(TensorIteratorBase& iter, const Scalar& negative_slope) {
  lib.exec_binary_kernel(iter, "leaky_relu_backward", negative_slope);
}

REGISTER_DISPATCH(hardshrink_stub, hardshrink_kernel);
REGISTER_DISPATCH(softshrink_stub, softshrink_kernel);
REGISTER_DISPATCH(shrink_backward_stub, shrink_backward_kernel);
REGISTER_DISPATCH(hardsigmoid_stub, hardsigmoid_kernel);
REGISTER_DISPATCH(hardsigmoid_backward_stub, hardsigmoid_backward_kernel);
REGISTER_DISPATCH(hardswish_stub, hardswish_kernel);
REGISTER_DISPATCH(hardswish_backward_stub, hardswish_backward_kernel);
REGISTER_DISPATCH(elu_stub, elu_kernel);
REGISTER_DISPATCH(elu_backward_stub, elu_backward_kernel);
REGISTER_DISPATCH(leaky_relu_stub, leaky_relu_kernel);
REGISTER_DISPATCH(leaky_relu_backward_stub, leaky_relu_backward_kernel);
REGISTER_DISPATCH(silu_stub, silu_kernel);
REGISTER_DISPATCH(silu_backward_stub, silu_backward_kernel);

} // namespace at::native
