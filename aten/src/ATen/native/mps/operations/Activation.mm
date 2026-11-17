//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Activation.h>
#include <ATen/native/Gelu.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/relu_native.h>
#endif

using namespace at::mps;

namespace at::native {
using namespace mps;

TORCH_IMPL_FUNC(gelu_out_mps)(const Tensor& self, std::string_view approximate, const Tensor& output) {
  GeluKernel(kMPS, *this, get_gelutype_enum(approximate));
}

TORCH_IMPL_FUNC(gelu_backward_out_mps)
(const Tensor& grad, const Tensor& self, std::string_view approximate, const Tensor& grad_input) {
  GeluBackwardKernel(kMPS, *this, get_gelutype_enum(approximate));
}

} // namespace at::native
