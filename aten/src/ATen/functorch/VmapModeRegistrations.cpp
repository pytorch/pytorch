// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/core/dispatch/Dispatcher.h>

// functorch's vmap has two Dispatch Keys that implement it:
// FuncTorchBatched and FuncTorchVmapMode. This file contains registrations for
// FuncTorchVmapMode -- these registrations are to error out on operations
// that we don't support on regular Tensors.

namespace at {
namespace functorch {

void unsupportedRandomOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "vmap: We do not support calling out variants of random operations inside of vmap. ",
              "Please use non-out variants as a workaround");
}

TORCH_LIBRARY_IMPL(_, FuncTorchVmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

void nyiRandomOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "vmap: we do not yet support ", op.schema().operator_name(),
              ". Please file an issue");
}

#define UNSUPPORTED_RANDOM(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define UNSUPPORTED_RANDOM2(op, overload) \
  m.impl(#op"."#overload, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define NYI_RANDOM(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&nyiRandomOp>());

#define NYI_RANDOM2(op, overload) \
  m.impl(#op"."#overload, torch::CppFunction::makeFromBoxedFunction<&nyiRandomOp>());

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  UNSUPPORTED_RANDOM2(bernoulli, out);
  UNSUPPORTED_RANDOM2(rand, generator_out);
  UNSUPPORTED_RANDOM2(rand, out);
  UNSUPPORTED_RANDOM2(randint, generator_out);
  UNSUPPORTED_RANDOM2(randint, out);
  UNSUPPORTED_RANDOM2(randn, generator_out);
  UNSUPPORTED_RANDOM2(randn, out);
  UNSUPPORTED_RANDOM2(randperm, generator_out);
  UNSUPPORTED_RANDOM2(randperm, out);
  UNSUPPORTED_RANDOM2(multinomial, out);
  UNSUPPORTED_RANDOM2(normal, float_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, Tensor_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, float_float_out);
  UNSUPPORTED_RANDOM2(rrelu_with_noise, out);

  NYI_RANDOM(rrelu_with_noise);
  NYI_RANDOM(rrelu_with_noise_);
  NYI_RANDOM(rrelu_);
  NYI_RANDOM(rrelu);
}


}
} // namespace at
