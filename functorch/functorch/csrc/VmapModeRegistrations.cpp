// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <functorch/csrc/VmapTransforms.h>
#include <functorch/csrc/BatchedTensorImpl.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>
#include <functorch/csrc/DynamicLayer.h>


namespace at {
namespace functorch {

void unsupportedRandomOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "vmap: We do not support calling out variants of random operations inside of vmap. ",
              "Please use non-out variants as a workaround");
}

TORCH_LIBRARY_IMPL(_, FuncTorchVmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}


#define UNSUPPORTED_RANDOM(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define UNSUPPORTED_RANDOM2(op, overload) \
  m.impl(#op"."#overload, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());


TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  UNSUPPORTED_RANDOM2(bernoulli, out);
}


}
} // namespace at
