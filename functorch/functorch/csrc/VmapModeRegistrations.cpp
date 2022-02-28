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
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

TORCH_LIBRARY_IMPL(_, FuncTorchVmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}


#define UNSUPPORTED_RANDOM(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define UNSUPPORTED_RANDOM2(op, overload) \
  m.impl(#op"."#overload, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());


TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  UNSUPPORTED_RANDOM(bernoulli);
  UNSUPPORTED_RANDOM2(bernoulli, out);
  UNSUPPORTED_RANDOM2(bernoulli, p);
  UNSUPPORTED_RANDOM2(bernoulli_, Tensor);
  UNSUPPORTED_RANDOM(bernoulli_.float);

  UNSUPPORTED_RANDOM(cauchy_);
  UNSUPPORTED_RANDOM(exponential_);
  UNSUPPORTED_RANDOM(geometric_);
  UNSUPPORTED_RANDOM(log_normal_);
  UNSUPPORTED_RANDOM(multinomial);
  UNSUPPORTED_RANDOM2(multinomial, out);

  UNSUPPORTED_RANDOM(poisson);

  UNSUPPORTED_RANDOM(randint_like);
  UNSUPPORTED_RANDOM2(randint_like, low_dtype);

  UNSUPPORTED_RANDOM(uniform_);
}


}
} // namespace at
