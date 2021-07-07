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

#define TENSOROPTIONSPARAMS c10::optional<c10::ScalarType> dtype, c10::optional<c10::Layout> layout, c10::optional<c10::Device> device, c10::optional<bool> pin_memory
#define TENSOROPTIONSARGS dtype, layout, device, pin_memory

Tensor randn_mbatching_rule(IntArrayRef shape, TENSOROPTIONSPARAMS) {
    TORCH_WARN("Automatically expanding the shape of randn. This means that different elements of vmap will get different random values. If you want different behavior, like using the same random values across vmap, please make a github issue.");
    c10::impl::ExcludeDispatchKeyGuard guard(kVmapModeKey);
    auto maybe_layer = maybeCurrentDynamicLayer();
    VmapDimVector shapeVec(shape.begin(), shape.end());
    shapeVec.insert(shapeVec.begin(), maybe_layer->batchSize());
    return makeBatched(at::randn(shapeVec, TENSOROPTIONSARGS), 0, maybe_layer->layerId());
}


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

  UNSUPPORTED_RANDOM2(normal, Tensor_float);
  UNSUPPORTED_RANDOM2(normal, Tensor_float_out);
  UNSUPPORTED_RANDOM2(normal, float_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, float_Tensor);
  UNSUPPORTED_RANDOM2(normal, Tensor_Tensor);
  UNSUPPORTED_RANDOM2(normal, Tensor_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, float_float);
  UNSUPPORTED_RANDOM2(normal, float_float_out);
  UNSUPPORTED_RANDOM(normal_);

  UNSUPPORTED_RANDOM(poisson);

  UNSUPPORTED_RANDOM2(random_, from);
  UNSUPPORTED_RANDOM2(random_, to);
  UNSUPPORTED_RANDOM(random_);

  // UNSUPPORTED_RANDOM(rand_like);
  // UNSUPPORTED_RANDOM(randn_like);

  UNSUPPORTED_RANDOM(randint_like);
  UNSUPPORTED_RANDOM2(randint_like, low_dtype);

  UNSUPPORTED_RANDOM(rand);
  UNSUPPORTED_RANDOM2(rand, generator);
  UNSUPPORTED_RANDOM2(rand, names);
  UNSUPPORTED_RANDOM2(rand, generator_with_names);
  UNSUPPORTED_RANDOM2(rand, out);
  UNSUPPORTED_RANDOM2(rand, generator_out);

//   UNSUPPORTED_RANDOM(randn);
  UNSUPPORTED_RANDOM2(randn, generator);
  UNSUPPORTED_RANDOM2(randn, names);
  UNSUPPORTED_RANDOM2(randn, generator_with_names);
  UNSUPPORTED_RANDOM2(randn, out);
  UNSUPPORTED_RANDOM2(randn, generator_out);

  UNSUPPORTED_RANDOM(randperm);
  UNSUPPORTED_RANDOM2(randperm, generator);
  UNSUPPORTED_RANDOM2(randperm, out);
  UNSUPPORTED_RANDOM2(randperm, generator_out);

  UNSUPPORTED_RANDOM(randint);
  UNSUPPORTED_RANDOM2(randint, generator);
  UNSUPPORTED_RANDOM2(randint, low);
  UNSUPPORTED_RANDOM2(randint, low_generator);
  UNSUPPORTED_RANDOM2(randint, out);
  UNSUPPORTED_RANDOM2(randint, generator_out);
  UNSUPPORTED_RANDOM2(randint, low_out);
  UNSUPPORTED_RANDOM2(randint, low_generator_out);

  UNSUPPORTED_RANDOM(uniform_);


  m.impl("randn", randn_mbatching_rule);
}


}
} // namespace at
