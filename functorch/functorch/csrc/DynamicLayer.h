// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <functorch/csrc/Macros.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/function_schema.h>
#include <c10/util/Optional.h>
#include <c10/util/variant.h>
#include <unordered_map>
#include <mutex>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <functorch/csrc/Interpreter.h>
#include <functorch/csrc/VmapInterpreter.h>
#include <functorch/csrc/ADInterpreters.h>
#include <functorch/csrc/FunctionalizeInterpreter.h>

// Forward declared bc I am lazy
namespace c10 { struct AutogradMetaInterface; }

namespace at {
namespace functorch {

// TODO: we can excise DynamicLayer in favor of Interpreter,
// But I am going to leave it for now as a compatiblity shim to avoid
// needing to refactor a lot of callsites...
struct FUNCTORCH_API DynamicLayer {
  explicit DynamicLayer(
      TransformType transform_type,
      int64_t layerId,
      optional<int64_t> batchSize = nullopt,
      optional<RandomnessType> randomness = nullopt,
      optional<bool> prev_grad_mode = nullopt,
      optional<bool> pre_fwd_grad_mode = nullopt,
      optional<bool> functionalize_add_back_views = nullopt);

  TransformType key() const;
  int64_t layerId() const;

  const Interpreter& interpreter() const { return interpreter_; }
  Interpreter& interpreter() { return interpreter_; }

  // Only valid for vmap
  int64_t batchSize() const;
  RandomnessType randomness() const;

 private:
  Interpreter interpreter_;
};

FUNCTORCH_API int64_t initAndPushDynamicLayer(
    TransformType transform_type,
    optional<int64_t> batch_size = nullopt,
    optional<RandomnessType> randomness = nullopt,
    optional<bool> prev_grad_mode = nullopt,
    optional<bool> prev_fwd_grad_mode = nullopt,
    optional<bool> functionalize_add_back_views = nullopt);
FUNCTORCH_API DynamicLayer popDynamicLayerAndDeleteMetadata();
FUNCTORCH_API c10::optional<DynamicLayer> maybeCurrentDynamicLayer();
FUNCTORCH_API const std::vector<DynamicLayer>& getDynamicLayerStack();
FUNCTORCH_API void setDynamicLayerStack(const std::vector<DynamicLayer>& stack);
FUNCTORCH_API void setDynamicLayerFrontBackKeysIncluded(bool included);

// NB: Not lock safe, you should only call this from Python where the GIL will
// prevent race conditions.
FUNCTORCH_API bool areTransformsActive();

// NB: not lock safe. TODO: does it need a lock?
FUNCTORCH_API std::shared_ptr<bool> getLifeHandleForLevel(int64_t level);

// Returns if an operator is in-place. An operator is inplace if:
// 1. The first argument is a Tensor and it is being written to
// 2. The first argument is being returned
// 3. No other arguments are aliased
// Here is an example of an in-place operator:
// add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
bool isInplaceOp(const c10::FunctionSchema& schema);

Tensor unwrapIfDead(const Tensor& tensor);

// Pretty printers
std::ostream& operator<<(std::ostream& os, const DynamicLayer& layer);
std::ostream& operator<<(std::ostream& os, const std::vector<DynamicLayer>& dynamicLayerStack);

void setInplaceRequiresGradAllowed(bool allowed);
bool getInplaceRequiresGradAllowed();


}
} // namespace at
