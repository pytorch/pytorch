// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <c10/core/DispatchKey.h>
#include <ATen/core/function_schema.h>
#include <c10/util/Optional.h>
#include <unordered_map>
#include <mutex>

// Forward declared bc I am lazy
namespace c10 { struct AutogradMetaInterface; }

namespace at {
namespace functorch {

struct TORCH_API DynamicLayer {
  explicit DynamicLayer(
      DispatchKey key,
      int64_t layerId,
      optional<int64_t> batchSize = nullopt,
      optional<bool> prev_grad_mode = nullopt);

  DispatchKey key() const;
  int64_t layerId() const;

  // Only valid for vmap
  int64_t batchSize() const;

  // only valid for grad-based transforms
  optional<bool> prevGradMode() const;
 private:
  DispatchKey key_;
  int64_t layerId_;

  // Honestly these should be a union or some extendable metadata class.
  // Not doing that for now because I don't think we'll use this mechanism for very long.
  optional<int64_t> batchSize_;
  optional<bool> prevGradMode_;
};

TORCH_API int64_t initAndPushDynamicLayer(
    DispatchKey key,
    optional<int64_t> batch_size = nullopt,
    optional<bool> prev_grad_mode = nullopt);
TORCH_API DynamicLayer popDynamicLayerAndDeleteMetadata();
TORCH_API c10::optional<DynamicLayer> maybeCurrentDynamicLayer();
TORCH_API const std::vector<DynamicLayer>& getDynamicLayerStack();
TORCH_API void setDynamicLayerStack(const std::vector<DynamicLayer>& stack);
TORCH_API void setDynamicLayerFrontBackKeysIncluded(bool included);

// NB: Not lock safe, you should only call this from Python where the GIL will
// prevent race conditions.
TORCH_API bool areTransformsActive();

// NB: not lock safe. TODO: does it need a lock?
TORCH_API std::shared_ptr<bool> getLifeHandleForLevel(int64_t level);

// Returns if an operator is in-place. An operator is inplace if:
// 1. The first argument is a Tensor and it is being written to
// 2. The first argument is being returned
// 3. No other arguments are aliased
// Here is an example of an in-place operator:
// add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
bool isInplaceOp(const c10::FunctionSchema& schema);

// Applies the following for-loop:
// for i in range(begin, end):
//   args[i] = func(args[i])
void foreachTensorInplace(std::vector<IValue>& args, int64_t begin, int64_t end,
    std::function<Tensor(const Tensor&)> func);

// Pretty printers
std::ostream& operator<<(std::ostream& os, const DynamicLayer& layer);
std::ostream& operator<<(std::ostream& os, const std::vector<DynamicLayer>& dynamicLayerStack);

}
} // namespace at
