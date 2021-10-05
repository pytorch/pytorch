// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <c10/core/DispatchKey.h>
#include <c10/util/Optional.h>
#include <unordered_map>
#include <mutex>

// Forward declared bc I am lazy
namespace c10 { struct AutogradMetaInterface; }

namespace at {
namespace functorch {

struct TORCH_API DynamicLayer {
  DynamicLayer(
      DispatchKey key,
      int64_t layerId,
      optional<int64_t> batchSize = nullopt,
      optional<bool> prev_grad_mode = nullopt):
    key_(key), layerId_(layerId), batchSize_(batchSize), prevGradMode_(prev_grad_mode)
  {
    if (key_ == DispatchKey::Autograd) {
      TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
    }
  }

  DispatchKey key() const { return key_; }
  int64_t layerId() const { return layerId_; }
  // Only valid for vmap
  int64_t batchSize() const {
    TORCH_INTERNAL_ASSERT(batchSize_);
    return *batchSize_;
  }
  // only valid for grad-based transforms
  optional<bool> prevGradMode() const {
    return prevGradMode_;
  }
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

// NB: Not lock safe, you should only call this from Python where the GIL will
// prevent race conditions.
TORCH_API bool areTransformsActive();

// NB: not lock safe. TODO: does it need a lock?
TORCH_API std::shared_ptr<bool> getLifeHandleForLevel(int64_t level);

}
} // namespace at
