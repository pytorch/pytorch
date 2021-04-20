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
  DynamicLayer(DispatchKey key, int64_t layerId): key_(key), layerId_(layerId) {}

  DispatchKey key() const { return key_; }
  int64_t layerId() const { return layerId_; }
 private:
  DispatchKey key_;
  int64_t layerId_;
};

TORCH_API int64_t initAndPushDynamicLayer(DispatchKey key);
TORCH_API DynamicLayer popDynamicLayerAndDeleteMetadata();
TORCH_API c10::optional<DynamicLayer> maybeCurrentDynamicLayer();
TORCH_API const std::vector<DynamicLayer>& getDynamicLayerStack();
TORCH_API void setDynamicLayerStack(const std::vector<DynamicLayer>& stack);

// NB: not lock safe. TODO: does it need a lock?
TORCH_API std::shared_ptr<bool> getLifeHandleForLevel(int64_t level);

}
} // namespace at
