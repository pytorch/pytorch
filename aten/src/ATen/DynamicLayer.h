#pragma once
#include <c10/core/DispatchKey.h>
#include <c10/util/Optional.h>

namespace at {

struct TORCH_API DynamicLayer {
  DynamicLayer(DispatchKey key, int64_t layerId): key_(key), layerId_(layerId) {}

  DispatchKey key() const { return key_; }
  int64_t layerId() const { return layerId_; }
 private:
  DispatchKey key_;
  int64_t layerId_;
};

TORCH_API int64_t pushDynamicLayer(DispatchKey key);
TORCH_API DynamicLayer popDynamicLayer();
TORCH_API bool gradLayerAtTop();
TORCH_API c10::optional<DynamicLayer> maybeCurrentDynamicLayer();
TORCH_API std::vector<DynamicLayer>& getDynamicLayerStack();

} // namespace at
