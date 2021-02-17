#pragma once
#include <torch/library.h>

namespace at {

struct TORCH_API DynamicLayer {
  DynamicLayer(DispatchKey key, int64_t layerId): key_(key), layerId_(layerId) {}

  DispatchKey key() { return key_; }
  int64_t layerId() { return layerId_; }
 private:
  DispatchKey key_;
  int64_t layerId_;
};

TORCH_API int64_t pushDynamicLayer(DispatchKey key);
TORCH_API DynamicLayer popDynamicLayer();
TORCH_API bool gradLayerAtTop();

} // namespace at
