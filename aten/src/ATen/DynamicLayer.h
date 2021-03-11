#pragma once
#include <c10/core/DispatchKey.h>
#include <c10/util/Optional.h>
#include <unordered_map>
#include <mutex>

// Forward declared bc I am lazy
namespace c10 { struct AutogradMetaInterface; }

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
TORCH_API DynamicLayer popDynamicLayerAndDeleteMetadata();
TORCH_API bool gradLayerAtTop();
TORCH_API c10::optional<DynamicLayer> maybeCurrentDynamicLayer();
TORCH_API std::vector<DynamicLayer>& getDynamicLayerStack();


using DynmetaData = std::unordered_map<int64_t, std::vector<std::weak_ptr<std::unique_ptr<c10::AutogradMetaInterface>>>>;

// NB: not lock safe
TORCH_API DynmetaData& getGlobalDynmetaData();
TORCH_API std::mutex& getGlobalDynmetaDataMutex();

} // namespace at
