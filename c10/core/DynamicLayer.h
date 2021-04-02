#pragma once
#include <c10/core/DispatchKey.h>
#include <c10/util/Optional.h>
#include <c10/core/TensorImpl.h>
#include <unordered_map>
#include <mutex>

// Forward declared bc I am lazy
// namespace c10 { struct AutogradMetaInterface; }

namespace c10 {

struct C10_API DynamicLayer {
  DynamicLayer(DispatchKey key, int64_t layerId): key_(key), layerId_(layerId) {}

  DispatchKey key() const { return key_; }
  int64_t layerId() const { return layerId_; }
 private:
  DispatchKey key_;
  int64_t layerId_;
};

C10_API int64_t pushDynamicLayer(DispatchKey key);
C10_API DynamicLayer popDynamicLayer();
C10_API DynamicLayer popDynamicLayerAndDeleteMetadata();
C10_API bool gradLayerAtTop();
C10_API c10::optional<DynamicLayer> maybeCurrentDynamicLayer();
C10_API std::vector<DynamicLayer>& getDynamicLayerStack();


using DynmetaData = std::unordered_map<int64_t, std::vector<std::weak_ptr<std::unique_ptr<c10::AutogradMetaInterface>>>>;

// NB: not lock safe
C10_API DynmetaData& getGlobalDynmetaData();
C10_API std::mutex& getGlobalDynmetaDataMutex();

} // namespace at
