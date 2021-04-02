#include <c10/core/DynamicLayer.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10 {

// Initial autograd layer, because autograd is always "on"
std::vector<DynamicLayer> dynamicLayerStack = { DynamicLayer(DispatchKey::Autograd, 1) };

DynmetaData kDynMetaDataSingleton;
std::mutex kDynMetaDataSingletonMutex;

DynmetaData& getGlobalDynmetaData() {
  return kDynMetaDataSingleton;
}

std::mutex& getGlobalDynmetaDataMutex() {
  return kDynMetaDataSingletonMutex;
}


bool gradLayerAtTop() {
  return dynamicLayerStack.back().key() == DispatchKey::Autograd;
}

optional<DynamicLayer> maybeCurrentDynamicLayer() {
  // NB: Exception for regular autograd, maybe tweak this
  if (dynamicLayerStack.size() <= 1) {
    return {};
  }
  return dynamicLayerStack.back();
}

std::vector<DynamicLayer>& getDynamicLayerStack() {
  return dynamicLayerStack;
}

int64_t pushDynamicLayer(DispatchKey key) {
  TORCH_INTERNAL_ASSERT(key != DispatchKey::Undefined);
  auto layerId = 1 + dynamicLayerStack.size();
  dynamicLayerStack.emplace_back(key, layerId);

  if (layerId == 2) {
    // std::cout << "DynamicLayer on" << std::endl;
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);
  }

  return layerId;
}

DynamicLayer popDynamicLayer() {
  TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
  auto result = dynamicLayerStack.back();
  TORCH_INTERNAL_ASSERT(result.key() != DispatchKey::Undefined);
  dynamicLayerStack.pop_back();

  if (dynamicLayerStack.size() == 0) {
    // std::cout << "DynamicLayer off" << std::endl;
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, false);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, false);
  }

  return result;
}

DynamicLayer popDynamicLayerAndDeleteMetadata() {
  auto result = popDynamicLayer();
  auto level = result.layerId();

  // There is unfortunately a deadlock somewhere :/
  // std::lock_guard<std::mutex> guard(getGlobalDynmetaDataMutex());
  auto& data = getGlobalDynmetaData();
  auto it = data.find(level);
  if (it == data.end()) {
    return result;
  }
  for (auto& ptr : it->second) {
    auto val = ptr.lock();
    if (!val) continue;
    // Clear the unique_ptr inside the shared_ptr.
    (*val).reset();
  }
  // Clear the queue of weak_ptrs
  data[level].clear();

  return result;
}

} // namespace c10
