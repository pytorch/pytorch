#include "caffe2/onnx/onnxifi_graph_info.h"

namespace caffe2 {
namespace onnx {

SharedPtrBackendGraphInfo OnnxBackendGraphMap::lookup(const std::string& key) {
  std::lock_guard<std::mutex> guard(backend_graph_map_lock_);
  auto it = backend_graph_map_.find(key);
  if (it != backend_graph_map_.end()) {
    return it->second;
  }
  return nullptr;
}

SharedPtrBackendGraphInfo OnnxBackendGraphMap::insert(
    const std::string& key,
    BackendGraphInfo backend_graph_info) {
  // First acquire lock.
  std::lock_guard<std::mutex> guard(backend_graph_map_lock_);
  // Then check if the backend_graph_info already exists in the map.
  if (backend_graph_map_.find(key) != backend_graph_map_.end()) {
    // If it already exists, return it.
    // The onus is on the caller to release onnxGraph pointed by
    // backend_graph_info
    return backend_graph_map_[key];
  }
  const auto& ret_pair = backend_graph_map_.emplace(
      key, std::make_shared<BackendGraphInfo>(std::move(backend_graph_info)));
  return ret_pair.first->second;
}

void OnnxBackendGraphMap::remove(const std::string& key) {
  SharedPtrBackendGraphInfo tmp;
  {
    std::lock_guard<std::mutex> guard(backend_graph_map_lock_);
    auto it = backend_graph_map_.find(key);
    if (it != backend_graph_map_.end()) {
      if (it->second.unique()) {
        tmp = it->second;
        backend_graph_map_.erase(it);
      }
    }
  }
}

OnnxBackendGraphMap* getOnnxBackendGraphMap() {
  static OnnxBackendGraphMap onnx_backend_graph_map;
  return &onnx_backend_graph_map;
}

} // namespace onnx
} // namespace caffe2
