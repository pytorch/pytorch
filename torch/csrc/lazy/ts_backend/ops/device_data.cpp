#include <torch/csrc/lazy/generated/LazyNonNativeIr.h>


namespace torch {
namespace lazy {

bool DeviceData::CanBeReused(const std::shared_ptr<BackendData>& data) const {
  return this->data->shape() == data->shape();
}

NodePtr DeviceData::Create(const std::shared_ptr<BackendData>& data) {
  NodePtr node = ReuseOrMakeNode<DeviceData>(data);
  // ReuseOrMakeNode may return a reused node which has the same shape,
  // however, we need to replace the old data_ with the new one.
  // Ditching the old data_ is safe because tracing is done iteration
  // by iteration, and after we lauch the async device execution for the
  // previous iteration, data_ in DeviceData nodes are not needed anymore.
  DeviceData* device_data = static_cast<DeviceData*>(node.get());
  device_data->data = data;
  return node;
}

} // namespace lazy
} // namespace torch
