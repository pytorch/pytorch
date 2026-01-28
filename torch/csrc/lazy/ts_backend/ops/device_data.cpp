#include <torch/csrc/lazy/ts_backend/ops/device_data.h>

#include <torch/csrc/lazy/core/ir_builder.h>

#include <sstream>

namespace torch::lazy {

DeviceData::DeviceData(std::shared_ptr<BackendData> data)
    : TsNode(
          ClassOpKind(),
          data->shape(),
          /*num_outputs=*/1,
          /*hash_seed=*/static_cast<uint32_t>(101)),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", device=" << data_->device();
  return ss.str();
}

const DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node);
}

NodePtr DeviceData::Create(const std::shared_ptr<BackendData>& data) {
  NodePtr node = ReuseOrMakeNode<DeviceData>(data);
  // ReuseOrMakeNode may return a reused node which has the same shape,
  // however, we need to replace the old data_ with the new one.
  // Ditching the old data_ is safe because tracing is done iteration
  // by iteration, and after we launch the async device execution for the
  // previous iteration, data_ in DeviceData nodes are not needed anymore.
  DeviceData* device_data = static_cast<DeviceData*>(node.get());
  device_data->SetData(data);
  return node;
}

} // namespace torch::lazy
