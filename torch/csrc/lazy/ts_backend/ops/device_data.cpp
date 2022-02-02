#include <torch/csrc/lazy/ts_backend/ops/device_data.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

#include <sstream>

namespace torch {
namespace lazy {

DeviceData::DeviceData(std::shared_ptr<BackendData> data)
    : TsNode(
          ltc_device_data,
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
  return NodeCast<DeviceData>(node, ltc_device_data);
}

} // namespace lazy
} // namespace torch
