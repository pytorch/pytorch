#include "lazy_tensor_core/csrc/ops/device_data.h"

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

#include <sstream>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DeviceData::DeviceData(std::shared_ptr<torch::lazy::BackendData> data)
    : torch::lazy::TsNode(torch::lazy::ltc_device_data, data->shape(),
                          /*num_outputs=*/1,
                          /*hash_seed=*/static_cast<uint32_t>(101)),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", device=" << data_->device();
  return ss.str();
}

const DeviceData* DeviceData::Cast(const torch::lazy::Node* node) {
  return torch::lazy::NodeCast<DeviceData>(node, torch::lazy::ltc_device_data);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
