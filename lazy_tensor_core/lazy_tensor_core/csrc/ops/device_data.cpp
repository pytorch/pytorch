#include "lazy_tensor_core/csrc/ops/device_data.h"

#include <sstream>

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DeviceData::DeviceData(std::shared_ptr<lazy_tensors::client::Data> data)
    : TsNode(ltc_device_data, data->shape(),
           /*num_outputs=*/1,
           /*hash_seed=*/static_cast<uint32_t>(101)),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", device=" << data_->device();
  return ss.str();
}

NodePtr DeviceData::Clone(OpList operands) const {
  return torch::lazy::MakeNode<DeviceData>(data_);
}

const DeviceData* DeviceData::Cast(const torch::lazy::Node* node) {
  return torch::lazy::NodeCast<DeviceData>(node, ltc_device_data);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
