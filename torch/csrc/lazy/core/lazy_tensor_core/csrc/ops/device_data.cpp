#include "lazy_tensor_core/csrc/ops/device_data.h"

#include <sstream>

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DeviceData::DeviceData(std::shared_ptr<lazy_tensors::client::Data> data)
    : Node(ltc_device_data, data->shape(),
           /*num_outputs=*/1,
           /*hash_seed=*/101),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", device=" << data_->device();
  return ss.str();
}

NodePtr DeviceData::Clone(OpList operands) const {
  return MakeNode<DeviceData>(data_);
}

DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node, ltc_device_data);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
