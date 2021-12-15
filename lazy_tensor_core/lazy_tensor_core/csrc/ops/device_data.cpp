#include "lazy_tensor_core/csrc/ops/device_data.h"

#include <sstream>

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
// 
#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DeviceData::DeviceData(std::shared_ptr<torch::lazy::BackendData> data)
    : TsNode(ltc_device_data, data->shape(),
             /*num_outputs=*/1,
             /*hash_seed=*/static_cast<uint32_t>(101)),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", device=" << data_->device();
  return ss.str();
}

const DeviceData* DeviceData::Cast(const torch::lazy::Node* node) {
  return torch::lazy::NodeCast<DeviceData>(node, ltc_device_data);
}


}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
