#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class DeviceData : public torch::lazy::TsNode {
 public:
  DeviceData(std::shared_ptr<torch::lazy::BackendData> data);

  std::string ToString() const override;

  const std::shared_ptr<torch::lazy::BackendData>& data() const {
    return data_;
  }

  static const DeviceData* Cast(const torch::lazy::Node* node);

 private:
  std::shared_ptr<torch::lazy::BackendData> data_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
