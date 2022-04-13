#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API DeviceData : public TsNode {
 public:
  explicit DeviceData(std::shared_ptr<BackendData> data);

  bool Equal(std::shared_ptr<BackendData> data) const {
    return data_->shape() == data->shape();
  }

  std::string ToString() const override;

  const std::shared_ptr<BackendData>& data() const {
    return data_;
  }

  void SetData(std::shared_ptr<BackendData> data) {
    data_ = data;
  }

  static const DeviceData* Cast(const Node* node);

  static NodePtr Create(std::shared_ptr<BackendData> data);

 private:
  std::shared_ptr<BackendData> data_;
};

} // namespace lazy
} // namespace torch
