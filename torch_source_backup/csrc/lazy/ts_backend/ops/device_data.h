#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <utility>

namespace torch::lazy {

class TORCH_API DeviceData : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return ltc_device_data;
  }

  explicit DeviceData(std::shared_ptr<BackendData> data);

  // A DeviceData node can be reused if the shape matches,
  // but we will substitute the actual data_ pointer under
  // the hood.
  bool CanBeReused(const std::shared_ptr<BackendData>& data) const {
    return data_->shape() == data->shape();
  }

  std::string ToString() const override;

  const std::shared_ptr<BackendData>& data() const {
    return data_;
  }

  void SetData(std::shared_ptr<BackendData> data) {
    data_ = std::move(data);
  }

  static const DeviceData* Cast(const Node* node);

  // To reuse IR nodes, use this method to create DeviceData nodes
  // instead of calling the constructor directconst ly.
  static NodePtr Create(const std::shared_ptr<BackendData>& data);

  TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override;

 private:
  std::shared_ptr<BackendData> data_;
};

} // namespace torch::lazy
