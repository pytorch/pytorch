#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/ir.h>

namespace torch {
namespace lazy {

class TORCH_API DeviceData : public Node {
 public:
  explicit DeviceData(std::shared_ptr<BackendData> data);

  std::string ToString() const override;

  const std::shared_ptr<BackendData>& data() const {
    return data_;
  }

  static const DeviceData* Cast(const Node* node);

 private:
  std::shared_ptr<BackendData> data_;
};

} // namespace lazy
} // namespace torch
