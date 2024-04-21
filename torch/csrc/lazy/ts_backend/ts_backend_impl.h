#pragma once

#include <torch/csrc/lazy/backend/backend_interface.h>

namespace torch {
namespace lazy {

class TORCH_API TSData : public torch::lazy::BackendData {
 public:
  TSData(const at::Scalar& scalar, const torch::lazy::BackendDevice& device)
      : torch::lazy::BackendData(device, torch::lazy::Shape(scalar.type(), {})),
        scalar(scalar) {}

  TSData(
      const at::Tensor& data,
      const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device)
      : torch::lazy::BackendData(device, shape), data_(data) {}

  TSData(
      const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device)
      : torch::lazy::BackendData(device, shape) {}

  Handle GetHandle() override {
    return reinterpret_cast<int64_t>(this);
  }

  void Assign(const torch::lazy::BackendData& data) override {
    data_ = static_cast<const TSData&>(data).data_;
  }

  bool HasValue() const override {
    return data_.defined();
  }

  at::Tensor data() {
    return data_;
  }

  c10::optional<at::Scalar> scalar;

 private:
  at::Tensor data_;
};

TORCH_API torch::lazy::BackendImplInterface* GetTSBackendImpl();

TORCH_API void InitTorchScriptBackend();

} // namespace lazy
} // namespace torch
