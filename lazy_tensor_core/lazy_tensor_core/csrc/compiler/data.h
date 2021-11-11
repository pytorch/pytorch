#pragma once

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/shape.h>

#include <string.h>

namespace torch_lazy_tensors {
namespace compiler {

class BackendData {
 public:
  struct Info {
    /**
     * Used by Lazy Graph Executor to tag info on BackendData objs
     * */
    virtual ~Info() {}
  };
  /**
   * Represents (Tensor) data stored on a backend device
   * in its native format.
   * */
  using Handle = int64_t;

  BackendData(const torch::lazy::BackendDevice& device, const torch::lazy::Shape& shape)
      : device_(device), shape_(shape) {}

  virtual ~BackendData() {}

  const torch::lazy::BackendDevice& device() const { return device_; }

  const torch::lazy::Shape& shape() const { return shape_; }

  Info* info() const { return info_.get(); }

  std::shared_ptr<Info> SetInfo(std::shared_ptr<Info> info) {
    std::swap(info, info_);
    return info;
  }

  virtual Handle GetOpaqueHandle() = 0;

  virtual void Assign(const BackendData& data) = 0;

  virtual bool HasValue() const = 0;

 private:
  torch::lazy::BackendDevice device_;
  torch::lazy::Shape shape_;
  std::shared_ptr<Info> info_;
};

using BackendDataPtr = std::shared_ptr<BackendData>;

}
}  // namespace torch_lazy_tensors
