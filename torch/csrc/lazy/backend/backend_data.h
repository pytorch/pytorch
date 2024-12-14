#pragma once

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/shape.h>
#include <cstring>

namespace torch::lazy {

class TORCH_API BackendData {
 public:
  struct Info {
    /**
     * Used by Lazy Graph Executor to tag info on BackendData objs
     * */
    virtual ~Info() = default;
  };
  /**
   * Represents (Tensor) data stored on a backend device
   * in its native format.
   * */
  using Handle = int64_t;

  BackendData(BackendDevice device, Shape shape)
      : device_(std::move(device)), shape_(std::move(shape)) {}

  virtual ~BackendData() = default;

  const BackendDevice& device() const {
    return device_;
  }

  const Shape& shape() const {
    return shape_;
  }

  Info* info() const {
    return info_.get();
  }

  std::shared_ptr<Info> SetInfo(std::shared_ptr<Info> info) {
    std::swap(info, info_);
    return info;
  }

  virtual Handle GetHandle() = 0;

  virtual void Assign(const BackendData& data) = 0;

  virtual bool HasValue() const = 0;

 private:
  BackendDevice device_;
  Shape shape_;
  std::shared_ptr<Info> info_;
};

using BackendDataPtr = std::shared_ptr<BackendData>;

} // namespace torch::lazy
