#pragma once

#include <string.h>
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensors/shape.h"

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

  BackendData(const Device& device, const lazy_tensors::Shape& shape)
      : device_(device), shape_(shape) {}

  virtual ~BackendData() {}

  const Device& device() const { return device_; }

  const lazy_tensors::Shape& shape() const { return shape_; }

  Info* info() const { return info_.get(); }

  std::shared_ptr<Info> SetInfo(std::shared_ptr<Info> info) {
    std::swap(info, info_);
    return info;
  }

  virtual Handle GetOpaqueHandle() = 0;

  virtual void Assign(const BackendData& data) = 0;

  virtual bool HasValue() const = 0;

 private:
  Device device_;
  lazy_tensors::Shape shape_;
  std::shared_ptr<Info> info_;
};

using BackendDataPtr = std::shared_ptr<BackendData>;

}
}  // namespace torch_lazy_tensors
