#pragma once

#include <string.h>
#include "lazy_tensors/shape.h"

namespace torch_lazy_tensors {
namespace compiler {

// TODO(whc) rename this to BackendTensorData or something?  BackendDataHandle?
class Data {
 public:
  struct Info {
    virtual ~Info() {}
  };

  using OpaqueHandle = int64_t;

  Data(std::string device, lazy_tensors::Shape shape)
      : device_(std::move(device)), shape_(std::move(shape)) {}

  virtual ~Data() {}

  const std::string& device() const { return device_; }

  const lazy_tensors::Shape& shape() const { return shape_; }

  Info* info() const { return info_.get(); }

  std::shared_ptr<Info> SetInfo(std::shared_ptr<Info> info) {
    std::swap(info, info_);
    return info;
  }

  virtual OpaqueHandle GetOpaqueHandle() = 0;

  virtual void Assign(const Data& data) = 0;

  virtual bool HasValue() const = 0;

 private:
  std::string device_;
  lazy_tensors::Shape shape_;
  std::shared_ptr<Info> info_;
};

using DataPtr = std::shared_ptr<Data>;

}
}  // namespace torch_lazy_tensors