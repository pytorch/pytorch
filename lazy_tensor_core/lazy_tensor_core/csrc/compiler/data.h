#pragma once

#include <string.h>
#include "lazy_tensors/shape.h"
#include "lazy_tensors/statusor.h"

namespace torch_lazy_tensors {
namespace compiler {

class GenericComputation {
 public:
  virtual int parameters_size() const  = 0;

  virtual const std::vector<lazy_tensors::Shape>& parameter_shapes() const = 0;

  virtual const std::vector<std::string>& parameter_names() const = 0;

  virtual const lazy_tensors::Shape& result_shape() const = 0;

  virtual ~GenericComputation() = default;
};

struct ExecuteOptions {
  bool explode_tuple = true;
};

struct ExecuteComputationOptions : public ExecuteOptions {};

class Computation {
 public:
  Computation(std::shared_ptr<GenericComputation> computation, std::vector<std::string> devices)
      : computation_(std::move(computation)),
        devices_(std::move(devices)) {}

  virtual ~Computation() {}

  GenericComputation* computation() const { return computation_.get(); }

  const std::vector<std::string>& devices() const { return devices_; }

 private:
  std::shared_ptr<GenericComputation> computation_;
  std::vector<std::string> devices_;
};
using ComputationPtr = std::shared_ptr<Computation>;


// TODO(whc)
// what is vector<device> used for here?
// what is compilation_device?
struct CompileInstance {
  CompileInstance() = default;
  CompileInstance(std::shared_ptr<GenericComputation> computation,
                  std::string compilation_device,
                  std::vector<std::string> devices)
      : computation(std::move(computation)),
        compilation_device(std::move(compilation_device)),
        devices(std::move(devices)) {}

  std::shared_ptr<GenericComputation> computation;
  std::string compilation_device;
  std::vector<std::string> devices;
};

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