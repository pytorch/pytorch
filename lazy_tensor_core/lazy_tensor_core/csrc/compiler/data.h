#pragma once

#include <string.h>
#include "lazy_tensors/shape.h"
#include "lazy_tensors/statusor.h"

namespace torch_lazy_tensors {
namespace compiler {

class ProgramShape {
 public:
  ProgramShape(std::vector<lazy_tensors::Shape> parameters,
               std::vector<std::string> parameter_names, lazy_tensors::Shape result)
      : parameters_(std::move(parameters)),
        parameter_names_(std::move(parameter_names)),
        result_(std::move(result)) {
    CHECK_EQ(parameters_.size(), parameter_names_.size());
  }

  int parameters_size() const { return parameters_.size(); }

  const std::vector<lazy_tensors::Shape>& parameters() const { return parameters_; }

  const std::vector<std::string>& parameter_names() const {
    return parameter_names_;
  }

  const lazy_tensors::Shape& result() const { return result_; }

 private:
  std::vector<lazy_tensors::Shape> parameters_;
  std::vector<std::string> parameter_names_;
  lazy_tensors::Shape result_;
};

class GenericComputation {
 public:
  virtual lazy_tensors::StatusOr<ProgramShape> GetProgramShape() const = 0;

  virtual ~GenericComputation() = default;
};

struct ExecuteOptions {
  bool explode_tuple = true;
};

struct ExecuteComputationOptions : public ExecuteOptions {};

class Computation {
 public:
  Computation(std::shared_ptr<GenericComputation> computation,
              ProgramShape program_shape, std::vector<std::string> devices)
      : computation_(std::move(computation)),
        program_shape_(std::move(program_shape)),
        devices_(std::move(devices)) {}

  virtual ~Computation() {}

  GenericComputation* computation() const { return computation_.get(); }

  const ProgramShape& program_shape() const { return program_shape_; }

  const std::vector<std::string>& devices() const { return devices_; }

 private:
  std::shared_ptr<GenericComputation> computation_;
  ProgramShape program_shape_;
  std::vector<std::string> devices_;
};
using ComputationPtr = std::shared_ptr<Computation>;

struct CompileInstance {
  CompileInstance() = default;
  CompileInstance(std::shared_ptr<GenericComputation> computation,
                  std::string compilation_device,
                  std::vector<std::string> devices, const lazy_tensors::Shape* output_shape)
      : computation(std::move(computation)),
        compilation_device(std::move(compilation_device)),
        devices(std::move(devices)),
        output_shape(output_shape) {}

  std::shared_ptr<GenericComputation> computation;
  std::string compilation_device;
  std::vector<std::string> devices;
  const lazy_tensors::Shape* output_shape = nullptr;
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