#include <torch/nn/module.h>

#include <torch/nn/cursor.h>

#include <torch/csrc/autograd/generated/VariableType.h>

#include <ATen/Error.h>

#include <algorithm>
#include <map>
#include <string>
#include <typeinfo>
#include <unordered_map>

namespace torch {
namespace nn {

Module::Module(std::string name) : name_(std::move(name)) {}

const std::string& Module::name() const noexcept {
  // If the name optional is empty at this point, we grab the name of the
  // dynamic type via RTTI. Note that we cannot do this in the constructor,
  // because in the constructor of a base class `this` always refers to the base
  // type. Inheritance effectively does not work in constructors. Also this note
  // from http://en.cppreference.com/w/cpp/language/typeid:
  // If typeid is used on an object under construction or destruction (in a
  // destructor or in a constructor, including constructor's initializer list
  // or default member initializers), then the std::type_info object referred
  // to by this typeid represents the class that is being constructed or
  // destroyed even if it is not the most-derived class.
  if (!name_.has_value()) {
    name_ = at::demangle(typeid(*this).name());
  }
  return *name_;
}

std::shared_ptr<Module> Module::clone() const {
  AT_ERROR(
      "clone() has not been implemented for ",
      name(),
      ". Use the copy constructor if you don't require polymorphic cloning. "
      "Otherwise, subclass torch::nn::Cloneable<",
      name(),
      "> instead of torch::nn::Module to inherit the ability to clone.");
}

ModuleCursor Module::modules() {
  return ModuleCursor(*this);
}

ConstModuleCursor Module::modules() const {
  return ConstModuleCursor(*this);
}

ModuleCursor Module::children() {
  return ModuleCursor(*this, /*maximum_depth=*/1);
}

ConstModuleCursor Module::children() const {
  return ConstModuleCursor(*this, /*maximum_depth=*/1);
}

ParameterCursor Module::parameters() {
  return ParameterCursor(*this);
}

ConstParameterCursor Module::parameters() const {
  return ConstParameterCursor(*this);
}

BufferCursor Module::buffers() {
  return BufferCursor(*this);
}

ConstBufferCursor Module::buffers() const {
  return ConstBufferCursor(*this);
}

void Module::train() {
  for (auto& child : children_) {
    child.value->train();
  }
  is_training_ = true;
}

void Module::eval() {
  for (auto& child : children_) {
    child.value->eval();
  }
  is_training_ = false;
}

void Module::cpu(bool non_blocking) {
  to(at::Device(at::Device::Type::CPU), non_blocking);
}

void Module::cuda(int32_t device_index, bool non_blocking) {
  to(at::Device(at::Device::Type::CUDA, device_index), non_blocking);
}

void Module::to(at::Device device, at::ScalarType dtype, bool non_blocking) {
  to(device, dtype, non_blocking);
}

void Module::to(at::ScalarType dtype, bool non_blocking) {
  to(dtype, non_blocking);
}

void Module::to(at::Device device, bool non_blocking) {
  to(device, non_blocking);
}

bool Module::is_training() const noexcept {
  return is_training_;
}

void Module::zero_grad() {
  for (auto& child : children_) {
    child.value->zero_grad();
  }
  for (auto& parameter : parameters_) {
    parameter->grad().zero_();
  }
}

Tensor& Module::register_parameter(
    std::string name,
    Tensor tensor,
    bool requires_grad) {
  tensor.set_requires_grad(requires_grad);
  return parameters_.insert(std::move(name), std::move(tensor));
}

Tensor& Module::register_buffer(std::string name, Tensor tensor) {
  return buffers_.insert(std::move(name), std::move(tensor));
}

void Module::clone_(Module& other) {}
} // namespace nn
} // namespace torch
