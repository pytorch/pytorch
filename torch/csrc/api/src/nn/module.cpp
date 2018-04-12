#include <torch/nn/module.h>
#include <torch/detail/ordered_dict.h>
#include <torch/nn/cursor.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch { namespace nn {

Module::Module(std::string name)
    : name_(std::move(name)), is_training_(false) {}

Module::~Module() = default;

std::unique_ptr<Module> Module::clone() {
  TORCH_ERROR(
      "clone() has not been implemented for %s. "
      "Use the copy constructor if you don't require polymorphic cloning. "
      "Otherwise, subclass CloneableModule<%s> to inherit cloning behavior.",
      name_.c_str(),
      name_.c_str());
}

std::vector<Tensor> Module::operator()(const std::vector<Tensor>& inputs) {
  return forward(inputs);
}

// Train/Eval mode
void Module::train() {
  is_training_ = true;
}

void Module::eval() {
  is_training_ = false;
}

bool Module::is_training() const noexcept {
  return is_training_;
}

// Recursive Transformations
void Module::cpu() {
  // parameters().apply([](Tensor& tensor) { tensor.toBackend_(kCPU); });
}

void Module::cuda() {
  // parameters().apply([](Tensor& tensor) { tensor.toBackend_(kCUDA); });
}

void Module::type(at::ScalarType new_type) {
  // parameters().apply([=](Tensor& tensor) { tensor.toType_(new_type); });
  // buffers().apply([=](Tensor& tensor) { tensor.toType_(new_type); });
}

void Module::zero_grad() {
  parameters().apply([](Tensor& tensor) {
    // Temporary!!! Downcast should not be necessary...
    auto& variable = as_variable_ref(tensor);
    if (variable.requires_grad()) {
      variable.grad().detach_();
      variable.grad().zero_();
    }
  });
}

// Recursive Accessors
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

// Serialization/Deserialization
void Module::serialize(Archive& archive) {}
void Module::deserialize(Archive&& archive) {}

const std::string& Module::name() const noexcept {
  return name_;
}

void Module::register_parameters(detail::OrderedDict<Tensor>&& parameters) {
  parameters_.update(std::move(parameters));
}

void Module::register_buffers(detail::OrderedDict<Tensor>&& buffers) {
  buffers_.update(std::move(buffers));
}

void Module::register_modules(
    detail::OrderedDict<std::shared_ptr<Module>>&& modules) {
  children_.update(std::move(modules));
}

}} // namespace torch::nn
