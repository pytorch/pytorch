#include <torch/nn/module.h>

#include <torch/csrc/autograd/generated/VariableType.h>

#include <ATen/Error.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>

namespace torch { namespace nn {

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

std::unique_ptr<Module> Module::clone() const {
  AT_ERROR(
      "clone() has not been implemented for ",
      name(),
      ". Use the copy constructor if you don't require polymorphic cloning. "
      "Otherwise, subclass torch::nn::CloneableModule<",
      name(),
      "> instead of torch::nn::Module to inherit the ability to clone.");
}

std::map<std::string, Variable> Module::parameters() const {
  std::map<std::string, Variable> ret;
  for (auto pair : children_) {
    auto& name = pair.first;
    auto& child = pair.second;
    for (auto& p : child->parameters()) {
      ret[name + "." + p.first] = p.second;
    }
  }
  for (auto pair : parameters_) {
    ret[pair.first] = pair.second;
  }
  return ret;
}

Variable& Module::param(std::string const& name) {
  Module* container = this;
  auto begin = 0;
  while (true) {
    auto dot_pos = name.find('.', begin);
    if (dot_pos == std::string::npos) {
      break;
    }

    auto child_name = name.substr(begin, dot_pos - begin);
    auto it = container->children_.find(child_name);
    if (it == container->children_.end()) {
      throw std::runtime_error("No such child: " + child_name);
    }

    container = it->second.get();
    begin = dot_pos + 1; // Skip the dot
  }

  auto param_name = name.substr(begin);
  auto it = container->parameters_.find(param_name);
  if (it == parameters_.end()) {
    throw std::runtime_error("No such param: " + param_name);
  }
  return it->second;
}

void Module::train() {
  for (auto& pair : children_) {
    pair.second->train();
  }
  is_training_ = true;
}

void Module::eval() {
  for (auto& pair : children_) {
    pair.second->eval();
  }
  is_training_ = false;
}

void Module::cuda() {
  to(at::kCUDA);
}

void Module::cpu() {
  to(at::kCPU);
}

void Module::to(at::Type& type) {
  for (auto& child : children_) {
    child.second->to(type);
  }
  for (auto& pair : parameters_) {
    auto& parameter = pair.second;
    at::detail::set_data(parameter, parameter.data().toType(type));
    AT_ASSERT(parameter.data().type() == type);
    AT_ASSERT(&parameter.type() == autograd::VariableType::getType(type));
  }
}

void Module::to(at::ScalarType scalar_type) {
  for (auto& child : children_) {
    child.second->to(scalar_type);
  }
  for (auto& pair : parameters_) {
    auto& parameter = pair.second;
    auto& new_type = parameter.data().type().toScalarType(scalar_type);
    at::detail::set_data(parameter, parameter.data().toType(new_type));
    AT_ASSERT(parameter.data().type().scalarType() == scalar_type);
    AT_ASSERT(parameter.type().scalarType() == scalar_type);
  }
}

void Module::to(at::Backend backend) {
  for (auto& child : children_) {
    child.second->to(backend);
  }
  for (auto& pair : parameters_) {
    auto& parameter = pair.second;
    auto& new_type = parameter.data().type().toBackend(backend);
    at::detail::set_data(parameter, parameter.data().toType(new_type));
    AT_ASSERT(parameter.data().type().backend() == backend);
    AT_ASSERT(parameter.type().backend() == backend);
  }
}

bool Module::is_training() const noexcept {
  return is_training_;
}

void Module::zero_grad() {
  for (auto& child : children_) {
    child.second->zero_grad();
  }
  for (auto& pair : parameters_) {
    pair.second.grad().zero_();
  }
}

std::shared_ptr<nn::Module> Module::add(
    std::shared_ptr<nn::Module> m,
    std::string const& name) {
  if (this->children_.find(name) != this->children_.end()) {
    throw std::runtime_error("Trying to add container that already exists");
  }
  if (std::find(name.begin(), name.end(), '.') != name.end()) {
    // We can't allow containers with dots in their names, as that would make
    // their parameters not findable with parameters().
    throw std::runtime_error("Trying to add parameter with a '.' in its name");
  }
  this->children_[name] = std::move(m);
  return this->children_[name];
}

Variable& Module::add(Variable v, std::string const& name) {
  if (this->parameters_.find(name) != this->parameters_.end()) {
    throw std::runtime_error("Trying to add parameter that already exists");
  }
  if (std::find(name.begin(), name.end(), '.') != name.end()) {
    // We can't allow parameters with dots in their names, as that would make
    // them not findable with parameters().
    throw std::runtime_error("Trying to add parameter with a '.' in its name");
  }
  this->parameters_[name] = v;
  return this->parameters_[name];
}
}} // namespace torch::nn
