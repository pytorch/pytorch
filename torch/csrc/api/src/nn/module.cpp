#include <torch/nn/module.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace torch { namespace nn {
std::map<std::string, Variable> Module::parameters() const {
  std::map<std::string, Variable> ret;
  for (auto pair : children_) {
    auto& name = pair.first;
    auto& child = pair.second;
    for (auto& p : child->parameters()) {
      ret[name + "." + p.first] = p.second;
    }
  }
  for (auto pair : params_) {
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
  auto it = container->params_.find(param_name);
  if (it == params_.end()) {
    throw std::runtime_error("No such param: " + param_name);
  }
  return it->second;
}

void Module::cuda() {
  for (auto& pair : children_) {
    pair.second->cuda();
  }
  cuda_ = true;
  auto copied = params_;
  params_.clear();
  initialize_parameters();
  for (auto pair : params_) {
    pair.second.data().copy_(copied[pair.first].data());
  }
}

void Module::cpu() {
  for (auto& pair : children_) {
    pair.second->cpu();
  }
  cuda_ = false;
  auto copied = params_;
  params_.clear();
  initialize_parameters();
  for (auto pair : params_) {
    pair.second.data().copy_(copied[pair.first].data());
  }
}

void Module::train() {
  for (auto& pair : children_) {
    pair.second->train();
  }
  train_ = true;
}

void Module::eval() {
  for (auto& pair : children_) {
    pair.second->eval();
  }
  train_ = false;
}

std::shared_ptr<nn::Module> Module::add(std::shared_ptr<nn::Module> m, std::string const& name) {
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
  if (this->params_.find(name) != this->params_.end()) {
    throw std::runtime_error("Trying to add parameter that already exists");
  }
  if (std::find(name.begin(), name.end(), '.') != name.end()) {
    // We can't allow parameters with dots in their names, as that would make
    // them not findable with parameters().
    throw std::runtime_error("Trying to add parameter with a '.' in its name");
  }
  this->params_[name] = v;
  return this->params_[name];
}

at::Type& Module::DefaultTensor(at::ScalarType s) {
  if (cuda_)
    return at::CUDA(s);
  else
    return at::CPU(s);
}

}} // namespace torch::nn
