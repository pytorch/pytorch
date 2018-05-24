#pragma once

#include <torch/detail.h>
#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

namespace torch {
namespace nn {

template <class Derived>
class ContainerListImpl : public Module {
  // Lets you use a container like a vector without making a new class,
  // just for simple implementations
 public:
  virtual variable_list forward(variable_list) override {
    throw std::runtime_error(
        "ContainerList has no forward, maybe you"
        " wanted to subclass and override this function?");
  }

  std::shared_ptr<Module> add(std::shared_ptr<Module> m) {
    return append(m).modules_.back();
  }

  ContainerListImpl<Derived>& append(std::shared_ptr<Module> m) {
    modules_.push_back(
        register_module(std::to_string(size() - 1), std::move(m)));
    return *this;
  }

  std::shared_ptr<Module>& operator[](int index) {
    return modules_[index];
  }

  int size() {
    return modules_.size();
  }

  std::vector<std::shared_ptr<Module>>::iterator begin() {
    return modules_.begin();
  }

  std::vector<std::shared_ptr<Module>>::iterator end() {
    return modules_.end();
  }

  std::vector<std::shared_ptr<Module>> modules_;
};

class ContainerList : public ContainerListImpl<ContainerList> {};

class Sequential : public ContainerListImpl<Sequential> {
  // Mimics nn.Sequential from pytorch.
 public:
  variable_list forward(variable_list input) override {
    for (auto& container : modules_) {
      input = container->forward(input);
    }
    return input;
  }

  std::shared_ptr<Module> add(
      std::shared_ptr<Module> m,
      std::string name = "") {
    return append(m, name).modules_.back();
  }

  Sequential& append(std::shared_ptr<Module> m, std::string name = "") {
    if (name == "") {
      name = std::to_string(size());
    }
    modules_.push_back(register_module(name, std::move(m)));
    return *this;
  }
};
} // namespace nn
} // namespace torch
