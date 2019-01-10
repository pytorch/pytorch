#pragma once

#include <torch/detail.h>
#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

namespace torch { namespace nn {

template <class Derived>
class ContainerListImpl : public CloneableModule<Derived> {
  // Lets you use a container like a vector without making a new class,
  // just for simple implementations
 public:
  virtual variable_list forward(variable_list) override {
    throw std::runtime_error(
        "ContainerList has no forward, maybe you"
        " wanted to subclass and override this function?");
  }

  std::shared_ptr<Module> add(std::shared_ptr<Module> m) {
    return append(m).children_.back();
  }

  ContainerListImpl<Derived>& append(std::shared_ptr<Module> m) {
    children_.push_back(m);
    Module::add(children_.back(), std::to_string(size() - 1));
    return *this;
  }

  std::shared_ptr<Module>& operator[](int index) {
    return children_[index];
  }

  int size() {
    return children_.size();
  }

  std::vector<std::shared_ptr<Module>>::iterator begin() {
    return children_.begin();
  }

  std::vector<std::shared_ptr<Module>>::iterator end() {
    return children_.end();
  }

  std::vector<std::shared_ptr<Module>> children_;
};

class ContainerList : public ContainerListImpl<ContainerList> {};

class Sequential : public ContainerListImpl<Sequential> {
  // Mimics nn.Sequential from pytorch.
 public:
  variable_list forward(variable_list input) override {
    for (auto& container : children_) {
      input = container->forward(input);
    }
    return input;
  }

  std::shared_ptr<Module> add(
      std::shared_ptr<Module> m,
      std::string name = "") {
    return append(m, name).children_.back();
  }

  Sequential& append(std::shared_ptr<Module> m, std::string name = "") {
    if (name == "") {
      name = std::to_string(size());
    }
    children_.push_back(m);
    Module::add(children_.back(), name);
    return *this;
  }
};

class SimpleContainer : public CloneableModule<SimpleContainer> {
  // Lets you use a container without making a new class,
  // for experimental implementations
 public:
  virtual variable_list forward(variable_list) override {
    throw std::runtime_error(
        "SimpleContainer has no forward, maybe you"
        " wanted to subclass and override this function?");
  }
  using Module::add;
};
}} // namespace torch::nn
