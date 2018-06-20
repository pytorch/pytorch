#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace torch {

// Lets you use a container without making a new class,
// for experimental implementations
class SimpleContainer : public nn::Cloneable<SimpleContainer> {
 public:
  virtual std::vector<Variable> forward(std::vector<Variable>) {
    throw std::runtime_error(
        "SimpleContainer has no forward, maybe you"
        " wanted to subclass and override this function?");
  }

  void reset() override {}

  template <typename ModuleHolder>
  ModuleHolder add(
      ModuleHolder module_holder,
      std::string name = std::string()) {
    return Module::register_module(std::move(name), module_holder);
  }
};
} // namespace torch
