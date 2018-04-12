#pragma once

#include <torch/error.h>
#include <torch/nn/module.h>
#include <torch/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace torch { namespace nn {
class Sequential : public CloneableModule<Sequential> {
 public:
  template <typename... Modules>
  explicit Sequential(Modules&&... modules)
      : CloneableModule<Sequential>("Sequential") {
    push_back(std::forward<Modules>(modules)...);
  }

  /// Feeds the `inputs` to the first module, then chains the output of each
  /// module with the input of the next, in order of construction.
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) {
    if (modules_.empty()) {
      return inputs;
    }
    auto iterator = modules_.begin();
    auto intermediate = (**iterator)(inputs);
    for (; iterator != modules_.end(); ++iterator) {
      intermediate = (**iterator)(std::move(intermediate));
    }
    return intermediate;
  }

  /// Adds a new (boxed) `Module` to the `Sequential` container.
  void push_back(const std::shared_ptr<Module>& module_ptr) {
    modules_.push_back(module_ptr);
    const auto index = modules_.size() - 1;
    register_modules({{std::to_string(index), modules_[index]}});
  }

  /// Adds a new `Module` to the `Sequential` container, moving or copying it
  /// into a `shared_ptr` internally. This method allows passing value types,
  /// and letting the container deal with the boxing. This means you can write
  /// `Sequential(Linear(3, 4), ReLU())` instead of
  /// `Sequential(std::make_shared<Linear>(3, 4), std::make_shared<ReLU>())`.
  template <typename M, typename = enable_if_module<M>>
  void push_back(M&& module) {
    // Need to get rid of any reference components for make_unique.
    using Type = typename std::remove_reference<M>::type;
    // Here we move (or copy) the module into a new shared_ptr.
    push_back(std::make_shared<Type>(std::forward<M>(module)));
  }

  /// Accesses the `Module` at the given index.
  Module& operator[](size_t index) {
    AT_ASSERT(index < size(), "Index out of range");
    return *modules_[index];
  }

  /// Accesses the `Module` at the given index.
  const Module& operator[](size_t index) const {
    AT_ASSERT(index < size(), "Index out of range");
    return *modules_[index];
  }

  /// Gets a `shared_ptr` to the `Module` at the given index.
  std::shared_ptr<Module> get(size_t index) const {
    AT_ASSERT(index < size(), "Index out of range");
    return modules_[index];
  }

  /// The current size of the `Sequential` container.
  size_t size() const noexcept {
    return modules_.size();
  }

 private:
  /// Takes a First *and* Second parameter, to avoid ambiguity when a parameter
  /// pack has only one type, in which case the template would be preferred,
  /// even if the other `push_back` functions are better fits (e.g. `unique_ptr`
  /// -> `shared_ptr` overload).
  template <typename First, typename Second, typename... Rest>
  void push_back(First&& first, Second&& second, Rest&&... rest) {
    push_back(std::forward<First>(first));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `push_back()` a final time (above).
    push_back(std::forward<Second>(second), std::forward<Rest>(rest)...);
  }

  /// The base case, when the list of modules is empty.
  void push_back() {}

  std::vector<std::shared_ptr<Module>> modules_;
};
}} // namespace torch::nn
