#pragma once

#include <torch/nn/module.h>
#include <torch/tensor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace torch { namespace nn {
namespace detail {
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace detail

class Sequential : public CloneableModule<Sequential> {
 public:
  template <typename... Modules>
  explicit Sequential(Modules&&... modules)
      : CloneableModule<Sequential>("Sequential") {
    static_assert(sizeof...(Modules) > 0, "Sequential must not be empty");
    append(std::forward<Modules>(modules)...);
  }

  /// Feeds the `inputs` to the first module, then chains the output of each
  /// module with the input of the next, in order of construction.
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) {
    if (modules_.empty()) {
      return {};
    }
    auto iterator = modules_.begin();
    auto intermediate = (**iterator)(inputs);
    for (; iterator != modules_.end(); ++iterator) {
      intermediate = (**iterator)(std::move(intermediate));
    }
    return intermediate;
  }

  /// Adds a new `Module` to the `Sequential` container.
  template <typename M>
  void append(M&& module) {
    static_assert(is_module<M>::value, "Sequential can only hold Modules");
    // Need to get rid of any reference components for make_unique.
    using Type = typename std::remove_reference<M>::type;
    // Here we copy the module into a new unique_ptr.
    modules_.push_back(detail::make_unique<Type>(std::forward<Type>(module)));
    const size_t index = modules_.size() - 1;
    // Since we allocated the module on the heap, the pointer to the module in
    // the base class will always be valid.
    register_modules({{std::to_string(index), modules_.back().get()}});
  }

  /// The current size of the `Sequential` container.
  size_t size() const noexcept {
    return modules_.size();
  }

 private:
  template <typename HeadType, typename... Tail>
  void append(HeadType&& head, Tail&&... tail) {
    append(std::forward<HeadType>(head));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `append()` a final time.
    append(std::forward<Tail>(tail)...);
  }

  std::vector<std::unique_ptr<Module>> modules_;
};
}} // namespace torch::nn
