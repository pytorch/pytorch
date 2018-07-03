#pragma once

#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/any.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/Error.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace nn {
/// A `Sequential` module is a container for any number of other modules. Its
/// `forward()` method chains outputs to inputs and returns the final output.
/// The `Sequential` class reference semantics.
class Sequential : public Cloneable<Sequential> {
 public:
  using Iterator = std::vector<std::shared_ptr<AnyModule>>::iterator;
  using ConstIterator = std::vector<std::shared_ptr<AnyModule>>::const_iterator;

  /// Constructs the `Sequential` from a pack of modules. Each module can either
  /// be a plain value (e.g. `Linear`) or a boxed value (e.g.
  /// `shared_ptr<Linear>`). Unboxed modules will be moved into `shared_ptr`s
  /// internally.
  template <
      typename... Modules,
      typename = disable_if_contains_t<Sequential, Modules...>>
  explicit Sequential(Modules&&... modules) {
    modules_.reserve(sizeof...(Modules));
    push_back(std::forward<Modules>(modules)...);
  }

  /// reset() is empty for `Sequential`, since it does not have parameter of its
  /// own.
  void reset() override {}

  /// Feeds the `inputs` to the first module, then chains the output of each
  /// module with the input of the next, in order of construction.
  template <typename ReturnType = Tensor, typename... ArgumentTypes>
  ReturnType forward(ArgumentTypes&&... arguments) {
    AT_CHECK(!is_empty(), "Cannot call forward() on an empty Sequential");

    auto iterator = modules_.begin();
    auto input =
        (*iterator)->forward(std::forward<ArgumentTypes>(arguments)...);

    for (++iterator; iterator != modules_.end(); ++iterator) {
      input = (*iterator)->forward(std::move(input));
    }

    // Check the return value and give a nice error message if the requsted
    // return type was incorrect.
    if (auto* return_value = input.template try_get<ReturnType>()) {
      return std::move(*return_value);
    }
    AT_ERROR(
        "The type of the return value is ",
        at::demangle(input.type_info().name()),
        ", but you asked for type ",
        at::demangle(typeid(ReturnType).name()));
  }

  /// Adds a new (boxed) `Module` to the `Sequential` container.
  template <typename ModuleType>
  void push_back(std::shared_ptr<ModuleType> module_ptr) {
    // Nesting Sequential doesn't work because `forward()`'s return type is
    // templatized, so it'll give a nasty compiler error.
    static_assert(
        !std::is_same<Sequential, ModuleType>::value,
        "Sequential is not nestable");
    static_assert(
        torch::detail::is_module<ModuleType>::value,
        "Can only add objects derived from nn::Module to Sequential");
    static_assert(
        torch::detail::has_forward<ModuleType>::value,
        "Can only add modules with a forward() method to Sequential");
    push_back(std::make_shared<AnyModule>(std::move(module_ptr)));
  }

  /// Adds a new `Module` to the `Sequential` container, moving or copying it
  /// into a `shared_ptr` internally. This method allows passing value types,
  /// and letting the container deal with the boxing. This means you can write
  /// `Sequential(Module(3, 4))` instead of
  /// `Sequential(std::make_shared<Module>(3, 4))`.
  template <typename M, typename = torch::detail::disable_if_module_holder_t<M>>
  void push_back(M&& module) {
    // Need to get rid of any reference components for make_unique.
    using Type = typename std::remove_reference<M>::type;
    // Here we move (or copy) the module into a new shared_ptr.
    push_back(std::make_shared<Type>(std::forward<M>(module)));
  }

  /// Unwraps the contained module of a `ModuleHolder` and adds it to the
  /// `Sequential`.
  template <typename M>
  void push_back(const ModuleHolder<M>& module_holder) {
    push_back(module_holder.ptr());
  }

  /// Adds a type-erased `AnyModule` to the `Sequential`.
  void push_back(std::shared_ptr<AnyModule> any_module) {
    modules_.push_back(std::move(any_module));
    const auto index = modules_.size() - 1;
    register_module(std::to_string(index), modules_[index]->ptr());
  }

  /// Iterates over the container and calls `push_back()` on each iterated
  /// value.
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& module : container) {
      push_back(module);
    }
  }

  /// Returns an iterator to the start of the `Sequential`.
  Iterator begin() {
    return modules_.begin();
  }
  ConstIterator begin() const {
    return modules_.begin();
  }

  /// Returns an iterator to the end of the `Sequential`.
  Iterator end() {
    return modules_.end();
  }
  ConstIterator end() const {
    return modules_.end();
  }

  /// Attempts to return the module at the given index as the requested type.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  T& at(size_t index) {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call Sequential::at with an nn::Module type");
    AT_CHECK(index < size(), "Index out of range");
    return modules_[index]->get<T>();
  }

  /// Attempts to return the module at the given index as the requested type.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  const T& at(size_t index) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call Sequential::at with an nn::Module type");
    AT_CHECK(index < size(), "Index out of range");
    return modules_[index]->get<T>();
  }

  /// Attempts to return a `std::shared_ptr` whose dynamic type is that of the
  /// underlying module at the given index. Throws an exception if the index is
  /// out of bounds.
  std::shared_ptr<Module> ptr(size_t index) const {
    AT_CHECK(index < size(), "Index out of range");
    return modules_[index]->ptr();
  }

  /// Attempts to return a `std::shared_ptr` whose type is the one provided.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  std::shared_ptr<T> ptr(size_t index) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call Sequential::ptr with an nn::Module type");
    AT_CHECK(index < size(), "Index out of range");
    return modules_[index]->ptr<T>();
  }

  /// Like `ptr(index)`.
  std::shared_ptr<Module> operator[](size_t index) const {
    // This is the only method we can call without a type.
    return ptr(index);
  }

  /// The current size of the `Sequential` container.
  size_t size() const noexcept {
    return modules_.size();
  }

  /// True if there are no modules in the `Sequential`.
  bool is_empty() const noexcept {
    return size() == 0;
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

  // Box the AnyModules to give Sequential reference semantics, like the rest of
  // the API. Note that this is not required otherwise, this could just be a
  // `vector<AnyModule>`.
  std::vector<std::shared_ptr<AnyModule>> modules_;
};
} // namespace nn
} // namespace torch
