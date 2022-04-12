#pragma once

#include <c10/util/irange.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>

#include <vector>

namespace torch {
namespace nn {

/// A list of `Module`s that registers its elements.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::ModuleList mlist(
///     torch::nn::Linear(3, 4),
///     torch::nn::BatchNorm1d(4),
///     torch::nn::Dropout(0.5)
///   );
///
///   for (const auto &module : *mlist) {
///     module->pretty_print(std::cout);
///   }
///
/// \endrst
///
/// Why should you use `ModuleList` instead of a simple `std::vector`? The value
/// a `ModuleList` provides over manually calling a sequence of modules is that
/// it allows treating the whole container *as a single module*, such that
/// performing a transformation on the `ModuleList` applies to each of the
/// modules it stores (which are each a registered submodule of the
/// `ModuleList`). For example, calling
/// `.to(torch::kCUDA)` on a `ModuleList` will move each module in the list to
/// CUDA memory. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::ModuleList mlist(
///     torch::nn::Linear(3, 4),
///     torch::nn::BatchNorm1d(4),
///     torch::nn::Dropout(0.5)
///   );
///
///   // Convert all modules to CUDA.
///   mlist->to(torch::kCUDA);
///
/// \endrst
///
/// Finally, `ModuleList` provides a lightweight container API, such as allowing
/// iteration over submodules, positional access, adding a new module after
/// construction via `push_back`, as well as joining two `ModuleList`s via
/// `extend`.
// NOLINTNEXTLINE(bugprone-exception-escape)
class ModuleListImpl : public Cloneable<ModuleListImpl> {
 public:
  using Iterator = std::vector<std::shared_ptr<Module>>::iterator;
  using ConstIterator = std::vector<std::shared_ptr<Module>>::const_iterator;

  ModuleListImpl() = default;

  /// Constructs the `ModuleList` from a variadic list of modules.
  template <typename... Modules>
  explicit ModuleListImpl(Modules&&... modules) {
    modules_.reserve(sizeof...(Modules));
    push_back_var(std::forward<Modules>(modules)...);
  }

  /// Special cloning function for `ModuleList` because it does not use
  /// `reset()`.
  std::shared_ptr<Module> clone(
      const optional<Device>& device = nullopt) const override {
    auto clone = std::make_shared<ModuleListImpl>();
    for (const auto& module : modules_) {
      clone->push_back(module->clone(device));
    }
    return clone;
  }

  /// `reset()` is empty for `ModuleList`, since it does not have parameters of
  /// its own.
  void reset() override {}

  /// Pretty prints the `ModuleList` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ModuleList";
  }

  void push_back(std::shared_ptr<Module> module) {
    modules_.push_back(std::move(module));
    const auto index = modules_.size() - 1;
    register_module(c10::to_string(index), modules_[index]);
  }

  /// Adds a new `Module` to the `ModuleList` container, moving or copying
  /// it into a `shared_ptr` internally. This method allows passing value types,
  /// and letting the container deal with the boxing.
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  void push_back(M&& module) {
    using Type = typename std::remove_reference<M>::type;
    push_back(std::make_shared<Type>(std::forward<M>(module)));
  }

  /// Unwraps the contained module of a `ModuleHolder` and adds it to the
  /// `ModuleList`.
  template <typename M>
  void push_back(const ModuleHolder<M>& module_holder) {
    push_back(module_holder.ptr());
  }

  /// Iterates over the container and calls `push_back()` on each value.
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& module : container) {
      push_back(module);
    }
  }

  /// Returns an iterator to the start of the `ModuleList`.
  Iterator begin() {
    return modules_.begin();
  }

  /// Returns a const iterator to the start of the `ModuleList`.
  ConstIterator begin() const {
    return modules_.begin();
  }

  /// Returns an iterator to the end of the `ModuleList`.
  Iterator end() {
    return modules_.end();
  }

  /// Returns a const iterator to the end of the `ModuleList`.
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
        "Can only call ModuleList::at with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return *modules_[index]->as<T>();
  }

  /// Attempts to return the module at the given index as the requested type.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  const T& at(size_t index) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call ModuleList::at with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return *modules_[index]->as<T>();
  }

  /// Attempts to return a `std::shared_ptr` whose dynamic type is that of the
  /// underlying module at the given index. Throws an exception if the index is
  /// out of bounds.
  std::shared_ptr<Module> ptr(size_t index) const {
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index];
  }

  /// Attempts to return a `std::shared_ptr` whose type is the one provided.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  std::shared_ptr<T> ptr(size_t index) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call ModuleList::ptr with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return std::dynamic_pointer_cast<T>(modules_[index]);
  }

  /// Like `ptr(index)`.
  std::shared_ptr<Module> operator[](size_t index) const {
    // This is the only method we can call without a type.
    return ptr(index);
  }

  /// The current size of the `ModuleList` container.
  size_t size() const noexcept {
    return modules_.size();
  }

  /// True if there are no modules in the `ModuleList`.
  bool is_empty() const noexcept {
    return size() == 0;
  }

  void insert(size_t index, std::shared_ptr<Module> module) {
    TORCH_CHECK(index <= size(), "Index out of range");

    if (index == size())
      push_back(module);
    else {
      modules_.insert(
          modules_.begin() + Iterator::difference_type(index),
          std::move(module));

      for (const auto i : c10::irange(index, size() - 1)) {
        (void)i; // Suppress unused variable warning
        replace_module(c10::to_string(index), modules_[index]);
      }
      register_module(c10::to_string(size() - 1), modules_.back());
    }
  }

  /// Unwraps the contained module of a `ModuleHolder` and inserts it in the
  /// `ModuleList`.
  template <typename M>
  void insert(size_t index, const ModuleHolder<M>& module_holder) {
    insert(index, module_holder.ptr());
  }

  /// inserts a new `Module` to the `ModuleList` container, moving or copying
  /// it into a `shared_ptr` internally. This method allows passing value types,
  /// and letting the container deal with the boxing.
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  void insert(size_t index, M&& module) {
    using Type = typename std::remove_reference<M>::type;
    insert(index, std::make_shared<Type>(std::forward<M>(module)));
  }

 private:
  template <typename Head, typename... Tail>
  void push_back_var(Head&& head, Tail&&... tail) {
    push_back(std::forward<Head>(head));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `push_back()` a final time (above).
    push_back_var(std::forward<Tail>(tail)...);
  }

  /// The base case, when the list of modules is empty.
  void push_back_var() {}

  // Box the AnyModules to give ModuleList reference semantics, like the rest of
  // the API. Note that this is not required otherwise, this could just be a
  // `vector<AnyModule>`.
  std::vector<std::shared_ptr<Module>> modules_;
};

/// A `ModuleHolder` subclass for `ModuleListImpl`.
/// See the documentation for `ModuleListImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ModuleList);

} // namespace nn
} // namespace torch
