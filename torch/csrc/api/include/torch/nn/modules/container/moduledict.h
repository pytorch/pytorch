#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/ordered_dict.h>
#include <vector>

namespace torch::nn {

/// An OrderedDict of `Module`s that registers its elements by their `key`s.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
///     {"linear", Linear(10, 3).ptr()},
///     {"conv", Conv2d(1, 2, 3).ptr()},
///     {"dropout", Dropout(0.5).ptr()},
///   };
///   torch::nn::ModuleDict dict1(ordereddict);
///
///   for (const auto &module : *dict1) {
///     module->pretty_print(std::cout);
///   }
///
///   std::vector<std::pair<std::string, std::shared_ptr<Module>>> list = {
///     {"linear", Linear(10, 3).ptr()},
///     {"conv", Conv2d(1, 2, 3).ptr()},
///     {"dropout", Dropout(0.5).ptr()},
///   };
///   torch::nn::ModuleDict dict2(list);
///
///   for (const auto &module : *dict2) {
///     module->pretty_print(std::cout);
///   }
///
/// \endrst
///
/// Why should you use `ModuleDict` instead of a simple `map` or `OrderedDict`?
/// The value a `ModuleDict` provides over manually calling an ordered map of
/// modules is that it allows treating the whole container *as a single module*,
/// such that performing a transformation on the `ModuleDict` applies to each of
/// the modules it stores (which are each a registered submodule of the
/// `ModuleDict`). For example, calling `.to(torch::kCUDA)` on a `ModuleDict`
/// will move each module in the map to CUDA memory. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
///     {"linear", Linear(10, 3).ptr()},
///     {"conv", Conv2d(1, 2, 3).ptr()},
///     {"dropout", Dropout(0.5).ptr()},
///   };
///   torch::nn::ModuleDict dict(ordereddict);
///
///   // Convert all modules to CUDA.
///   dict->to(torch::kCUDA);
///
/// \endrst
///
/// Finally, `ModuleDict` provides a lightweight container API, such as allowing
/// iteration over submodules, positional access, adding new modules from a
/// vector of key-module pairs or an `OrderedDict` or another `ModuleDict` after
/// construction via `update`.
class ModuleDictImpl : public Cloneable<ModuleDictImpl> {
 public:
  using Iterator =
      torch::OrderedDict<std::string, std::shared_ptr<Module>>::Iterator;
  using ConstIterator =
      torch::OrderedDict<std::string, std::shared_ptr<Module>>::ConstIterator;

  ModuleDictImpl() = default;

  /// Constructs the `ModuleDict` from a list of string-Module pairs.
  explicit ModuleDictImpl(
      const std::vector<std::pair<std::string, std::shared_ptr<Module>>>&
          modules) {
    update(modules);
  }

  /// Constructs the `ModuleDict` from an `OrderedDict`.
  explicit ModuleDictImpl(
      const torch::OrderedDict<std::string, std::shared_ptr<Module>>& modules) {
    update(modules);
  }

  /// Return the items in the `ModuleDict`.
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> items() const {
    return modules_.pairs();
  }

  /// Return the keys in the `ModuleDict`.
  std::vector<std::string> keys() const {
    return modules_.keys();
  }

  /// Return the values in the `ModuleDict`.
  std::vector<std::shared_ptr<Module>> values() const {
    return modules_.values();
  }

  /// Return an iterator to the start of `ModuleDict`.
  Iterator begin() {
    return modules_.begin();
  }

  /// Return a const iterator to the start of `ModuleDict`.
  ConstIterator begin() const {
    return modules_.begin();
  }

  /// Return an iterator to the end of `ModuleDict`.
  Iterator end() {
    return modules_.end();
  }

  /// Return a const iterator to the end of `ModuleDict`.
  ConstIterator end() const {
    return modules_.end();
  }

  /// Return the number of items currently stored in the `ModuleDict`.
  size_t size() const noexcept {
    return modules_.size();
  }

  /// Return true if the `ModuleDict` is empty, otherwise return false.
  bool empty() const noexcept {
    return modules_.is_empty();
  }

  /// Check if the centain parameter with the key in the `ModuleDict`.
  bool contains(const std::string& key) const noexcept {
    return modules_.contains(key);
  }

  /// Remove all items from the `ModuleDict`.
  void clear() {
    // Not remove the registration of modules to make it consistent with python
    // version.
    modules_.clear();
  }

  /// Special cloning function for `ModuleDict` because it does not use
  /// `reset()`.
  std::shared_ptr<Module> clone(
      const std::optional<Device>& device = std::nullopt) const override {
    auto clone = std::make_shared<ModuleDictImpl>();
    for (const auto& module : modules_) {
      clone->insert(module.key(), module.value()->clone(device));
    }
    return clone;
  }

  /// `reset()` is empty for `ModuleDict`, since it does not have parameters of
  /// its own.
  void reset() override {}

  /// Pretty prints the `ModuleDict` into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ModuleDict";
  }

  /// Attempts to returns the `Module` associated with the given `key`. Throws
  /// an exception if no such `key` is stored in the `ModuleDict`. Check
  /// contains(key) before for a non-throwing way of access.
  std::shared_ptr<Module> operator[](const std::string& key) const {
    return modules_[key];
  }

  /// Attempts to return the module at the given key as the requested type.
  /// Throws an exception if no such `key` is stored in the `ModuleDict`.
  /// Check contains(key) before for a non-throwing way of access.
  template <typename T>
  T& at(const std::string& key) {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call ModuleList::at with an nn::Module type");
    auto module = modules_[key]->as<T>();
    TORCH_CHECK(
        module,
        "Unable to cast module[",
        key,
        "] to ",
        c10::demangle(typeid(T).name()));
    return *module;
  }

  /// Attempts to return the module at the given key as the requested type.
  /// Throws an exception if no such `key` is stored in the `ModuleDict`.
  /// Check contains(key) before for a non-throwing way of access.
  template <typename T>
  const T& at(const std::string& key) const {
    static_assert(
        torch::detail::is_module<T>::value,
        "Can only call ModuleList::at with an nn::Module type");
    const auto module = modules_[key]->as<T>();
    TORCH_CHECK(
        module,
        "Unable to cast module[",
        key,
        "] to ",
        c10::demangle(typeid(T).name()));
    return *module;
  }

  /// Removes and returns the `Module` associated with the given `key`.
  /// Throws an exception if no such `key` is stored in the `ModuleDict`.
  /// Check contains(key) before for a non-throwing way of access.
  std::shared_ptr<Module> pop(const std::string& key) {
    auto module = modules_[key];
    modules_.erase(key);
    // Not remove the registration of the module to make it consistent with
    // python version.
    return module;
  }

  /// Updated the `ModuleDict` with a vector of key-module pairs.
  void update(
      const std::vector<std::pair<std::string, std::shared_ptr<Module>>>&
          modules) {
    for (auto& item : modules) {
      insert(item.first, item.second);
    }
  }

  /// Updated the `ModuleDict` with key-value pairs from `OrderedDict` or
  /// `ModuleDict`.
  template <typename Container>
  void update(const Container& container) {
    for (auto& item : container) {
      insert(item.key(), item.value());
    }
  }

 private:
  /// Private `OrderedDict` holding the key-Module pairs.
  torch::OrderedDict<std::string, std::shared_ptr<Module>> modules_;

  /// Insert a key-module pair by overwriting existing keys,
  /// and register or replace the `Module`.
  void insert(const std::string& key, std::shared_ptr<Module> module) {
    if (contains(key)) {
      modules_[key] = std::move(module);
      replace_module(key, modules_[key]);
    } else {
      modules_.insert(key, std::move(module));
      register_module(key, modules_.back().value());
    }
  }
};

/// A `ModuleHolder` subclass for `ModuleDictImpl`.
/// See the documentation for `ModuleDictImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ModuleDict);

} // namespace torch::nn
