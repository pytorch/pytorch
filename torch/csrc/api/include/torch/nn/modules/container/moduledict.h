#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/ordered_dict.h>
#include <vector>

namespace torch {
namespace nn {

class ModuleDictImpl : public Cloneable<ModuleDictImpl> {
 public:
  using Iterator = torch::OrderedDict<std::string, std::shared_ptr<Module>>::Iterator;
  using ConstIterator = torch::OrderedDict<std::string, std::shared_ptr<Module>>::ConstIterator;

  ModuleDictImpl() = default;

  /// Constructs the `ModuleDict` from an OrderedDict or `ModuleDict`.
  explicit ModuleDictImpl(
      const torch::OrderedDict<std::string, std::shared_ptr<Module>>& modules) {
    update(modules);
  }

  /// Constructs the `ModuleDict` from a list of string-Module paris.
  explicit ModuleDictImpl(
      const std::vector<std::pair<std::string, std::shared_ptr<Module>>>& modules) {
    update(modules);
  }

  /// Special cloning function for `ModuleDict` because it does not use
  /// `reset()`.
  std::shared_ptr<Module> clone(
      const optional<Device>& device = nullopt) const override {
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
    stream << "torch::nn::ModuleDict(" << std::endl;
    for (const auto& pair : modules_) {
      stream << "  (" << pair.key() << "): ";
      pair.value()->pretty_print(stream);
      stream << std::endl;
    }
    stream << ")";
  }

  void clear() {
    modules_.clear();
  }

  /// Return the items in the `ModuleDict`.
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> items() const {
    return modules_.pairs();
  }

  /// Return the keys in the `ModuleDict`.
  std::vector<std::string> keys() const {
    return modules_.keys();
  }

  /// Removes and returns the value associated with the given `key`.
  /// Throws an exception if no such key is stored in the `ModuleDict`. 
  /// Check contains(key) before for a non-throwing way of access.
  std::shared_ptr<Module> pop(const std::string& key) {
    auto module = modules_[key];
    modules_.erase(key);
    return module;
  }

  /// Updated the `ModuleDict` with key-value paris from OrderedDict
  /// or another `ModuleDict`.
  template <typename Container>
  void update(const Container& container) {
    for (auto& item : container) {
      if (contains(item.key())) {
        modules_[item.key()] = item.value();
        replace_module(item.key(), modules_[item.key()]);
      }
      else {
        insert(item.key(), item.value());
      }
    }
  }

  /// Updated the `ModuleDict` with key-value paris from vector of key-module pairs
  void update(
      const std::vector<std::pair<std::string, std::shared_ptr<Module>>>& container) {
    for (auto& item : container) {
      if (contains(item.first)) {
        modules_[item.first] = item.second;
        replace_module(item.first, modules_[item.first]);
      }
      else {
        insert(item.first, item.second);
      }
    }
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

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ModuleDict`. Check contains(key) before
  /// for a non-throwing way of access.
  std::shared_ptr<Module>& operator[](const std::string& key) {
    return modules_[key];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ModuleDict`. Check contains(key) before
  /// for a non-throwing way of access.
  const std::shared_ptr<Module>& operator[](const std::string& key) const {
    return modules_[key];
  }

private:
  torch::OrderedDict<std::string, std::shared_ptr<Module>> modules_;

  /// Insert a new key-module pair. Throws an exception if the key is already present.
  void insert(const std::string& key, std::shared_ptr<Module> module) {
    modules_.insert(key, std::move(module));
    register_module(key, modules_.back().value());
  }

};

TORCH_MODULE(ModuleDict);

} // namespace nn
} // namespace torch
