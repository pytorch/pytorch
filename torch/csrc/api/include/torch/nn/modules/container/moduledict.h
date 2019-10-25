#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/ordered_dict.h>

#include <vector>

namespace torch {
namespace nn {

class ModuleDictImpl : public Cloneable<ModuleDictImpl> {
 public:
  using Iterator = OrderedDict<std::string, std::shared_ptr<Module>>::Iterator;
  using ConstIterator =
      OrderedDict<std::string, std::shared_ptr<Module>>::ConstIterator;

  ModuleDictImpl() = default;

  std::shared_ptr<Module> clone(
      const optional<Device>& device = nullopt) const override {
    auto clone = std::make_shared<ModuleDictImpl>();
    for (const auto& pair : children_) {
      clone->insert(pair.key(), pair.value()->clone(device));
    }
    return clone;
  }

  /// `reset()` is empty for `ModuleList`, since it does not have parameters of
  /// its own.
  void reset() override {}

  /// Pretty prints the `ModuleList` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ModuleDict";
  }

  void erase(const std::string& name) {
    children_.erase(name);
  }

  /// Returns the number of items currently stored in the `ModuleDict`.
  size_t size() const noexcept {
    return children_.size();
  }

  /// Returns true if the `ModuleDict` contains no elements.
  bool is_empty() const noexcept {
    return children_.is_empty();
  }

  /// Removes all items from this `ModuleDict`.
  void clear();

  std::shared_ptr<Module> pop(const std::string& name) {
    auto value = children_[name];
    children_.erase(name);
    return value;
  }

  OrderedDict<std::string, std::shared_ptr<Module>> items() const noexcept {
    return children_;
  }

  /// Inserts all items from `other` into this `ModuleDict`. If any key from
  /// `other` is already present in this `ModuleDict`, an exception is thrown.
  void update(ModuleDictImpl&& other) {
    children_.update(std::move(other.children_));
  }

  /// Inserts all items from `other` into this `ModuleDict`. If any key from
  /// `other` is already present in this `ModuleDict`, an exception is thrown.
  void update(const ModuleDictImpl& other) {
    children_.update(other.children_);
  }

  std::shared_ptr<Module>& insert(
      const std::string& key,
      std::shared_ptr<Module>&& value) {
    return children_.insert(key, std::move(value));
  }

  std::shared_ptr<Module>& insert(
      const std::string& key,
      const std::shared_ptr<Module>& value) {
    return children_.insert(key, value);
  }

  /// Returns an iterator to the start of the `ModuleDict`.
  Iterator begin() {
    return children_.begin();
  }

  /// Returns a const iterator to the start of the `ModuleDict`.
  ConstIterator begin() const {
    return children_.begin();
  }

  /// Returns an iterator to the end of the `ModuleDict`.
  Iterator end() {
    return children_.end();
  }

  /// Returns a const iterator to the end of the `ModuleDict`.
  ConstIterator end() const {
    return children_.end();
  }

  std::shared_ptr<Module>& operator[](const std::string& key) {
    return children_[key];
  }

  const std::shared_ptr<Module>& operator[](const std::string& key) const {
    return children_[key];
  }
};

TORCH_MODULE(ModuleDict);

} // namespace nn
} // namespace torch
