#pragma once

#include <torch/nn/cloneable.h>
#include <torch/ordered_dict.h>

#include <vector>

namespace torch {
namespace nn {

class ParameterDictImpl : public Cloneable<ParameterDictImpl> {
 public:
  using Iterator = OrderedDict<std::string, Tensor>::Iterator;
  using ConstIterator = OrderedDict<std::string, Tensor>::ConstIterator;

  ParameterDictImpl() = default;

  explicit ParameterDictImpl(
      torch::OrderedDict<std::string, torch::Tensor> params) {
    parameters_.reserve(params.size());
    for (const auto& item : params) {
      insert(std::move(item.key()), std::move(item.value()));
    }
  }

  /// `reset()` is empty for `ParameterDict`, since it does not have parameters
  /// of its own.
  void reset() override {}

  /// Pretty prints the `ParameterDict` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ParameterDict(" << std::endl;
    for (const auto& pair : parameters_) {
      stream << "(" << pair.key() << ")"
             << ": Parameter containing: [" << pair.value().type()
             << " of size " << pair.value().sizes() << "]";
      ;
      stream << std::endl;
    }
    stream << ")";
  }

  /// Insert the parameter along with the key into ParameterDict
  /// The parameter is set to be require grad by default
  Tensor& insert(std::string key, Tensor param, bool requires_grad = true) {
    return register_parameter(key, param, requires_grad);
  }

  /// Remove key from the ParameterDict, and return its value
  Tensor pop(const std::string& key) {
    TORCH_CHECK(
        parameters_.contains(key),
        "No Parameter with name `",
        key,
        "` is registered");
    torch::Tensor v = parameters_[key].clone();
    parameters_.erase(key);
    return v;
  }

  /// Return the keys in the dict
  ::std::vector<std::string> keys() const {
    std::vector<std::string> keys;
    keys.reserve(parameters_.size());
    for (const auto& param : parameters_) {
      keys.push_back(param.key());
    }
    return keys;
  }

  /// Return the Values in the dict
  ::std::vector<torch::Tensor> values() const {
    std::vector<torch::Tensor> values;
    values.reserve(parameters_.size());
    for (const auto& param : parameters_) {
      values.push_back(param.value());
    }
    return values;
  }

  /// Returns an `OrderedDict` with the parameters of this `Module` along with
  /// their keys, and if `recurse` is true also recursively of every submodule.
  OrderedDict<std::string, Tensor> named_parameters(bool recurse = true) const {
    return named_parameters(recurse);
  }

  /// Return an iterator to the start of ParameterDict
  Iterator begin() {
    return parameters_.begin();
  }

  /// Return a const iterator to the start of ParameterDict
  ConstIterator begin() const {
    return parameters_.begin();
  }

  /// Return an iterator to the end of ParameterDict
  Iterator end() {
    return parameters_.end();
  }

  /// Return a const iterator to the end of ParameterDict
  ConstIterator end() const {
    return parameters_.end();
  }

  /// Return the number of items currently stored in the ParameterDict
  size_t size() const noexcept {
    return parameters_.size();
  }

  /// Return true if the ParameterDict is empty, otherwise return false
  bool empty() const noexcept {
    return parameters_.is_empty();
  }

  /// Update the ParameterDict with the key-value pairs from
  /// another ParameterDict, overwriting existing key
  template <typename Container>
  void update(const Container& container) {
    parameters_.reserve(parameters_.size() + container.size());
    for (auto& item : container) {
      // erase and overwrite the duplicate keys
      if (contains(item.key())) {
        pop(item.key());
      }
      insert(std::move(item.key()), std::move(item.value()));
    }
  }

  /// Remove all parameters in the ParameterDict
  void clear() {
    parameters_.clear();
  }

  /// Check if the centain parameter with the key in the ParameterDict
  bool contains(const std::string& key) {
    return parameters_.contains(key);
  }

  Tensor& get(std::string key) {
    return parameters_[key];
  }

  Tensor& operator[](const std::string& key) {
    return parameters_[key];
  }

  const Tensor& operator[](const std::string& key) const {
    return parameters_[key];
  }
};

TORCH_MODULE(ParameterDict);

} // namespace nn
} // namespace torch