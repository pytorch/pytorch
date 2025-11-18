#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/ordered_dict.h>
#include <utility>
#include <vector>

namespace torch::nn {

class ParameterDictImpl : public Cloneable<ParameterDictImpl> {
 public:
  using Iterator = OrderedDict<std::string, Tensor>::Iterator;
  using ConstIterator = OrderedDict<std::string, Tensor>::ConstIterator;

  ParameterDictImpl() = default;

  explicit ParameterDictImpl(
      const torch::OrderedDict<std::string, torch::Tensor>& params) {
    parameters_ = params;
  }

  /// `reset()` is empty for `ParameterDict`, since it does not have
  /// parameters of its own.
  void reset() override {}

  /// Pretty prints the `ParameterDict` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ParameterDict(" << '\n';
    for (const auto& pair : parameters_) {
      stream << "(" << pair.key() << ")"
             << ": Parameter containing: [" << pair.value().scalar_type()
             << " of size " << pair.value().sizes() << "]";
      ;
      stream << '\n';
    }
    stream << ")";
  }

  /// Insert the parameter along with the key into ParameterDict
  /// The parameter is set to be require grad by default
  Tensor& insert(const std::string& key, const Tensor& param) {
    bool requires_grad = param.requires_grad();
    return register_parameter(key, param, requires_grad);
  }

  /// Remove key from the ParameterDict and return its value, throw exception
  /// if the key is not contained. Please check contains(key) before for a
  /// non-throwing access.
  Tensor pop(const std::string& key) {
    torch::Tensor v = parameters_[key];
    parameters_.erase(key);
    return v;
  }

  /// Return the keys in the dict
  ::std::vector<std::string> keys() const {
    return parameters_.keys();
  }

  /// Return the Values in the dict
  ::std::vector<torch::Tensor> values() const {
    return parameters_.values();
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
    for (auto& item : container) {
      parameters_[item.key()] = item.value();
    }
  }

  /// Remove all parameters in the ParameterDict
  void clear() {
    parameters_.clear();
  }

  /// Check if the certain parameter with the key in the ParameterDict
  bool contains(const std::string& key) const noexcept {
    return parameters_.contains(key);
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  const Tensor& get(const std::string& key) const {
    return parameters_[key];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  Tensor& get(const std::string& key) {
    return parameters_[key];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  Tensor& operator[](const std::string& key) {
    return parameters_[key];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  const Tensor& operator[](const std::string& key) const {
    return parameters_[key];
  }
};

TORCH_MODULE(ParameterDict);

} // namespace torch::nn
