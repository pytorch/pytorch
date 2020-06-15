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
    params_.reserve(params.size());
    for (auto& item : params) {
      insert(std::move(item.key()), std::move(item.value()));
    }
  }

  /// `reset()` is empty for `ParameterDict`, since it does not have parameters
  /// of its own.
  void reset() override {}

  /// Pretty prints the `ParameterDict` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "ParameterDict(" << std::endl;
    for (const auto& pair : params_) {
      stream << std::endl;
      stream << pair.key() << ": " << pair.value();
    }
    stream << ")";
  }

  /// Return the keys in the dict
  ::std::vector<std::string> keys() const {
    std::vector<std::string> keys;
    keys.reserve(params_.size());
    for (const auto& param : params_) {
      keys.push_back(param.key());
    }
    return keys;
  }

  /// Return the Values in the dict
  ::std::vector<torch::Tensor> values() const {
    std::vector<torch::Tensor> values;
    values.reserve(params_.size());
    for (const auto& param : params_) {
      values.push_back(param.value());
    }
    return values;
  }

  /// Return an iterator to the start of ParameterDict
  Iterator begin() {
    return params_.begin();
  }

  /// Return a const iterator to the start of ParameterDict
  ConstIterator begin() const {
    return params_.begin();
  }

  /// Return an iterator to the end of ParameterDict
  Iterator end() {
    return params_.end();
  }

  /// Return a const iterator to the end of ParameterDict
  ConstIterator end() const {
    return params_.end();
  }

  /// Return the number of items currently stored in the ParameterDict
  size_t size() const noexcept {
    return params_.size();
  }

  /// Return true if the ParameterDict is empty, otherwise return false
  bool empty() const noexcept {
    return params_.is_empty();
  }

  void update(ParameterDictImpl other) {
    params_.update(std::move(other.params_));
  }

  void erase(const std::string& name) {
    params_.erase(name);
  }

  void clear() {
    params_.clear();
  }

  Tensor& insert(const std::string& key, Tensor value) {
    params_.insert(key, std::move(value));
    register_parameter(key, params_[key]);
    return params_[key];
  }

  bool contains(const std::string& key) {
    return params_.contains(key);
  }

  Tensor& operator[](const std::string& key) {
    return params_[key];
  }

  const Tensor& operator[](const std::string& key) const {
    return params_[key];
  }

  OrderedDict<std::string, torch::Tensor> params_;
};

TORCH_MODULE(ParameterDict);

} // namespace nn
} // namespace torch