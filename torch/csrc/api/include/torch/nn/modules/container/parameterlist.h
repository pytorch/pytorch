#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>

#include <vector>

namespace torch {
namespace nn {
class ParameterListImpl : public Cloneable<ParameterListImpl> {
 public:
  using Iterator = std::vector<torch::Tensor>::iterator;
  using ConstIterator = std::vector<torch::Tensor>::const_iterator;

  ParameterListImpl() = default;

  /// Constructs the `ParameterList` from a variadic list of ParameterList.
  template <typename... Tensors>
  explicit ParameterListImpl(Tensors&&... params) {
    parameters.reserve(sizeof...(Tensors));
    push_back_var(std::forward<Tensors>(params)...);
  }

  template <typename... Tensors>
  explicit ParameterListImpl(const Tensors&... params) {
    parameters.reserve(sizeof...(Tensors));
    push_back_var(std::forward<Tensors>(params)...);
  }

  /// `reset()` is empty for `ParameterList`, since it does not have parameters
  /// of its own.
  void reset() override {}

  /// Pretty prints the `ParameterList` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ParameterList(" << std::endl;
    for (const auto& pair : parameters_) {
      stream << "(" << pair.key() << ")"
             << ": Parameter containing: [" << pair.value().scalar_type()
             << " of size " << pair.value().sizes() << "]";
      ;
      stream << std::endl;
    }
    stream << ")";
  }

  /// push the a given parameter at the end of the list
  void append(torch::Tensor&& param) {
    bool requires_grad = param.requires_grad();
    parameters.push_back(std::move(param));
    register_parameter(
        c10::to_string(parameters.size() - 1),
        parameters[parameters.size() - 1],
        requires_grad);
  }

  /// push the a given parameter at the end of the list
  void append(const torch::Tensor& param) {
    parameters.push_back(param);
    register_parameter(
        c10::to_string(parameters.size() - 1),
        parameters[parameters.size() - 1],
        param.requires_grad());
  }

  /// push the a given parameter at the end of the list
  // void append(const OrderedDict<std::string, torch::Tensor>::Item& pair) {
  //   register_parameter(c10::to_string(parameters_.size()), pair.value(),
  //   pair.value().requires_grad());
  // }

  /// extend parameters from a container to the end of the list
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& param : container) {
      append(param);
    }
  }

  /// Returns an iterator to the start of the ParameterList
  Iterator begin() {
    return parameters.begin();
  }

  /// Returns a const iterator to the start of the ParameterList
  ConstIterator begin() const {
    return parameters.begin();
  }

  /// Returns an iterator to the end of the ParameterList
  Iterator end() {
    return parameters.end();
  }

  /// Returns a const iterator to the end of the ParameterList
  ConstIterator end() const {
    return parameters.end();
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  at::Tensor& at(size_t idx) {
    TORCH_CHECK(idx < size(), "Index out of range");
    return parameters[idx];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  const at::Tensor& at(size_t idx) const {
    TORCH_CHECK(idx < size(), "Index out of range");
    return parameters[idx];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  at::Tensor& operator[](size_t idx) {
    return at(idx);
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterDict`. Check contains(key) before
  /// for a non-throwing way of access
  const at::Tensor& operator[](size_t idx) const {
    return at(idx);
  }

  /// Return the size of the ParameterList
  size_t size() const noexcept {
    return parameters.size();
  }
  /// True if the ParameterList is empty
  bool is_empty() const noexcept {
    return parameters.empty();
  }

 private:
  template <typename Head, typename... Tail>
  void push_back_var(Head&& head, Tail&&... tail) {
    append(std::forward<Head>(head));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `push_back()` a final time (above).
    push_back_var(std::forward<Tail>(tail)...);
  }

  /// The base case, when the list of modules is empty.
  void push_back_var() {}

  // since the `parameters_` in `modules` is an `OrderedDict`
  // And there is no better way to get the reference of the
  // iterator of value from an `OrderedDict`, so a vector is needed here
  std::vector<torch::Tensor> parameters;
};
TORCH_MODULE(ParameterList);
} // namespace nn
} // namespace torch
