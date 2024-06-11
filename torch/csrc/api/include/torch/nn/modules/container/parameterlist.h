#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>

#include <vector>

namespace torch {
namespace nn {
class ParameterListImpl : public Cloneable<ParameterListImpl> {
 public:
  using Iterator = typename std::vector<
      OrderedDict<std::string, torch::Tensor>::Item>::iterator;
  using ConstIterator = typename std::vector<
      OrderedDict<std::string, torch::Tensor>::Item>::const_iterator;

  ParameterListImpl() = default;

  /// Constructs the `ParameterList` from a variadic list of ParameterList.
  template <typename... Tensors>
  explicit ParameterListImpl(Tensors&&... params) {
    parameters_.reserve(sizeof...(Tensors));
    push_back_var(std::forward<Tensors>(params)...);
  }

  template <typename... Tensors>
  explicit ParameterListImpl(const Tensors&... params) {
    parameters_.reserve(sizeof...(Tensors));
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
    register_parameter(
        c10::to_string(parameters_.size()), std::move(param), requires_grad);
  }

  /// push the a given parameter at the end of the list
  void append(const torch::Tensor& param) {
    bool requires_grad = param.requires_grad();
    register_parameter(
        c10::to_string(parameters_.size()), param, requires_grad);
  }

  /// push the a given parameter at the end of the list
  /// And the key of the pair will be discarded, only the value
  /// will be added into the `ParameterList`
  void append(const OrderedDict<std::string, torch::Tensor>::Item& pair) {
    register_parameter(
        c10::to_string(parameters_.size()),
        pair.value(),
        pair.value().requires_grad());
  }

  /// extend parameters from a container to the end of the list
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& param : container) {
      append(param);
    }
  }

  /// Returns an iterator to the start of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string,
  /// torch::Tensor>::Item`
  Iterator begin() {
    return parameters_.begin();
  }

  /// Returns a const iterator to the start of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string,
  /// torch::Tensor>::Item`
  ConstIterator begin() const {
    return parameters_.begin();
  }

  /// Returns an iterator to the end of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string,
  /// torch::Tensor>::Item`
  Iterator end() {
    return parameters_.end();
  }

  /// Returns a const iterator to the end of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string,
  /// torch::Tensor>::Item`
  ConstIterator end() const {
    return parameters_.end();
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterList`. Check contains(key) before
  /// for a non-throwing way of access
  at::Tensor& at(size_t idx) {
    TORCH_CHECK(idx < size(), "Index out of range");
    return parameters_[c10::to_string(idx)];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterList`. Check contains(key) before
  /// for a non-throwing way of access
  const at::Tensor& at(size_t idx) const {
    TORCH_CHECK(idx < size(), "Index out of range");
    return parameters_[c10::to_string(idx)];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterList`. Check contains(key) before
  /// for a non-throwing way of access
  at::Tensor& operator[](size_t idx) {
    return at(idx);
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterList`. Check contains(key) before
  /// for a non-throwing way of access
  const at::Tensor& operator[](size_t idx) const {
    return at(idx);
  }

  /// Return the size of the ParameterList
  size_t size() const noexcept {
    return parameters_.size();
  }
  /// True if the ParameterList is empty
  bool is_empty() const noexcept {
    return parameters_.is_empty();
  }

  /// Overload the +=, so that two ParameterList could be incrementally added
  template <typename Container>
  Container& operator+=(const Container& other) {
    extend(other);
    return *this;
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
};
TORCH_MODULE(ParameterList);
} // namespace nn
} // namespace torch
