#pragma once

#include <torch/tensor.h>

#include <ATen/ArrayRef.h>
#include <ATen/Tensor.h>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace torch {
class TensorListView {
 public:
  /// Constructs an TensorListView from a single element.
  /*implicit*/ TensorListView(const torch::Tensor& tensor)
      : backing_vector_({tensor}) {}

  /// Constructs an TensorListView from a std::vector.
  template <typename A>
  /*implicit*/ TensorListView(const std::vector<torch::Tensor, A>& vector)
      : backing_vector_(vector.size()) {
    std::copy(vector.begin(), vector.end(), backing_vector_.begin());
  }

  /// Constructs an TensorListView from a std::array
  template <size_t N>
  /*implicit*/ TensorListView(const std::array<torch::Tensor, N>& array)
      : backing_vector_(N) {
    std::copy(array.begin(), array.end(), backing_vector_.begin());
  }

  /// Constructs an TensorListView from a C array.
  template <size_t N>
  /*implicit*/ TensorListView(const torch::Tensor (&array)[N])
      : backing_vector_(N) {
    std::copy(std::begin(array), std::end(array), backing_vector_.begin());
  }

  /// Constructs an TensorListView from a std::initializer_list.
  /*implicit*/ TensorListView(const std::initializer_list<torch::Tensor>& list)
      : backing_vector_(list.size()) {
    std::copy(list.begin(), list.end(), backing_vector_.begin());
  }

  /// Implicitly converts the `TensorListView` to an `ArrayRef` of the backing
  /// vector.
  operator at::ArrayRef<at::Tensor>() {
    return backing_vector_;
  }

 private:
  std::vector<at::Tensor> backing_vector_;
};
} // namespace torch
