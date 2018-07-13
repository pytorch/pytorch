#pragma once
#include "ATen/ATen.h"

#include <array>
#include <type_traits>
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace jit {

//////////////////////////////////////////////////////////////////////////////////
// Tensor -> T conversion
//////////////////////////////////////////////////////////////////////////////////
struct tensor_conversion_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

template<typename T>
inline T tensor_as(at::Tensor t);

namespace detail {

template<typename T, typename EnableIf = void>
struct tensor_as_impl {};

template<typename T>
struct tensor_as_impl<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  T operator()(at::Tensor&& t) {
    // workaround for 1-dim 1-element pytorch tensors until zero-dim
    // tensors are fully supported
    if(t.ndimension() == 1 && t.size(0) == 1) {
      t = t[0];
    }
    return at::Scalar(t).to<T>();
  }
};

template<>
struct tensor_as_impl<bool> {
  bool operator()(at::Tensor&& t) {
    return tensor_as<int64_t>(std::move(t)) != 0;
  }
};

// this is an identity but is needed in constant_as in the compiler
template<>
struct tensor_as_impl<at::Tensor> {
  at::Tensor operator()(at::Tensor&& t) {
    return t;
  }
};

template<size_t N>
struct tensor_as_impl<std::array<bool, N>> {
  std::array<bool, N> operator()(at::Tensor&& t) {
    throw tensor_conversion_error("tensor_as<std::array<bool, N>>: NYI");
  }
};

template<>
struct tensor_as_impl<std::vector<int64_t>> {
  std::vector<int64_t> operator()(at::Tensor&& t) {
    if (t.type().scalarType() != at::ScalarType::Long)
      throw tensor_conversion_error("Expected a LongTensor");
    if (t.dim() != 1)
      throw tensor_conversion_error("Expected a 1D LongTensor");
    if (!t.is_contiguous())
      throw tensor_conversion_error("Expected a contiguous LongTensor");
    return std::vector<int64_t>(t.data<int64_t>(), t.data<int64_t>() + t.numel());
  }
};

template<>
struct tensor_as_impl<at::Scalar> {
  at::Scalar operator()(at::Tensor&& t) {
    return at::Scalar(t.view({}));
  }
};

}

template<typename T>
inline T tensor_as(at::Tensor t) {
  return detail::tensor_as_impl<T>()(std::move(t));
}

template<typename T>
inline at::Tensor as_variable(const T& t) {
  return autograd::make_variable(as_tensor(t));
}

}} // namespace torch::jit
