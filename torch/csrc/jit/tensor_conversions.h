#pragma once
#include "ATen/ATen.h"

#include <array>
#include <type_traits>

namespace torch { namespace jit {

//////////////////////////////////////////////////////////////////////////////////
// Tensor -> T conversion
//////////////////////////////////////////////////////////////////////////////////

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

template<size_t N>
struct tensor_as_impl<std::array<bool, N>> {
  std::array<bool, N> operator()(at::Tensor&& t) {
    throw std::runtime_error("tensor_as<std::array<bool, N>>: NYI");
  }
};

template<>
struct tensor_as_impl<at::IntList> {
  at::IntList operator()(at::Tensor&& t) {
    if (t.type().scalarType() != at::ScalarType::Long)
      throw std::runtime_error("Expected a LongTensor");
    if (t.dim() != 1)
      throw std::runtime_error("Expected a 1D LongTensor");
    if (!t.is_contiguous())
      throw std::runtime_error("Expected a contiguous LongTensor");
    return at::IntList{t.data<int64_t>(), static_cast<size_t>(t.numel())};
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
inline T tensor_as(at::Tensor&& t) {
  return detail::tensor_as_impl<T>()(std::move(t));
}

//////////////////////////////////////////////////////////////////////////////////
// T -> Tensor conversion
//////////////////////////////////////////////////////////////////////////////////

inline at::Tensor as_tensor(int64_t v) {
  return at::Scalar(v).toTensor();
}

inline at::Tensor as_tensor(double v) {
  return at::Scalar(v).toTensor();
}

inline at::Tensor as_tensor(bool v) {
  return at::Scalar(v).toTensor();
}

inline at::Tensor as_tensor(at::IntList l) {
  return at::CPU(at::kLong).tensorFromBlob(const_cast<void*>(reinterpret_cast<const void*>(l.data())),
                                           {static_cast<int64_t>(l.size())}).clone();
}


inline at::Tensor as_tensor(at::Scalar&& s) {
  return s.toTensor();
}

}} // namespace torch::jit
