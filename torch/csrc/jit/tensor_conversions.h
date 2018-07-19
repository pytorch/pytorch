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
inline T tensor_as(at::Tensor&& t);

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
  void* data = const_cast<void*>(reinterpret_cast<const void*>(l.data()));
  auto sizes = {static_cast<int64_t>(l.size())};
  return at::from_blob(data, sizes, at::kLong).clone();
}

inline at::Tensor as_tensor(const at::Scalar& s) {
  return s.toTensor();
}

template<size_t N>
inline at::Tensor as_tensor(std::array<bool, N>&& bools) {
  auto r = at::empty({N}, at::kByte);
  auto accessor = r.accessor<uint8_t, 1>();
  for(size_t i = 0; i < N; ++i) {
    accessor[i] = bools[i];
  }
  return r;
}

template<typename T>
inline at::Tensor as_variable(const T& t) {
  return autograd::make_variable(as_tensor(t));
}

//////////////////////////////////////////////////////////////////////////////////
// Helper for retrieving constants
//////////////////////////////////////////////////////////////////////////////////

// if a value is a constant then try to turn into type T using the
// same rules as the interpreter
template<typename T>
at::optional<T> constant_as(Value* v) {
  if(v->node()->kind() != prim::Constant)
    return at::nullopt;
  auto tensor = v->node()->t(attr::value);
  try {
    return tensor_as<T>(std::move(tensor));
  } catch (tensor_conversion_error& err) {
    return at::nullopt;
  }
}

}} // namespace torch::jit
