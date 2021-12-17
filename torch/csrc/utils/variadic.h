#pragma once

#include <ATen/core/Tensor.h>
#include <torch/csrc/autograd/variable.h>
#include <ATen/core/Variadic.h>

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace torch {

using at::IterArgs;

struct CountTensors : IterArgs<CountTensors> {
  size_t out = 0;
  void operator()(const at::Tensor& x) {
    out += 1;
  }
  void operator()(const c10::optional<at::Tensor>& x) {
    out += x.has_value();
  }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    out += xs.size();
  }
};

template <typename... Args>
size_t count_tensors(Args&&... args) {
  return CountTensors().apply(std::forward<Args>(args)...).out;
}

struct CountVariables : IterArgs<CountVariables> {
  size_t out = 0;
  void operator()(const autograd::Variable& x) {
    out += 1;
  }
  void operator()(at::ArrayRef<autograd::Variable> xs) {
    out += xs.size();
  }
};

template <typename... Args>
inline size_t count_variables(Args&&... args) {
  return CountVariables().apply(std::forward<Args>(args)...).out;
}

//===----------------------------------------------------------------------===//
//                std::index_sequence shim for C++11
//===----------------------------------------------------------------------===//

// A container of type-template parameter indices.
template <size_t... Is>
struct Indices {};

// Decrements the index N, adds N-1 to the list of indices and forwards
// whatever we already have.
template <size_t N, size_t... Is>
struct MakeIndices : MakeIndices<N - 1, N - 1, Is...> {};

// Partial specialization that forms our base case. When N is zero, we stop
// and define a typedef that will be visible to earlier classes due to
// inheritance. The typedef we define is an index list containing the numbers
// 0 through N-1.
template <size_t... Is>
struct MakeIndices<0, Is...> {
  using indices = Indices<Is...>;
};

//===----------------------------------------------------------------------===//
//                                 Utilities
//===----------------------------------------------------------------------===//

template <bool value, typename T = void>
using enable_if_t = typename std::enable_if<value, T>::type;

template <bool value, typename T = void>
using disable_if_t = enable_if_t<!value, T>;

template <typename T>
using decay_t = typename std::decay<T>::type;

namespace detail {
template <bool...>
struct pack;
} // namespace detail

template <bool... values>
struct all_of : std::is_same<
                    detail::pack<values..., true>,
                    detail::pack<true, values...>> {};

template <bool...>
struct any_of;

template <>
struct any_of<> : std::false_type {};

template <bool head, bool... tail>
struct any_of<head, tail...> {
  static constexpr bool value = head || any_of<tail...>::value;
};

template <bool... values>
struct none_of {
  static constexpr bool value = !any_of<values...>::value;
};

template <bool... values>
using enable_if_all_of_t = enable_if_t<all_of<values...>::value>;

template <typename T, typename... Ts>
using disable_if_contains_t =
    enable_if_all_of_t<(!std::is_same<T, decay_t<Ts>>::value)...>;

template <typename Function, typename... Ts>
void apply(Function function, Ts&&... ts) {
  // https://stackoverflow.com/questions/13978916/inserting-a-variadic-argument-list-into-a-vector
  // Creates a dummy array, so that each function call is evaluated in order.
  // `(function(), 0)` is because `function` should (!) return `void`, so
  // according to the comma operator, it is evaluated and its result (`void`)
  // is discarded. Then the zero is evaluated and used as an element in the
  // array. The first zero ensures the array is not empty.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  int _[]{0, (function(std::forward<Ts>(ts)), 0)...};
  (void)_;
}

template <typename ReturnType, typename... Ts, typename Function, typename Accessor>
ReturnType unpack(Function function, Accessor accessor) {
  return ReturnType(unpack<ReturnType, Ts...>(
      std::move(function),
      std::move(accessor),
      typename MakeIndices<sizeof...(Ts)>::indices()));
}

template <typename ReturnType, typename... Ts, typename Function, typename Accessor, size_t... Is>
ReturnType unpack(Function function, Accessor accessor, Indices<Is...>) {
  return ReturnType(function(accessor.template operator()<Ts>(Is)...));
}

} // namespace torch
