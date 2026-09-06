#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Variadic.h>
#include <torch/csrc/autograd/variable.h>

#include <type_traits>
#include <utility>

namespace torch {

using at::IterArgs;

struct CountTensors : IterArgs<CountTensors> {
  size_t out = 0;
  void operator()(const at::Tensor& x) {
    out += 1;
  }
  void operator()(const std::optional<at::Tensor>& x) {
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
//                                 Utilities
//===----------------------------------------------------------------------===//

template <typename Function, typename... Ts>
void apply(Function function, Ts&&... ts) {
  ((void)function(std::forward<Ts>(ts)), ...);
}

template <
    typename ReturnType,
    typename... Ts,
    typename Function,
    typename Accessor,
    size_t... Is>
ReturnType unpack(
    Function function,
    Accessor accessor,
    std::index_sequence<Is...> /*unused*/);

template <
    typename ReturnType,
    typename... Ts,
    typename Function,
    typename Accessor>
ReturnType unpack(Function function, Accessor accessor) {
  return ReturnType(unpack<ReturnType, Ts...>(
      std::move(function),
      std::move(accessor),
      std::make_index_sequence<sizeof...(Ts)>()));
}

template <
    typename ReturnType,
    typename... Ts,
    typename Function,
    typename Accessor,
    size_t... Is>
ReturnType unpack(
    Function function,
    Accessor accessor,
    std::index_sequence<Is...> /*unused*/) {
  return ReturnType(function(accessor.template operator()<Ts>(Is)...));
}

} // namespace torch
