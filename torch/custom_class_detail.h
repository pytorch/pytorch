#pragma once

#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeTraits.h>

namespace torch {
namespace jit {

namespace detail {

// Argument type utilities
template <class R, class...>
struct types {
  using type = types;
};

template <class Sig>
struct args;

// Method
template <class R, class CurClass, class... Args>
struct args<R (CurClass::*)(Args...)> : types<R, Args...> {};

// Const method
template <class R, class CurClass, class... Args>
struct args<R (CurClass::*)(Args...) const> : types<R, Args...> {};

template <class Sig>
using args_t = typename args<Sig>::type;

} // namespace detail

TORCH_API std::vector<c10::RegisterOperators>& registeredOps();
TORCH_API std::shared_ptr<script::CompilationUnit>& classCU();

} // namespace jit
} // namespace torch
