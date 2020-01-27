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

template <
    typename Get,
    typename Set,
    typename = typename c10::guts::infer_function_traits<Get>::type::func_type,
    typename = typename c10::guts::infer_function_traits<Set>::type::func_type>
struct pickle_factory;

template <
    typename Get,
    typename Set,
    typename RetState,
    typename Self,
    typename NewInstance,
    typename ArgState>
struct pickle_factory<Get, Set, RetState(Self), NewInstance(ArgState)> {
  pickle_factory(Get&& g, Set&& s) : g(std::move(g)), s(std::move(s)) {}

  using arg_state_type = ArgState;

  Get g;
  Set s;
};

} // namespace detail

TORCH_API std::vector<c10::RegisterOperators>& registeredOps();
TORCH_API std::shared_ptr<script::CompilationUnit>& classCU();

template <typename GetState, typename SetState>
detail::pickle_factory<GetState, SetState> pickle_(GetState&& g, SetState&& s) {
  static_assert(
      c10::guts::is_stateless_lambda<std::decay_t<GetState>>::value &&
          c10::guts::is_stateless_lambda<std::decay_t<SetState>>::value,
      "torch::jit::pickle_ currently only supports lambdas as "
      "__getstate__ and __setstate__ arguments.");
  return detail::pickle_factory<GetState, SetState>(std::move(g), std::move(s));
}

} // namespace jit
} // namespace torch
