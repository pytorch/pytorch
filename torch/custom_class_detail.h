#pragma once

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

template <typename Get, typename Set,
          typename = c10::guts::function_signature_t<Get>, typename = c10::guts::function_signature_t<Set>>
struct pickle_factory;

template <typename Get, typename Set,
          typename RetState, typename Self, typename NewInstance, typename ArgState>
struct pickle_factory<Get, Set, RetState(Self), NewInstance(ArgState)> {
  pickle_factory(Get &&g, Set &&s) : g(std::move(g)), s(std::move(s)) {}

  [[noreturn]] ArgState arg_state_type() {}

  Get g;
  Set s;
};

} // namespace detail


template <class... Types>
detail::types<void, Types...> init() { return detail::types<void, Types...>{}; }

template <typename GetState, typename SetState>
detail::pickle_factory<GetState, SetState> pickle_(GetState &&g, SetState &&s) {
  static_assert(c10::guts::is_stateless_lambda<std::decay_t<GetState>>::value &&
                c10::guts::is_stateless_lambda<std::decay_t<SetState>>::value,
                "torch::jit::pickle_ currently only supports lambdas as "
                "__getstate__ and __setstate__ arguments.");
  return detail::pickle_factory<GetState, SetState>(std::move(g), std::move(s));
}

}}  // namespace torch::jit
