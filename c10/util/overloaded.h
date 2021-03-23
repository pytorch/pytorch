#pragma once

namespace c10 {
namespace detail {

template<class...Ts>
struct overloaded_t {};

template<class T0>
struct overloaded_t<T0>:T0 {
  using T0::operator();
  overloaded_t(T0 t0):T0(std::move(t0)) {}
};
template<class T0, class...Ts>
struct overloaded_t<T0, Ts...>:T0, overloaded_t<Ts...> {
  using T0::operator();
  using overloaded_t<Ts...>::operator();
  overloaded_t(T0 t0, Ts... ts):
    T0(std::move(t0)),
    overloaded_t<Ts...>(std::move(ts)...)
  {}
};

}  // namespace detail

// Construct an overloaded callable combining multiple callables, e.g. lambdas
template<class...Ts>
detail::overloaded_t<Ts...> overloaded(Ts...ts){ return {std::move(ts)...}; }

}  // namespace c10
