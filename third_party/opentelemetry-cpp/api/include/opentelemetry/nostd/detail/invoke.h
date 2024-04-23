// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#include "opentelemetry/nostd/detail/decay.h"
#include "opentelemetry/nostd/detail/void.h"
#include "opentelemetry/version.h"

#define OPENTELEMETRY_RETURN(...) \
  noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) { return __VA_ARGS__; }

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
namespace detail
{

template <typename T>
struct is_reference_wrapper : std::false_type
{};

template <typename T>
struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type
{};

template <bool, int>
struct Invoke;

template <>
struct Invoke<true /* pmf */, 0 /* is_base_of */>
{
  template <typename R, typename T, typename Arg, typename... Args>
  inline static constexpr auto invoke(R T::*pmf, Arg &&arg, Args &&... args)
      OPENTELEMETRY_RETURN((std::forward<Arg>(arg).*pmf)(std::forward<Args>(args)...))
};

template <>
struct Invoke<true /* pmf */, 1 /* is_reference_wrapper */>
{
  template <typename R, typename T, typename Arg, typename... Args>
  inline static constexpr auto invoke(R T::*pmf, Arg &&arg, Args &&... args)
      OPENTELEMETRY_RETURN((std::forward<Arg>(arg).get().*pmf)(std::forward<Args>(args)...))
};

template <>
struct Invoke<true /* pmf */, 2 /* otherwise */>
{
  template <typename R, typename T, typename Arg, typename... Args>
  inline static constexpr auto invoke(R T::*pmf, Arg &&arg, Args &&... args)
      OPENTELEMETRY_RETURN(((*std::forward<Arg>(arg)).*pmf)(std::forward<Args>(args)...))
};

template <>
struct Invoke<false /* pmo */, 0 /* is_base_of */>
{
  template <typename R, typename T, typename Arg>
  inline static constexpr auto invoke(R T::*pmo, Arg &&arg)
      OPENTELEMETRY_RETURN(std::forward<Arg>(arg).*pmo)
};

template <>
struct Invoke<false /* pmo */, 1 /* is_reference_wrapper */>
{
  template <typename R, typename T, typename Arg>
  inline static constexpr auto invoke(R T::*pmo, Arg &&arg)
      OPENTELEMETRY_RETURN(std::forward<Arg>(arg).get().*pmo)
};

template <>
struct Invoke<false /* pmo */, 2 /* otherwise */>
{
  template <typename R, typename T, typename Arg>
  inline static constexpr auto invoke(R T::*pmo, Arg &&arg)
      OPENTELEMETRY_RETURN((*std::forward<Arg>(arg)).*pmo)
};

template <typename R, typename T, typename Arg, typename... Args>
inline constexpr auto invoke_impl(R T::*f, Arg &&arg, Args &&... args)
    OPENTELEMETRY_RETURN(Invoke<std::is_function<R>::value,
                                (std::is_base_of<T, decay_t<Arg>>::value
                                     ? 0
                                     : is_reference_wrapper<decay_t<Arg>>::value ? 1 : 2)>::
                             invoke(f, std::forward<Arg>(arg), std::forward<Args>(args)...))

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4100)
#endif
        template <typename F, typename... Args>
        inline constexpr auto invoke_impl(F &&f, Args &&... args)
            OPENTELEMETRY_RETURN(std::forward<F>(f)(std::forward<Args>(args)...))
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
}  // namespace detail

/* clang-format off */
template <typename F, typename... Args>
inline constexpr auto invoke(F &&f, Args &&... args)
    OPENTELEMETRY_RETURN(detail::invoke_impl(std::forward<F>(f), std::forward<Args>(args)...))

namespace detail
/* clang-format on */
{

  template <typename Void, typename, typename...>
  struct invoke_result
  {};

  template <typename F, typename... Args>
  struct invoke_result<void_t<decltype(nostd::invoke(std::declval<F>(), std::declval<Args>()...))>,
                       F, Args...>
  {
    using type = decltype(nostd::invoke(std::declval<F>(), std::declval<Args>()...));
  };

}  // namespace detail

template <typename F, typename... Args>
using invoke_result = detail::invoke_result<void, F, Args...>;

template <typename F, typename... Args>
using invoke_result_t = typename invoke_result<F, Args...>::type;

namespace detail
{

template <typename Void, typename, typename...>
struct is_invocable : std::false_type
{};

template <typename F, typename... Args>
struct is_invocable<void_t<invoke_result_t<F, Args...>>, F, Args...> : std::true_type
{};

template <typename Void, typename, typename, typename...>
struct is_invocable_r : std::false_type
{};

template <typename R, typename F, typename... Args>
struct is_invocable_r<void_t<invoke_result_t<F, Args...>>, R, F, Args...>
    : std::is_convertible<invoke_result_t<F, Args...>, R>
{};

}  // namespace detail

template <typename F, typename... Args>
using is_invocable = detail::is_invocable<void, F, Args...>;

template <typename R, typename F, typename... Args>
using is_invocable_r = detail::is_invocable_r<void, R, F, Args...>;
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE

#undef OPENTELEMETRY_RETURN
