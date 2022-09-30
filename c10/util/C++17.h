#pragma once
#ifndef C10_UTIL_CPP17_H_
#define C10_UTIL_CPP17_H_

#include <c10/macros/Macros.h>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
    __GNUC__ < 5
#error \
    "You're trying to build PyTorch with a too old version of GCC. We need GCC 5 or later."
#endif

#if defined(__clang__) && __clang_major__ < 4
#error \
    "You're trying to build PyTorch with a too old version of Clang. We need Clang 4 or later."
#endif

#if (defined(_MSC_VER) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)) || \
    (!defined(_MSC_VER) && __cplusplus < 201703L)
#error You need C++17 to compile PyTorch
#endif

#if defined(_WIN32) && (defined(min) || defined(max))
#error Macro clash with min and max -- define NOMINMAX when compiling your program on Windows
#endif

/*
 * This header adds some polyfills with C++17 functionality
 */

namespace c10 {

template <typename F, typename... args>
using invoke_result = typename std::invoke_result<F, args...>;

template <typename F, typename... args>
using invoke_result_t = typename invoke_result<F, args...>::type;

namespace guts {

template <typename Base, typename Child, typename... Args>
typename std::enable_if<
    !std::is_array<Base>::value && !std::is_array<Child>::value &&
        std::is_base_of<Base, Child>::value,
    std::unique_ptr<Base>>::type
make_unique_base(Args&&... args) {
  return std::unique_ptr<Base>(new Child(std::forward<Args>(args)...));
}

template <class... B>
using conjunction = std::conjunction<B...>;
template <class... B>
using disjunction = std::disjunction<B...>;
template <bool B>
using bool_constant = std::bool_constant<B>;
template <class B>
using negation = std::negation<B>;

template <class T>
using void_t = std::void_t<T>;

#if defined(USE_ROCM)
// rocm doesn't like the C10_HOST_DEVICE
#define CUDA_HOST_DEVICE
#else
#define CUDA_HOST_DEVICE C10_HOST_DEVICE
#endif

template <class F, class Tuple>
CUDA_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

#undef CUDA_HOST_DEVICE

template <typename Functor, typename... Args>
typename std::enable_if<
    std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename c10::invoke_result_t<Functor, Args...>>::type
invoke(Functor&& f, Args&&... args) {
  return std::mem_fn(std::forward<Functor>(f))(std::forward<Args>(args)...);
}

template <typename Functor, typename... Args>
typename std::enable_if<
    !std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename c10::invoke_result_t<Functor, Args...>>::type
invoke(Functor&& f, Args&&... args) {
  return std::forward<Functor>(f)(std::forward<Args>(args)...);
}

namespace detail {
struct _identity final {
  template <class T>
  using type_identity = T;

  template <class T>
  decltype(auto) operator()(T&& arg) {
    return std::forward<T>(arg);
  }
};

template <class Func, class Enable = void>
struct function_takes_identity_argument : std::false_type {};
#if defined(_MSC_VER)
// For some weird reason, MSVC shows a compiler error when using guts::void_t
// instead of std::void_t. But we're only building on MSVC versions that have
// std::void_t, so let's just use that one.
template <class Func>
struct function_takes_identity_argument<
    Func,
    std::void_t<decltype(std::declval<Func>()(_identity()))>> : std::true_type {
};
#else
template <class Func>
struct function_takes_identity_argument<
    Func,
    void_t<decltype(std::declval<Func>()(_identity()))>> : std::true_type {};
#endif

} // namespace detail

/*
 * Get something like C++17 if constexpr in C++14.
 *
 * Example 1: simple constexpr if/then/else
 *   template<int arg> int increment_absolute_value() {
 *     int result = arg;
 *     if_constexpr<(arg > 0)>(
 *       [&] { ++result; }  // then-case
 *       [&] { --result; }  // else-case
 *     );
 *     return result;
 *   }
 *
 * Example 2: without else case (i.e. conditionally prune code from assembly)
 *   template<int arg> int decrement_if_positive() {
 *     int result = arg;
 *     if_constexpr<(arg > 0)>(
 *       // This decrement operation is only present in the assembly for
 *       // template instances with arg > 0.
 *       [&] { --result; }
 *     );
 *     return result;
 *   }
 *
 * Example 3: branch based on type (i.e. replacement for SFINAE)
 *   struct MyClass1 {int value;};
 *   struct MyClass2 {int val};
 *   template <class T>
 *   int func(T t) {
 *     return if_constexpr<std::is_same<T, MyClass1>::value>(
 *       [&](auto _) { return _(t).value; }, // this code is invalid for T ==
 * MyClass2, so a regular non-constexpr if statement wouldn't compile
 *       [&](auto _) { return _(t).val; }    // this code is invalid for T ==
 * MyClass1
 *     );
 *   }
 *
 * Note: The _ argument passed in Example 3 is the identity function, i.e. it
 * does nothing. It is used to force the compiler to delay type checking,
 * because the compiler doesn't know what kind of _ is passed in. Without it,
 * the compiler would fail when you try to access t.value but the member doesn't
 * exist.
 *
 * Note: In Example 3, both branches return int, so func() returns int. This is
 * not necessary. If func() had a return type of "auto", then both branches
 * could return different types, say func<MyClass1>() could return int and
 * func<MyClass2>() could return string.
 *
 * Note: if_constexpr<cond, t, f> is *eager* w.r.t. template expansion - meaning
 * this polyfill does not behave like a true "if statement at compilation time".
 *       The `_` trick above only defers typechecking, which happens after
 * templates have been expanded. (Of course this is all that's necessary for
 * many use cases).
 */
template <bool Condition, class ThenCallback, class ElseCallback>
decltype(auto) if_constexpr(
    ThenCallback&& thenCallback,
    ElseCallback&& elseCallback) {
  // If we have C++17, just use it's "if constexpr" feature instead of wrapping
  // it. This will give us better error messages.
  if constexpr (Condition) {
    if constexpr (detail::function_takes_identity_argument<
                      ThenCallback>::value) {
      // Note that we use static_cast<T&&>(t) instead of std::forward (or
      // ::std::forward) because using the latter produces some compilation
      // errors about ambiguous `std` on MSVC when using C++17. This static_cast
      // is just what std::forward is doing under the hood, and is equivalent.
      return static_cast<ThenCallback&&>(thenCallback)(detail::_identity());
    } else {
      return static_cast<ThenCallback&&>(thenCallback)();
    }
  } else {
    if constexpr (detail::function_takes_identity_argument<
                      ElseCallback>::value) {
      return static_cast<ElseCallback&&>(elseCallback)(detail::_identity());
    } else {
      return static_cast<ElseCallback&&>(elseCallback)();
    }
  }
}

template <bool Condition, class ThenCallback>
decltype(auto) if_constexpr(ThenCallback&& thenCallback) {
  // If we have C++17, just use it's "if constexpr" feature instead of wrapping
  // it. This will give us better error messages.
  if constexpr (Condition) {
    if constexpr (detail::function_takes_identity_argument<
                      ThenCallback>::value) {
      // Note that we use static_cast<T&&>(t) instead of std::forward (or
      // ::std::forward) because using the latter produces some compilation
      // errors about ambiguous `std` on MSVC when using C++17. This static_cast
      // is just what std::forward is doing under the hood, and is equivalent.
      return static_cast<ThenCallback&&>(thenCallback)(detail::_identity());
    } else {
      return static_cast<ThenCallback&&>(thenCallback)();
    }
  }
}

// GCC 4.8 doesn't define std::to_string, even though that's in C++11. Let's
// define it.
namespace detail {
class DummyClassForToString final {};
} // namespace detail
} // namespace guts
} // namespace c10
namespace std {
// We use SFINAE to detect if std::to_string exists for a type, but that only
// works if the function name is defined. So let's define a std::to_string for a
// dummy type. If you're getting an error here saying that this overload doesn't
// match your std::to_string() call, then you're calling std::to_string() but
// should be calling c10::guts::to_string().
inline std::string to_string(c10::guts::detail::DummyClassForToString) {
  return "";
}

} // namespace std
namespace c10 {
namespace guts {
namespace detail {

template <class T, class Enable = void>
struct to_string_ final {
  static std::string call(T value) {
    std::ostringstream str;
    str << value;
    return str.str();
  }
};
// If a std::to_string exists, use that instead
template <class T>
struct to_string_<T, void_t<decltype(std::to_string(std::declval<T>()))>>
    final {
  static std::string call(T value) {
    return std::to_string(value);
  }
};
} // namespace detail
template <class T>
inline std::string to_string(T value) {
  return detail::to_string_<T>::call(value);
}

template <class T>
constexpr const T& min(const T& a, const T& b) {
  return (b < a) ? b : a;
}

template <class T>
constexpr const T& max(const T& a, const T& b) {
  return (a < b) ? b : a;
}

} // namespace guts
} // namespace c10

#endif // C10_UTIL_CPP17_H_
