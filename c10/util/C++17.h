#pragma once
#ifndef C10_UTIL_CPP17_H_
#define C10_UTIL_CPP17_H_

#include <type_traits>
#include <utility>
#include <memory>
#include <sstream>
#include <string>
#include <cstdlib>
#include <functional>
#include <c10/macros/Macros.h>

#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
  __GNUC__ < 5
#error "You're trying to build PyTorch with a too old version of GCC. We need GCC 5 or later."
#endif

#if defined(__clang__) && __clang_major__ < 4
#error "You're trying to build PyTorch with a too old version of Clang. We need Clang 4 or later."
#endif

#if (defined(_MSC_VER) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201402L)) || (!defined(_MSC_VER) && __cplusplus < 201402L)
#error You need C++14 to compile PyTorch
#endif

#if defined(_WIN32) && (defined(min) || defined(max))
#  error Macro clash with min and max -- define NOMINMAX when compiling your program on Windows
#endif

/*
 * This header adds some polyfills with C++17 functionality
 */

namespace c10 { namespace guts {


template <typename Base, typename Child, typename... Args>
typename std::enable_if<!std::is_array<Base>::value && !std::is_array<Base>::value && std::is_base_of<Base, Child>::value, std::unique_ptr<Base>>::type
make_unique_base(Args&&... args) {
  return std::unique_ptr<Base>(new Child(std::forward<Args>(args)...));
}




#if defined(__cpp_lib_logical_traits) && !(defined(_MSC_VER) && _MSC_VER < 1920)

template <class... B>
using conjunction = std::conjunction<B...>;
template <class... B>
using disjunction = std::disjunction<B...>;
template <bool B>
using bool_constant = std::bool_constant<B>;
template <class B>
using negation = std::negation<B>;

#else

// Implementation taken from http://en.cppreference.com/w/cpp/types/conjunction
template<class...> struct conjunction : std::true_type { };
template<class B1> struct conjunction<B1> : B1 { };
template<class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

// Implementation taken from http://en.cppreference.com/w/cpp/types/disjunction
template<class...> struct disjunction : std::false_type { };
template<class B1> struct disjunction<B1> : B1 { };
template<class B1, class... Bn>
struct disjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>>  { };

// Implementation taken from http://en.cppreference.com/w/cpp/types/integral_constant
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

// Implementation taken from http://en.cppreference.com/w/cpp/types/negation
template<class B>
struct negation : bool_constant<!bool(B::value)> { };

#endif




#ifdef __cpp_lib_void_t

template<class T> using void_t = std::void_t<T>;

#else

// Implementation taken from http://en.cppreference.com/w/cpp/types/void_t
// (it takes CWG1558 into account and also works for older compilers)
template<typename... Ts> struct make_void { typedef void type;};
template<typename... Ts> using void_t = typename make_void<Ts...>::type;

#endif

#ifdef __HIP_PLATFORM_HCC__
// rocm doesn't like the C10_HOST_DEVICE
#define CUDA_HOST_DEVICE
#else
#define CUDA_HOST_DEVICE C10_HOST_DEVICE
#endif

#ifdef __cpp_lib_apply

template <class F, class Tuple>
CUDA_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

#else

// Implementation from http://en.cppreference.com/w/cpp/utility/apply (but modified)
// TODO This is an incomplete implementation of std::apply, not working for member functions.
namespace detail {
template <class F, class Tuple, std::size_t... INDEX>
#if defined(_MSC_VER)
// MSVC has a problem with the decltype() return type, but it also doesn't need it
C10_HOST_DEVICE constexpr auto apply_impl(F&& f, Tuple&& t, std::index_sequence<INDEX...>)
#else
// GCC/Clang need the decltype() return type
CUDA_HOST_DEVICE constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<INDEX...>)
#endif
{
    return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}
}  // namespace detail

template <class F, class Tuple>
CUDA_HOST_DEVICE constexpr decltype(auto) apply(F&& f, Tuple&& t) {
    return detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

#endif

#undef CUDA_HOST_DEVICE


template <typename Functor, typename... Args>
typename std::enable_if<
    std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename std::result_of<Functor && (Args && ...)>::type>::type
invoke(Functor&& f, Args&&... args) {
  return std::mem_fn(std::forward<Functor>(f))(std::forward<Args>(args)...);
}

template <typename Functor, typename... Args>
typename std::enable_if<
    !std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename std::result_of<Functor && (Args && ...)>::type>::type
invoke(Functor&& f, Args&&... args) {
  return std::forward<Functor>(f)(std::forward<Args>(args)...);
}






namespace detail {
struct _identity final {
  template<class T>
  using type_identity = T;

  template<class T>
  decltype(auto) operator()(T&& arg) {
    return std::forward<T>(arg);
  }
};

template<class Func, class Enable = void>
struct function_takes_identity_argument : std::false_type {};
#if defined(_MSC_VER)
// For some weird reason, MSVC shows a compiler error when using guts::void_t instead of std::void_t.
// But we're only building on MSVC versions that have std::void_t, so let's just use that one.
template<class Func>
struct function_takes_identity_argument<Func, std::void_t<decltype(std::declval<Func>()(_identity()))>> : std::true_type {};
#else
template<class Func>
struct function_takes_identity_argument<Func, void_t<decltype(std::declval<Func>()(_identity()))>> : std::true_type {};
#endif

template<bool Condition>
struct _if_constexpr;

template<>
struct _if_constexpr<true> final {
  template<class ThenCallback, class ElseCallback, std::enable_if_t<function_takes_identity_argument<ThenCallback>::value, void*> = nullptr>
  static decltype(auto) call(ThenCallback&& thenCallback, ElseCallback&& /* elseCallback */) {
    // The _identity instance passed in can be used to delay evaluation of an expression,
    // because the compiler can't know that it's just the identity we're passing in.
    return thenCallback(_identity());
  }

  template<class ThenCallback, class ElseCallback, std::enable_if_t<!function_takes_identity_argument<ThenCallback>::value, void*> = nullptr>
  static decltype(auto) call(ThenCallback&& thenCallback, ElseCallback&& /* elseCallback */) {
    return thenCallback();
  }
};

template<>
struct _if_constexpr<false> final {
  template<class ThenCallback, class ElseCallback, std::enable_if_t<function_takes_identity_argument<ElseCallback>::value, void*> = nullptr>
  static decltype(auto) call(ThenCallback&& /* thenCallback */, ElseCallback&& elseCallback) {
    // The _identity instance passed in can be used to delay evaluation of an expression,
    // because the compiler can't know that it's just the identity we're passing in.
    return elseCallback(_identity());
  }

  template<class ThenCallback, class ElseCallback, std::enable_if_t<!function_takes_identity_argument<ElseCallback>::value, void*> = nullptr>
  static decltype(auto) call(ThenCallback&& /* thenCallback */, ElseCallback&& elseCallback) {
    return elseCallback();
  }
};
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
 *       [&](auto _) { return _(t).value; }, // this code is invalid for T == MyClass2, so a regular non-constexpr if statement wouldn't compile
 *       [&](auto _) { return _(t).val; }    // this code is invalid for T == MyClass1
 *     );
 *   }
 *
 * Note: The _ argument passed in Example 3 is the identity function, i.e. it does nothing.
 *       It is used to force the compiler to delay type checking, because the compiler
 *       doesn't know what kind of _ is passed in. Without it, the compiler would fail
 *       when you try to access t.value but the member doesn't exist.
 *
 * Note: In Example 3, both branches return int, so func() returns int. This is not necessary.
 *       If func() had a return type of "auto", then both branches could return different
 *       types, say func<MyClass1>() could return int and func<MyClass2>() could return string.
 *
 * Note: if_constexpr<cond, t, f> is *eager* w.r.t. template expansion - meaning this
 *       polyfill does not behave like a true "if statement at compilation time".
 *       The `_` trick above only defers typechecking, which happens after templates
 *       have been expanded. (Of course this is all that's necessary for many use cases).
 */
template<bool Condition, class ThenCallback, class ElseCallback>
decltype(auto) if_constexpr(ThenCallback&& thenCallback, ElseCallback&& elseCallback) {
#if defined(__cpp_if_constexpr)
  // If we have C++17, just use it's "if constexpr" feature instead of wrapping it.
  // This will give us better error messages.
  if constexpr(Condition) {
    if constexpr (detail::function_takes_identity_argument<ThenCallback>::value) {
      return ::std::forward<ThenCallback>(thenCallback)(detail::_identity());
    } else {
      return ::std::forward<ThenCallback>(thenCallback)();
    }
  } else {
    if constexpr (detail::function_takes_identity_argument<ElseCallback>::value) {
      return ::std::forward<ElseCallback>(elseCallback)(detail::_identity());
    } else {
      return ::std::forward<ElseCallback>(elseCallback)();
    }
  }
#else
  // C++14 implementation of if constexpr
  return detail::_if_constexpr<Condition>::call(::std::forward<ThenCallback>(thenCallback),
                                                 ::std::forward<ElseCallback>(elseCallback));
#endif
}

template<bool Condition, class ThenCallback>
decltype(auto) if_constexpr(ThenCallback&& thenCallback) {
#if defined(__cpp_if_constexpr)
  // If we have C++17, just use it's "if constexpr" feature instead of wrapping it.
  // This will give us better error messages.
  if constexpr(Condition) {
    if constexpr (detail::function_takes_identity_argument<ThenCallback>::value) {
      return ::std::forward<ThenCallback>(thenCallback)(detail::_identity());
    } else {
      return ::std::forward<ThenCallback>(thenCallback)();
    }
  }
#else
  // C++14 implementation of if constexpr
  return if_constexpr<Condition>(::std::forward<ThenCallback>(thenCallback), [] (auto) {});
#endif
}



// GCC 4.8 doesn't define std::to_string, even though that's in C++11. Let's define it.
namespace detail {
class DummyClassForToString final {};
}}}
namespace std {
// We use SFINAE to detect if std::to_string exists for a type, but that only works
// if the function name is defined. So let's define a std::to_string for a dummy type.
// If you're getting an error here saying that this overload doesn't match your
// std::to_string() call, then you're calling std::to_string() but should be calling
// c10::guts::to_string().
inline std::string to_string(c10::guts::detail::DummyClassForToString) { return ""; }

}
namespace c10 { namespace guts { namespace detail {

template<class T, class Enable = void>
struct to_string_ final {
    static std::string call(T value) {
        std::ostringstream str;
        str << value;
        return str.str();
    }
};
// If a std::to_string exists, use that instead
template<class T>
struct to_string_<T, void_t<decltype(std::to_string(std::declval<T>()))>> final {
    static std::string call(T value) {
        return std::to_string(value);
    }
};
}
template<class T> inline std::string to_string(T value) {
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

}}

#endif // C10_UTIL_CPP17_H_
