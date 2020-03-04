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
CUDA_HOST_DEVICE constexpr auto apply_impl(F&& f, Tuple&& t, std::index_sequence<INDEX...>)
-> decltype(std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...))
#endif
{
    return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}
}  // namespace detail

template <class F, class Tuple>
CUDA_HOST_DEVICE constexpr auto apply(F&& f, Tuple&& t) -> decltype(detail::apply_impl(
    std::forward<F>(f), std::forward<Tuple>(t),
    std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{}))
{
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
