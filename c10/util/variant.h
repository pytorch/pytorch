// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)
//
// From https://github.com/mpark/variant
//
// C10
// - Move to `c10` namespace.
// - Rename namespace `detail` to `detail_`, to not conflict with existing
//   c10 implementations in `detail` namespace.
// - In two functions, the template name reference `I` is changed to
//   `detail_::best_match<Arg, Ts...>::value` to work around gcc 7.3.1 bug.
//   However, this workaround also limits the use cases of `c10::variant`.
//   Please see NOTE [gcc 7.3.1 bug workaround] for details.
// - The following code is moved to `c10/util/in_place.h`:
//   ```
//   struct in_place_t { explicit in_place_t() = default; };
//
//   template <std::size_t I>
//   struct in_place_index_t { explicit in_place_index_t() = default; };
//
//   template <typename T>
//   struct in_place_type_t { explicit in_place_type_t() = default; };
//
//   constexpr in_place_t in_place{};
//   ```
//   so that they can also be used in `c10/util/Optional.h`.

#ifndef C10_UTIL_VARIANT_H_
#define C10_UTIL_VARIANT_H_

/*
   variant synopsis

namespace std {

  // 20.7.2, class template variant
  template <class... Types>
  class variant {
  public:

    // 20.7.2.1, constructors
    constexpr variant() noexcept(see below);
    variant(const variant&);
    variant(variant&&) noexcept(see below);

    template <class T> constexpr variant(T&&) noexcept(see below);

    template <class T, class... Args>
    constexpr explicit variant(in_place_type_t<T>, Args&&...);

    template <class T, class U, class... Args>
    constexpr explicit variant(
        in_place_type_t<T>, initializer_list<U>, Args&&...);

    template <size_t I, class... Args>
    constexpr explicit variant(in_place_index_t<I>, Args&&...);

    template <size_t I, class U, class... Args>
    constexpr explicit variant(
        in_place_index_t<I>, initializer_list<U>, Args&&...);

    // 20.7.2.2, destructor
    ~variant();

    // 20.7.2.3, assignment
    variant& operator=(const variant&);
    variant& operator=(variant&&) noexcept(see below);

    template <class T> variant& operator=(T&&) noexcept(see below);

    // 20.7.2.4, modifiers
    template <class T, class... Args>
    T& emplace(Args&&...);

    template <class T, class U, class... Args>
    T& emplace(initializer_list<U>, Args&&...);

    template <size_t I, class... Args>
    variant_alternative<I, variant>& emplace(Args&&...);

    template <size_t I, class U, class...  Args>
    variant_alternative<I, variant>& emplace(initializer_list<U>, Args&&...);

    // 20.7.2.5, value status
    constexpr bool valueless_by_exception() const noexcept;
    constexpr size_t index() const noexcept;

    // 20.7.2.6, swap
    void swap(variant&) noexcept(see below);
  };

  // 20.7.3, variant helper classes
  template <class T> struct variant_size; // undefined

  template <class T>
  constexpr size_t variant_size_v = variant_size<T>::value;

  template <class T> struct variant_size<const T>;
  template <class T> struct variant_size<volatile T>;
  template <class T> struct variant_size<const volatile T>;

  template <class... Types>
  struct variant_size<variant<Types...>>;

  template <size_t I, class T> struct variant_alternative; // undefined

  template <size_t I, class T>
  using variant_alternative_t = typename variant_alternative<I, T>::type;

  template <size_t I, class T> struct variant_alternative<I, const T>;
  template <size_t I, class T> struct variant_alternative<I, volatile T>;
  template <size_t I, class T> struct variant_alternative<I, const volatile T>;

  template <size_t I, class... Types>
  struct variant_alternative<I, variant<Types...>>;

  constexpr size_t variant_npos = -1;

  // 20.7.4, value access
  template <class T, class... Types>
  constexpr bool holds_alternative(const variant<Types...>&) noexcept;

  template <size_t I, class... Types>
  constexpr variant_alternative_t<I, variant<Types...>>&
  get(variant<Types...>&);

  template <size_t I, class... Types>
  constexpr variant_alternative_t<I, variant<Types...>>&&
  get(variant<Types...>&&);

  template <size_t I, class... Types>
  constexpr variant_alternative_t<I, variant<Types...>> const&
  get(const variant<Types...>&);

  template <size_t I, class... Types>
  constexpr variant_alternative_t<I, variant<Types...>> const&&
  get(const variant<Types...>&&);

  template <class T, class...  Types>
  constexpr T& get(variant<Types...>&);

  template <class T, class... Types>
  constexpr T&& get(variant<Types...>&&);

  template <class T, class... Types>
  constexpr const T& get(const variant<Types...>&);

  template <class T, class... Types>
  constexpr const T&& get(const variant<Types...>&&);

  template <size_t I, class... Types>
  constexpr add_pointer_t<variant_alternative_t<I, variant<Types...>>>
  get_if(variant<Types...>*) noexcept;

  template <size_t I, class... Types>
  constexpr add_pointer_t<const variant_alternative_t<I, variant<Types...>>>
  get_if(const variant<Types...>*) noexcept;

  template <class T, class... Types>
  constexpr add_pointer_t<T>
  get_if(variant<Types...>*) noexcept;

  template <class T, class... Types>
  constexpr add_pointer_t<const T>
  get_if(const variant<Types...>*) noexcept;

  // 20.7.5, relational operators
  template <class... Types>
  constexpr bool operator==(const variant<Types...>&, const variant<Types...>&);

  template <class... Types>
  constexpr bool operator!=(const variant<Types...>&, const variant<Types...>&);

  template <class... Types>
  constexpr bool operator<(const variant<Types...>&, const variant<Types...>&);

  template <class... Types>
  constexpr bool operator>(const variant<Types...>&, const variant<Types...>&);

  template <class... Types>
  constexpr bool operator<=(const variant<Types...>&, const variant<Types...>&);

  template <class... Types>
  constexpr bool operator>=(const variant<Types...>&, const variant<Types...>&);

  // 20.7.6, visitation
  template <class Visitor, class... Variants>
  constexpr see below visit(Visitor&&, Variants&&...);

  // 20.7.7, class monostate
  struct monostate;

  // 20.7.8, monostate relational operators
  constexpr bool operator<(monostate, monostate) noexcept;
  constexpr bool operator>(monostate, monostate) noexcept;
  constexpr bool operator<=(monostate, monostate) noexcept;
  constexpr bool operator>=(monostate, monostate) noexcept;
  constexpr bool operator==(monostate, monostate) noexcept;
  constexpr bool operator!=(monostate, monostate) noexcept;

  // 20.7.9, specialized algorithms
  template <class... Types>
  void swap(variant<Types...>&, variant<Types...>&) noexcept(see below);

  // 20.7.10, class bad_variant_access
  class bad_variant_access;

  // 20.7.11, hash support
  template <class T> struct hash;
  template <class... Types> struct hash<variant<Types...>>;
  template <> struct hash<monostate>;

} // namespace std

*/

#include <cstddef>
#include <exception>
#include <functional>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#ifndef MPARK_CONFIG_HPP
#define MPARK_CONFIG_HPP

// MSVC 2015 Update 3.
#if __cplusplus < 201103L && (!defined(_MSC_VER) || _MSC_FULL_VER < 190024210)
#error "MPark.Variant requires C++11 support."
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#ifndef __has_include
#define __has_include(x) 0
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_attribute(always_inline) || defined(__GNUC__)
#define MPARK_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#elif defined(_MSC_VER)
#define MPARK_ALWAYS_INLINE __forceinline
#else
#define MPARK_ALWAYS_INLINE inline
#endif

#if __has_builtin(__builtin_addressof) || \
    (defined(__GNUC__) && __GNUC__ >= 7) || defined(_MSC_VER)
#define MPARK_BUILTIN_ADDRESSOF
#endif

#if __has_builtin(__builtin_unreachable) || defined(__GNUC__)
#define MPARK_BUILTIN_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define MPARK_BUILTIN_UNREACHABLE __assume(false)
#else
#define MPARK_BUILTIN_UNREACHABLE
#endif

#if __has_builtin(__type_pack_element)
#define MPARK_TYPE_PACK_ELEMENT
#endif

#if defined(__cpp_constexpr) && __cpp_constexpr >= 200704 && \
    !(defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 9)
#define MPARK_CPP11_CONSTEXPR
#endif

#if defined(__cpp_constexpr) && __cpp_constexpr >= 201304
#define MPARK_CPP14_CONSTEXPR
#endif

#if __has_feature(cxx_exceptions) || defined(__cpp_exceptions) || \
    (defined(_MSC_VER) && defined(_CPPUNWIND))
#define MPARK_EXCEPTIONS
#endif

#if defined(__cpp_generic_lambdas) || defined(_MSC_VER)
#define MPARK_GENERIC_LAMBDAS
#endif

#if defined(__cpp_lib_integer_sequence)
#define MPARK_INTEGER_SEQUENCE
#endif

#if defined(__cpp_return_type_deduction) || defined(_MSC_VER)
#define MPARK_RETURN_TYPE_DEDUCTION
#endif

#if defined(__cpp_lib_transparent_operators) || defined(_MSC_VER)
#define MPARK_TRANSPARENT_OPERATORS
#endif

#if defined(__cpp_variable_templates) || defined(_MSC_VER)
#define MPARK_VARIABLE_TEMPLATES
#endif

#if !defined(__GLIBCXX__) || __has_include(<codecvt>) // >= libstdc++-5
#define MPARK_TRIVIALITY_TYPE_TRAITS
#define MPARK_INCOMPLETE_TYPE_TRAITS
#endif

#endif // MPARK_CONFIG_HPP

// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#ifndef MPARK_IN_PLACE_HPP
#define MPARK_IN_PLACE_HPP

#include <c10/util/in_place.h>

#include <cstddef>

namespace c10 {

#ifdef MPARK_VARIABLE_TEMPLATES
template <std::size_t I>
constexpr in_place_index_t<I> in_place_index{};

template <typename T>
constexpr in_place_type_t<T> in_place_type{};
#endif

} // namespace c10

#endif // MPARK_IN_PLACE_HPP

// MPark.Variant
//
// Copyright Michael Park, 2015-2017
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at
// http://boost.org/LICENSE_1_0.txt)

#ifndef MPARK_LIB_HPP
#define MPARK_LIB_HPP

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#define MPARK_RETURN(...)                                  \
  noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) { \
    return __VA_ARGS__;                                    \
  }

namespace c10 {
namespace lib {
template <typename T>
struct identity {
  using type = T;
};

inline namespace cpp14 {
template <typename T, std::size_t N>
struct array {
  constexpr const T& operator[](std::size_t index) const {
    return data[index];
  }

  T data[N == 0 ? 1 : N];
};

template <typename T>
using add_pointer_t = typename std::add_pointer<T>::type;

template <typename... Ts>
using common_type_t = typename std::common_type<Ts...>::type;

template <typename T>
using decay_t = typename std::decay<T>::type;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T>
using remove_const_t = typename std::remove_const<T>::type;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
inline constexpr T&& forward(remove_reference_t<T>& t) noexcept {
  return static_cast<T&&>(t);
}

template <typename T>
inline constexpr T&& forward(remove_reference_t<T>&& t) noexcept {
  static_assert(
      !std::is_lvalue_reference<T>::value,
      "can not forward an rvalue as an lvalue");
  return static_cast<T&&>(t);
}

template <typename T>
inline constexpr remove_reference_t<T>&& move(T&& t) noexcept {
  return static_cast<remove_reference_t<T>&&>(t);
}

#ifdef MPARK_INTEGER_SEQUENCE
using std::index_sequence;
using std::index_sequence_for;
using std::integer_sequence;
using std::make_index_sequence;
#else
template <typename T, T... Is>
struct integer_sequence {
  using value_type = T;
  static constexpr std::size_t size() noexcept {
    return sizeof...(Is);
  }
};

template <std::size_t... Is>
using index_sequence = integer_sequence<std::size_t, Is...>;

template <typename Lhs, typename Rhs>
struct make_index_sequence_concat;

template <std::size_t... Lhs, std::size_t... Rhs>
struct make_index_sequence_concat<
    index_sequence<Lhs...>,
    index_sequence<Rhs...>>
    : identity<index_sequence<Lhs..., (sizeof...(Lhs) + Rhs)...>> {};

template <std::size_t N>
struct make_index_sequence_impl;

template <std::size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;

template <std::size_t N>
struct make_index_sequence_impl : make_index_sequence_concat<
                                      make_index_sequence<N / 2>,
                                      make_index_sequence<N - (N / 2)>> {};

template <>
struct make_index_sequence_impl<0> : identity<index_sequence<>> {};

template <>
struct make_index_sequence_impl<1> : identity<index_sequence<0>> {};

template <typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
#endif

// <functional>
#ifdef MPARK_TRANSPARENT_OPERATORS
using equal_to = std::equal_to<>;
#else
struct equal_to {
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      MPARK_RETURN(lib::forward<Lhs>(lhs) == lib::forward<Rhs>(rhs))
};
#endif

#ifdef MPARK_TRANSPARENT_OPERATORS
using not_equal_to = std::not_equal_to<>;
#else
struct not_equal_to {
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      MPARK_RETURN(lib::forward<Lhs>(lhs) != lib::forward<Rhs>(rhs))
};
#endif

#ifdef MPARK_TRANSPARENT_OPERATORS
using less = std::less<>;
#else
struct less {
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      MPARK_RETURN(lib::forward<Lhs>(lhs) < lib::forward<Rhs>(rhs))
};
#endif

#ifdef MPARK_TRANSPARENT_OPERATORS
using greater = std::greater<>;
#else
struct greater {
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      MPARK_RETURN(lib::forward<Lhs>(lhs) > lib::forward<Rhs>(rhs))
};
#endif

#ifdef MPARK_TRANSPARENT_OPERATORS
using less_equal = std::less_equal<>;
#else
struct less_equal {
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      MPARK_RETURN(lib::forward<Lhs>(lhs) <= lib::forward<Rhs>(rhs))
};
#endif

#ifdef MPARK_TRANSPARENT_OPERATORS
using greater_equal = std::greater_equal<>;
#else
struct greater_equal {
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs&& lhs, Rhs&& rhs) const
      MPARK_RETURN(lib::forward<Lhs>(lhs) >= lib::forward<Rhs>(rhs))
};
#endif
} // namespace cpp14

inline namespace cpp17 {

// <type_traits>
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

template <typename...>
struct voider : identity<void> {};

template <typename... Ts>
using void_t = typename voider<Ts...>::type;

namespace detail_ {
namespace swappable {

using std::swap;

template <typename T>
struct is_swappable {
 private:
  template <
      typename U,
      typename = decltype(swap(std::declval<U&>(), std::declval<U&>()))>
  inline static std::true_type test(int);

  template <typename U>
  inline static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <bool IsSwappable, typename T>
struct is_nothrow_swappable {
  static constexpr bool value =
      noexcept(swap(std::declval<T&>(), std::declval<T&>()));
};

template <typename T>
struct is_nothrow_swappable<false, T> : std::false_type {};

} // namespace swappable
} // namespace detail_

using detail_::swappable::is_swappable;

template <typename T>
using is_nothrow_swappable =
    detail_::swappable::is_nothrow_swappable<is_swappable<T>::value, T>;

// <functional>
namespace detail_ {

template <typename T>
struct is_reference_wrapper : std::false_type {};

template <typename T>
struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {};

template <bool, int>
struct Invoke;

template <>
struct Invoke<true /* pmf */, 0 /* is_base_of */> {
  template <typename R, typename T, typename Arg, typename... Args>
  inline static constexpr auto invoke(R T::*pmf, Arg&& arg, Args&&... args)
      MPARK_RETURN((lib::forward<Arg>(arg).*pmf)(lib::forward<Args>(args)...))
};

template <>
struct Invoke<true /* pmf */, 1 /* is_reference_wrapper */> {
  template <typename R, typename T, typename Arg, typename... Args>
  inline static constexpr auto invoke(R T::*pmf, Arg&& arg, Args&&... args)
      MPARK_RETURN(
          (lib::forward<Arg>(arg).get().*pmf)(lib::forward<Args>(args)...))
};

template <>
struct Invoke<true /* pmf */, 2 /* otherwise */> {
  template <typename R, typename T, typename Arg, typename... Args>
  inline static constexpr auto invoke(R T::*pmf, Arg&& arg, Args&&... args)
      MPARK_RETURN(
          ((*lib::forward<Arg>(arg)).*pmf)(lib::forward<Args>(args)...))
};

template <>
struct Invoke<false /* pmo */, 0 /* is_base_of */> {
  template <typename R, typename T, typename Arg>
  inline static constexpr auto invoke(R T::*pmo, Arg&& arg)
      MPARK_RETURN(lib::forward<Arg>(arg).*pmo)
};

template <>
struct Invoke<false /* pmo */, 1 /* is_reference_wrapper */> {
  template <typename R, typename T, typename Arg>
  inline static constexpr auto invoke(R T::*pmo, Arg&& arg)
      MPARK_RETURN(lib::forward<Arg>(arg).get().*pmo)
};

template <>
struct Invoke<false /* pmo */, 2 /* otherwise */> {
  template <typename R, typename T, typename Arg>
  inline static constexpr auto invoke(R T::*pmo, Arg&& arg)
      MPARK_RETURN((*lib::forward<Arg>(arg)).*pmo)
};

template <typename R, typename T, typename Arg, typename... Args>
inline constexpr auto invoke(R T::*f, Arg&& arg, Args&&... args) MPARK_RETURN(
    Invoke<
        std::is_function<R>::value,
        (std::is_base_of<T, lib::decay_t<Arg>>::value         ? 0
             : is_reference_wrapper<lib::decay_t<Arg>>::value ? 1
                                                              : 2)>::
        invoke(f, lib::forward<Arg>(arg), lib::forward<Args>(args)...))

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
    template <typename F, typename... Args>
    inline constexpr auto invoke(F&& f, Args&&... args)
        MPARK_RETURN(lib::forward<F>(f)(lib::forward<Args>(args)...))
#ifdef _MSC_VER
#pragma warning(pop)
#endif
} // namespace detail_

template <typename F, typename... Args>
inline constexpr auto invoke(F&& f, Args&&... args) MPARK_RETURN(
    detail_::invoke(lib::forward<F>(f), lib::forward<Args>(args)...))

    namespace detail_ {
  template <typename Void, typename, typename...>
  struct invoke_result {};

  template <typename F, typename... Args>
  struct invoke_result<
      void_t<decltype(lib::invoke(std::declval<F>(), std::declval<Args>()...))>,
      F,
      Args...>
      : identity<decltype(lib::invoke(
            std::declval<F>(), std::declval<Args>()...))> {};

} // namespace detail_

template <typename F, typename... Args>
using invoke_result = detail_::invoke_result<void, F, Args...>;

template <typename F, typename... Args>
using invoke_result_t = typename invoke_result<F, Args...>::type;

namespace detail_ {

template <typename Void, typename, typename...>
struct is_invocable : std::false_type {};

template <typename F, typename... Args>
struct is_invocable<void_t<invoke_result_t<F, Args...>>, F, Args...>
    : std::true_type {};

template <typename Void, typename, typename, typename...>
struct is_invocable_r : std::false_type {};

template <typename R, typename F, typename... Args>
struct is_invocable_r<void_t<invoke_result_t<F, Args...>>, R, F, Args...>
    : std::is_convertible<invoke_result_t<F, Args...>, R> {};

} // namespace detail_

template <typename F, typename... Args>
using is_invocable = detail_::is_invocable<void, F, Args...>;

template <typename R, typename F, typename... Args>
using is_invocable_r = detail_::is_invocable_r<void, R, F, Args...>;

namespace detail_ {

template <bool Invocable, typename F, typename... Args>
struct is_nothrow_invocable {
  static constexpr bool value =
      noexcept(lib::invoke(std::declval<F>(), std::declval<Args>()...));
};

template <typename F, typename... Args>
struct is_nothrow_invocable<false, F, Args...> : std::false_type {};

template <bool Invocable, typename R, typename F, typename... Args>
struct is_nothrow_invocable_r {
 private:
  inline static R impl() {
    return lib::invoke(std::declval<F>(), std::declval<Args>()...);
  }

 public:
  static constexpr bool value = noexcept(impl());
};

template <typename R, typename F, typename... Args>
struct is_nothrow_invocable_r<false, R, F, Args...> : std::false_type {};

} // namespace detail_

template <typename F, typename... Args>
using is_nothrow_invocable =
    detail_::is_nothrow_invocable<is_invocable<F, Args...>::value, F, Args...>;

template <typename R, typename F, typename... Args>
using is_nothrow_invocable_r = detail_::
    is_nothrow_invocable_r<is_invocable_r<R, F, Args...>::value, R, F, Args...>;

// <memory>
#ifdef MPARK_BUILTIN_ADDRESSOF
template <typename T>
inline constexpr T* addressof(T& arg) noexcept {
  return __builtin_addressof(arg);
}
#else
namespace detail_ {

namespace has_addressof_impl {

struct fail;

template <typename T>
inline fail operator&(T&&);

template <typename T>
inline static constexpr bool impl() {
  return (std::is_class<T>::value || std::is_union<T>::value) &&
      !std::is_same<decltype(&std::declval<T&>()), fail>::value;
}

} // namespace has_addressof_impl

template <typename T>
using has_addressof = bool_constant<has_addressof_impl::impl<T>()>;

template <typename T>
inline constexpr T* addressof(T& arg, std::true_type) noexcept {
  return std::addressof(arg);
}

template <typename T>
inline constexpr T* addressof(T& arg, std::false_type) noexcept {
  return &arg;
}

} // namespace detail_

template <typename T>
inline constexpr T* addressof(T& arg) noexcept {
  return detail_::addressof(arg, detail_::has_addressof<T>{});
}
#endif

template <typename T>
inline constexpr T* addressof(const T&&) = delete;

} // namespace cpp17

template <typename T>
struct remove_all_extents : identity<T> {};

template <typename T, std::size_t N>
struct remove_all_extents<array<T, N>> : remove_all_extents<T> {};

template <typename T>
using remove_all_extents_t = typename remove_all_extents<T>::type;

template <std::size_t N>
using size_constant = std::integral_constant<std::size_t, N>;

template <std::size_t I, typename T>
struct indexed_type : size_constant<I> {
  using type = T;
};

template <bool... Bs>
using all = std::is_same<
    integer_sequence<bool, true, Bs...>,
    integer_sequence<bool, Bs..., true>>;

#ifdef MPARK_TYPE_PACK_ELEMENT
template <std::size_t I, typename... Ts>
using type_pack_element_t = __type_pack_element<I, Ts...>;
#else
template <std::size_t I, typename... Ts>
struct type_pack_element_impl {
 private:
  template <typename>
  struct set;

  template <std::size_t... Is>
  struct set<index_sequence<Is...>> : indexed_type<Is, Ts>... {};

  template <typename T>
  inline static std::enable_if<true, T> impl(indexed_type<I, T>);

  inline static std::enable_if<false> impl(...);

 public:
  using type = decltype(impl(set<index_sequence_for<Ts...>>{}));
};

template <std::size_t I, typename... Ts>
using type_pack_element = typename type_pack_element_impl<I, Ts...>::type;

template <std::size_t I, typename... Ts>
using type_pack_element_t = typename type_pack_element<I, Ts...>::type;
#endif

#ifdef MPARK_TRIVIALITY_TYPE_TRAITS
using std::is_trivially_copy_assignable;
using std::is_trivially_copy_constructible;
using std::is_trivially_move_assignable;
using std::is_trivially_move_constructible;
#else
template <typename T>
struct is_trivially_copy_constructible
    : bool_constant<std::is_copy_constructible<T>::value&& __has_trivial_copy(
          T)> {};

template <typename T>
struct is_trivially_move_constructible : bool_constant<__is_trivial(T)> {};

template <typename T>
struct is_trivially_copy_assignable
    : bool_constant<std::is_copy_assignable<T>::value&& __has_trivial_assign(
          T)> {};

template <typename T>
struct is_trivially_move_assignable : bool_constant<__is_trivial(T)> {};
#endif

template <typename T, bool>
struct dependent_type : T {};

template <typename Is, std::size_t J>
struct push_back;

template <typename Is, std::size_t J>
using push_back_t = typename push_back<Is, J>::type;

template <std::size_t... Is, std::size_t J>
struct push_back<index_sequence<Is...>, J> {
  using type = index_sequence<Is..., J>;
};

} // namespace lib
} // namespace c10

#undef MPARK_RETURN

#endif // MPARK_LIB_HPP

namespace c10 {

#ifdef MPARK_RETURN_TYPE_DEDUCTION

#define AUTO auto
#define AUTO_RETURN(...) \
  { return __VA_ARGS__; }

#define AUTO_REFREF auto&&
#define AUTO_REFREF_RETURN(...) \
  { return __VA_ARGS__; }

#define DECLTYPE_AUTO decltype(auto)
#define DECLTYPE_AUTO_RETURN(...) \
  { return __VA_ARGS__; }

#else

#define AUTO auto
#define AUTO_RETURN(...)                  \
  ->lib::decay_t<decltype(__VA_ARGS__)> { \
    return __VA_ARGS__;                   \
  }

#define AUTO_REFREF auto
#define AUTO_REFREF_RETURN(...)                                           \
  ->decltype((__VA_ARGS__)) {                                             \
    static_assert(std::is_reference<decltype((__VA_ARGS__))>::value, ""); \
    return __VA_ARGS__;                                                   \
  }

#define DECLTYPE_AUTO auto
#define DECLTYPE_AUTO_RETURN(...) \
  ->decltype(__VA_ARGS__) {       \
    return __VA_ARGS__;           \
  }

#endif

class bad_variant_access : public std::exception {
 public:
  const char* what() const noexcept override {
    return "bad_variant_access";
  }
};

[[noreturn]] inline void throw_bad_variant_access() {
#ifdef MPARK_EXCEPTIONS
  throw bad_variant_access{};
#else
  std::terminate();
  MPARK_BUILTIN_UNREACHABLE;
#endif
}

template <typename... Ts>
class variant;

template <typename T>
struct variant_size;

#ifdef MPARK_VARIABLE_TEMPLATES
template <typename T>
constexpr std::size_t variant_size_v = variant_size<T>::value;
#endif

template <typename T>
struct variant_size<const T> : variant_size<T> {};

template <typename T>
struct variant_size<volatile T> : variant_size<T> {};

template <typename T>
struct variant_size<const volatile T> : variant_size<T> {};

template <typename... Ts>
struct variant_size<variant<Ts...>> : lib::size_constant<sizeof...(Ts)> {};

template <std::size_t I, typename T>
struct variant_alternative;

template <std::size_t I, typename T>
using variant_alternative_t = typename variant_alternative<I, T>::type;

template <std::size_t I, typename T>
struct variant_alternative<I, const T>
    : std::add_const<variant_alternative_t<I, T>> {};

template <std::size_t I, typename T>
struct variant_alternative<I, volatile T>
    : std::add_volatile<variant_alternative_t<I, T>> {};

template <std::size_t I, typename T>
struct variant_alternative<I, const volatile T>
    : std::add_cv<variant_alternative_t<I, T>> {};

template <std::size_t I, typename... Ts>
struct variant_alternative<I, variant<Ts...>> {
  static_assert(
      I < sizeof...(Ts),
      "index out of bounds in `std::variant_alternative<>`");
  using type = lib::type_pack_element_t<I, Ts...>;
};

constexpr std::size_t variant_npos = static_cast<std::size_t>(-1);

namespace detail_ {

constexpr std::size_t not_found = static_cast<std::size_t>(-1);
constexpr std::size_t ambiguous = static_cast<std::size_t>(-2);

#ifdef MPARK_CPP14_CONSTEXPR
template <typename T, typename... Ts>
inline constexpr std::size_t find_index() {
  constexpr lib::array<bool, sizeof...(Ts)> matches = {
      {std::is_same<T, Ts>::value...}};
  std::size_t result = not_found;
  for (std::size_t i = 0; i < sizeof...(Ts); ++i) {
    if (matches[i]) {
      if (result != not_found) {
        return ambiguous;
      }
      result = i;
    }
  }
  return result;
}
#else
inline constexpr std::size_t find_index_impl(std::size_t result, std::size_t) {
  return result;
}

template <typename... Bs>
inline constexpr std::size_t find_index_impl(
    std::size_t result,
    std::size_t idx,
    bool b,
    Bs... bs) {
  return b
      ? (result != not_found ? ambiguous : find_index_impl(idx, idx + 1, bs...))
      : find_index_impl(result, idx + 1, bs...);
}

template <typename T, typename... Ts>
inline constexpr std::size_t find_index() {
  return find_index_impl(not_found, 0, std::is_same<T, Ts>::value...);
}
#endif

template <std::size_t I>
using find_index_sfinae_impl =
    lib::enable_if_t<I != not_found && I != ambiguous, lib::size_constant<I>>;

template <typename T, typename... Ts>
using find_index_sfinae = find_index_sfinae_impl<find_index<T, Ts...>()>;

template <std::size_t I>
struct find_index_checked_impl : lib::size_constant<I> {
  static_assert(I != not_found, "the specified type is not found.");
  static_assert(I != ambiguous, "the specified type is ambiguous.");
};

template <typename T, typename... Ts>
using find_index_checked = find_index_checked_impl<find_index<T, Ts...>()>;

struct valueless_t {};

enum class Trait { TriviallyAvailable, Available, Unavailable };

template <
    typename T,
    template <typename>
    class IsTriviallyAvailable,
    template <typename>
    class IsAvailable>
inline constexpr Trait trait() {
  return IsTriviallyAvailable<T>::value ? Trait::TriviallyAvailable
      : IsAvailable<T>::value           ? Trait::Available
                                        : Trait::Unavailable;
}

#ifdef MPARK_CPP14_CONSTEXPR
template <typename... Traits>
inline constexpr Trait common_trait(Traits... traits_) {
  Trait result = Trait::TriviallyAvailable;
  lib::array<Trait, sizeof...(Traits)> traits = {{traits_...}};
  for (std::size_t i = 0; i < sizeof...(Traits); ++i) {
    Trait t = traits[i];
    if (static_cast<int>(t) > static_cast<int>(result)) {
      result = t;
    }
  }
  return result;
}
#else
inline constexpr Trait common_trait_impl(Trait result) {
  return result;
}

template <typename... Traits>
inline constexpr Trait common_trait_impl(Trait result, Trait t, Traits... ts) {
  return static_cast<int>(t) > static_cast<int>(result)
      ? common_trait_impl(t, ts...)
      : common_trait_impl(result, ts...);
}

template <typename... Traits>
inline constexpr Trait common_trait(Traits... ts) {
  return common_trait_impl(Trait::TriviallyAvailable, ts...);
}
#endif

template <typename... Ts>
struct traits {
  static constexpr Trait copy_constructible_trait =
      common_trait(trait<
                   Ts,
                   lib::is_trivially_copy_constructible,
                   std::is_copy_constructible>()...);

  static constexpr Trait move_constructible_trait =
      common_trait(trait<
                   Ts,
                   lib::is_trivially_move_constructible,
                   std::is_move_constructible>()...);

  static constexpr Trait copy_assignable_trait = common_trait(
      copy_constructible_trait,
      trait<
          Ts,
          lib::is_trivially_copy_assignable,
          std::is_copy_assignable>()...);

  static constexpr Trait move_assignable_trait = common_trait(
      move_constructible_trait,
      trait<
          Ts,
          lib::is_trivially_move_assignable,
          std::is_move_assignable>()...);

  static constexpr Trait destructible_trait = common_trait(
      trait<Ts, std::is_trivially_destructible, std::is_destructible>()...);
};

namespace access {

struct recursive_union {
#ifdef MPARK_RETURN_TYPE_DEDUCTION
  template <typename V>
  inline static constexpr auto&& get_alt(V&& v, in_place_index_t<0>) {
    return lib::forward<V>(v).head_;
  }

  template <typename V, std::size_t I>
  inline static constexpr auto&& get_alt(V&& v, in_place_index_t<I>) {
    return get_alt(lib::forward<V>(v).tail_, in_place_index_t<I - 1>{});
  }
#else
  template <std::size_t I, bool Dummy = true>
  struct get_alt_impl {
    template <typename V>
    inline constexpr AUTO_REFREF operator()(V&& v) const
        AUTO_REFREF_RETURN(get_alt_impl<I - 1>{}(lib::forward<V>(v).tail_))
  };

  template <bool Dummy>
  struct get_alt_impl<0, Dummy> {
    template <typename V>
    inline constexpr AUTO_REFREF operator()(V&& v) const
        AUTO_REFREF_RETURN(lib::forward<V>(v).head_)
  };

  template <typename V, std::size_t I>
  inline static constexpr AUTO_REFREF get_alt(V&& v, in_place_index_t<I>)
      AUTO_REFREF_RETURN(get_alt_impl<I>{}(lib::forward<V>(v)))
#endif
};

struct base {
  template <std::size_t I, typename V>
  inline static constexpr AUTO_REFREF get_alt(V&& v)
#ifdef _MSC_VER
      AUTO_REFREF_RETURN(recursive_union::get_alt(
          lib::forward<V>(v).data_,
          in_place_index_t<I>{}))
#else
      AUTO_REFREF_RETURN(recursive_union::get_alt(
          data(lib::forward<V>(v)),
          in_place_index_t<I>{}))
#endif
};

struct variant {
  template <std::size_t I, typename V>
  inline static constexpr AUTO_REFREF get_alt(V&& v)
      AUTO_REFREF_RETURN(base::get_alt<I>(lib::forward<V>(v).impl_))
};

} // namespace access

namespace visitation {

#if defined(MPARK_CPP14_CONSTEXPR) && !defined(_MSC_VER)
#define MPARK_VARIANT_SWITCH_VISIT
#endif

struct base {
  template <typename Visitor, typename... Vs>
  using dispatch_result_t = decltype(lib::invoke(
      std::declval<Visitor>(),
      access::base::get_alt<0>(std::declval<Vs>())...));

  template <typename Expected>
  struct expected {
    template <typename Actual>
    inline static constexpr bool but_got() {
      return std::is_same<Expected, Actual>::value;
    }
  };

  template <typename Expected, typename Actual>
  struct visit_return_type_check {
    static_assert(
        expected<Expected>::template but_got<Actual>(),
        "`visit` requires the visitor to have a single return type");

    template <typename Visitor, typename... Alts>
    inline static constexpr DECLTYPE_AUTO invoke(
        Visitor&& visitor,
        Alts&&... alts)
        DECLTYPE_AUTO_RETURN(lib::invoke(
            lib::forward<Visitor>(visitor),
            lib::forward<Alts>(alts)...))
  };

#ifdef MPARK_VARIANT_SWITCH_VISIT
  template <bool B, typename R, typename... ITs>
  struct dispatcher;

  template <typename R, typename... ITs>
  struct dispatcher<false, R, ITs...> {
    template <std::size_t B, typename F, typename... Vs>
    MPARK_ALWAYS_INLINE static constexpr R dispatch(
        F&&,
        typename ITs::type&&...,
        Vs&&...) {
      MPARK_BUILTIN_UNREACHABLE;
    }

    template <std::size_t I, typename F, typename... Vs>
    MPARK_ALWAYS_INLINE static constexpr R dispatch_case(F&&, Vs&&...) {
      MPARK_BUILTIN_UNREACHABLE;
    }

    template <std::size_t B, typename F, typename... Vs>
    MPARK_ALWAYS_INLINE static constexpr R dispatch_at(
        std::size_t,
        F&&,
        Vs&&...) {
      MPARK_BUILTIN_UNREACHABLE;
    }
  };

  template <typename R, typename... ITs>
  struct dispatcher<true, R, ITs...> {
    template <std::size_t B, typename F>
    MPARK_ALWAYS_INLINE static constexpr R dispatch(
        F&& f,
        typename ITs::type&&... visited_vs) {
      using Expected = R;
      using Actual = decltype(lib::invoke(
          lib::forward<F>(f),
          access::base::get_alt<ITs::value>(
              lib::forward<typename ITs::type>(visited_vs))...));
      return visit_return_type_check<Expected, Actual>::invoke(
          lib::forward<F>(f),
          access::base::get_alt<ITs::value>(
              lib::forward<typename ITs::type>(visited_vs))...);
    }

    template <std::size_t B, typename F, typename V, typename... Vs>
    MPARK_ALWAYS_INLINE static constexpr R dispatch(
        F&& f,
        typename ITs::type&&... visited_vs,
        V&& v,
        Vs&&... vs) {
#define MPARK_DISPATCH(I)                                  \
  dispatcher<                                              \
      (I < lib::decay_t<V>::size()),                       \
      R,                                                   \
      ITs...,                                              \
      lib::indexed_type<I, V>>::                           \
      template dispatch<0>(                                \
          lib::forward<F>(f),                              \
          lib::forward<typename ITs::type>(visited_vs)..., \
          lib::forward<V>(v),                              \
          lib::forward<Vs>(vs)...)

#define MPARK_DEFAULT(I)                                                      \
  dispatcher<(I < lib::decay_t<V>::size()), R, ITs...>::template dispatch<I>( \
      lib::forward<F>(f),                                                     \
      lib::forward<typename ITs::type>(visited_vs)...,                        \
      lib::forward<V>(v),                                                     \
      lib::forward<Vs>(vs)...)

      switch (v.index()) {
        case B + 0:
          return MPARK_DISPATCH(B + 0);
        case B + 1:
          return MPARK_DISPATCH(B + 1);
        case B + 2:
          return MPARK_DISPATCH(B + 2);
        case B + 3:
          return MPARK_DISPATCH(B + 3);
        case B + 4:
          return MPARK_DISPATCH(B + 4);
        case B + 5:
          return MPARK_DISPATCH(B + 5);
        case B + 6:
          return MPARK_DISPATCH(B + 6);
        case B + 7:
          return MPARK_DISPATCH(B + 7);
        case B + 8:
          return MPARK_DISPATCH(B + 8);
        case B + 9:
          return MPARK_DISPATCH(B + 9);
        case B + 10:
          return MPARK_DISPATCH(B + 10);
        case B + 11:
          return MPARK_DISPATCH(B + 11);
        case B + 12:
          return MPARK_DISPATCH(B + 12);
        case B + 13:
          return MPARK_DISPATCH(B + 13);
        case B + 14:
          return MPARK_DISPATCH(B + 14);
        case B + 15:
          return MPARK_DISPATCH(B + 15);
        case B + 16:
          return MPARK_DISPATCH(B + 16);
        case B + 17:
          return MPARK_DISPATCH(B + 17);
        case B + 18:
          return MPARK_DISPATCH(B + 18);
        case B + 19:
          return MPARK_DISPATCH(B + 19);
        case B + 20:
          return MPARK_DISPATCH(B + 20);
        case B + 21:
          return MPARK_DISPATCH(B + 21);
        case B + 22:
          return MPARK_DISPATCH(B + 22);
        case B + 23:
          return MPARK_DISPATCH(B + 23);
        case B + 24:
          return MPARK_DISPATCH(B + 24);
        case B + 25:
          return MPARK_DISPATCH(B + 25);
        case B + 26:
          return MPARK_DISPATCH(B + 26);
        case B + 27:
          return MPARK_DISPATCH(B + 27);
        case B + 28:
          return MPARK_DISPATCH(B + 28);
        case B + 29:
          return MPARK_DISPATCH(B + 29);
        case B + 30:
          return MPARK_DISPATCH(B + 30);
        case B + 31:
          return MPARK_DISPATCH(B + 31);
        default:
          return MPARK_DEFAULT(B + 32);
      }

#undef MPARK_DEFAULT
#undef MPARK_DISPATCH
    }

    template <std::size_t I, typename F, typename... Vs>
    MPARK_ALWAYS_INLINE static constexpr R dispatch_case(F&& f, Vs&&... vs) {
      using Expected = R;
      using Actual = decltype(lib::invoke(
          lib::forward<F>(f),
          access::base::get_alt<I>(lib::forward<Vs>(vs))...));
      return visit_return_type_check<Expected, Actual>::invoke(
          lib::forward<F>(f),
          access::base::get_alt<I>(lib::forward<Vs>(vs))...);
    }

    template <std::size_t B, typename F, typename V, typename... Vs>
    MPARK_ALWAYS_INLINE static constexpr R dispatch_at(
        std::size_t index,
        F&& f,
        V&& v,
        Vs&&... vs) {
      static_assert(
          lib::all<(
              lib::decay_t<V>::size() == lib::decay_t<Vs>::size())...>::value,
          "all of the variants must be the same size.");
#define MPARK_DISPATCH_AT(I)                                               \
  dispatcher<(I < lib::decay_t<V>::size()), R>::template dispatch_case<I>( \
      lib::forward<F>(f), lib::forward<V>(v), lib::forward<Vs>(vs)...)

#define MPARK_DEFAULT(I)                                                 \
  dispatcher<(I < lib::decay_t<V>::size()), R>::template dispatch_at<I>( \
      index, lib::forward<F>(f), lib::forward<V>(v), lib::forward<Vs>(vs)...)

      switch (index) {
        case B + 0:
          return MPARK_DISPATCH_AT(B + 0);
        case B + 1:
          return MPARK_DISPATCH_AT(B + 1);
        case B + 2:
          return MPARK_DISPATCH_AT(B + 2);
        case B + 3:
          return MPARK_DISPATCH_AT(B + 3);
        case B + 4:
          return MPARK_DISPATCH_AT(B + 4);
        case B + 5:
          return MPARK_DISPATCH_AT(B + 5);
        case B + 6:
          return MPARK_DISPATCH_AT(B + 6);
        case B + 7:
          return MPARK_DISPATCH_AT(B + 7);
        case B + 8:
          return MPARK_DISPATCH_AT(B + 8);
        case B + 9:
          return MPARK_DISPATCH_AT(B + 9);
        case B + 10:
          return MPARK_DISPATCH_AT(B + 10);
        case B + 11:
          return MPARK_DISPATCH_AT(B + 11);
        case B + 12:
          return MPARK_DISPATCH_AT(B + 12);
        case B + 13:
          return MPARK_DISPATCH_AT(B + 13);
        case B + 14:
          return MPARK_DISPATCH_AT(B + 14);
        case B + 15:
          return MPARK_DISPATCH_AT(B + 15);
        case B + 16:
          return MPARK_DISPATCH_AT(B + 16);
        case B + 17:
          return MPARK_DISPATCH_AT(B + 17);
        case B + 18:
          return MPARK_DISPATCH_AT(B + 18);
        case B + 19:
          return MPARK_DISPATCH_AT(B + 19);
        case B + 20:
          return MPARK_DISPATCH_AT(B + 20);
        case B + 21:
          return MPARK_DISPATCH_AT(B + 21);
        case B + 22:
          return MPARK_DISPATCH_AT(B + 22);
        case B + 23:
          return MPARK_DISPATCH_AT(B + 23);
        case B + 24:
          return MPARK_DISPATCH_AT(B + 24);
        case B + 25:
          return MPARK_DISPATCH_AT(B + 25);
        case B + 26:
          return MPARK_DISPATCH_AT(B + 26);
        case B + 27:
          return MPARK_DISPATCH_AT(B + 27);
        case B + 28:
          return MPARK_DISPATCH_AT(B + 28);
        case B + 29:
          return MPARK_DISPATCH_AT(B + 29);
        case B + 30:
          return MPARK_DISPATCH_AT(B + 30);
        case B + 31:
          return MPARK_DISPATCH_AT(B + 31);
        default:
          return MPARK_DEFAULT(B + 32);
      }

#undef MPARK_DEFAULT
#undef MPARK_DISPATCH_AT
    }
  };
#else
  template <typename T>
  inline static constexpr const T& at(const T& elem) noexcept {
    return elem;
  }

  template <typename T, std::size_t N, typename... Is>
  inline static constexpr const lib::remove_all_extents_t<T>& at(
      const lib::array<T, N>& elems,
      std::size_t i,
      Is... is) noexcept {
    return at(elems[i], is...);
  }

  template <typename F, typename... Fs>
  inline static constexpr lib::array<lib::decay_t<F>, sizeof...(Fs) + 1>
  make_farray(F&& f, Fs&&... fs) {
    return {{lib::forward<F>(f), lib::forward<Fs>(fs)...}};
  }

  template <typename F, typename... Vs>
  struct make_fmatrix_impl {
    template <std::size_t... Is>
    inline static constexpr dispatch_result_t<F, Vs...> dispatch(
        F&& f,
        Vs&&... vs) {
      using Expected = dispatch_result_t<F, Vs...>;
      using Actual = decltype(lib::invoke(
          lib::forward<F>(f),
          access::base::get_alt<Is>(lib::forward<Vs>(vs))...));
      return visit_return_type_check<Expected, Actual>::invoke(
          lib::forward<F>(f),
          access::base::get_alt<Is>(lib::forward<Vs>(vs))...);
    }

#ifdef MPARK_RETURN_TYPE_DEDUCTION
    template <std::size_t... Is>
    inline static constexpr auto impl(lib::index_sequence<Is...>) {
      return &dispatch<Is...>;
    }

    template <typename Is, std::size_t... Js, typename... Ls>
    inline static constexpr auto impl(
        Is,
        lib::index_sequence<Js...>,
        Ls... ls) {
      return make_farray(impl(lib::push_back_t<Is, Js>{}, ls...)...);
    }
#else
    template <typename...>
    struct impl;

    template <std::size_t... Is>
    struct impl<lib::index_sequence<Is...>> {
      inline constexpr AUTO operator()() const AUTO_RETURN(&dispatch<Is...>)
    };

    template <typename Is, std::size_t... Js, typename... Ls>
    struct impl<Is, lib::index_sequence<Js...>, Ls...> {
      inline constexpr AUTO operator()() const
          AUTO_RETURN(make_farray(impl<lib::push_back_t<Is, Js>, Ls...>{}()...))
    };
#endif
  };

#ifdef MPARK_RETURN_TYPE_DEDUCTION
  template <typename F, typename... Vs>
  inline static constexpr auto make_fmatrix() {
    return make_fmatrix_impl<F, Vs...>::impl(
        lib::index_sequence<>{},
        lib::make_index_sequence<lib::decay_t<Vs>::size()>{}...);
  }
#else
  template <typename F, typename... Vs>
  inline static constexpr AUTO make_fmatrix()
      AUTO_RETURN(typename make_fmatrix_impl<F, Vs...>::template impl<
                  lib::index_sequence<>,
                  lib::make_index_sequence<lib::decay_t<Vs>::size()>...>{}())
#endif

  template <typename F, typename... Vs>
  struct make_fdiagonal_impl {
    template <std::size_t I>
    inline static constexpr dispatch_result_t<F, Vs...> dispatch(
        F&& f,
        Vs&&... vs) {
      using Expected = dispatch_result_t<F, Vs...>;
      using Actual = decltype(lib::invoke(
          lib::forward<F>(f),
          access::base::get_alt<I>(lib::forward<Vs>(vs))...));
      return visit_return_type_check<Expected, Actual>::invoke(
          lib::forward<F>(f),
          access::base::get_alt<I>(lib::forward<Vs>(vs))...);
    }

    template <std::size_t... Is>
    inline static constexpr AUTO impl(lib::index_sequence<Is...>)
        AUTO_RETURN(make_farray(&dispatch<Is>...))
  };

  template <typename F, typename V, typename... Vs>
  inline static constexpr auto make_fdiagonal()
      -> decltype(make_fdiagonal_impl<F, V, Vs...>::impl(
          lib::make_index_sequence<lib::decay_t<V>::size()>{})) {
    static_assert(
        lib::all<(
            lib::decay_t<V>::size() == lib::decay_t<Vs>::size())...>::value,
        "all of the variants must be the same size.");
    return make_fdiagonal_impl<F, V, Vs...>::impl(
        lib::make_index_sequence<lib::decay_t<V>::size()>{});
  }
#endif
};

#if !defined(MPARK_VARIANT_SWITCH_VISIT) && \
    (!defined(_MSC_VER) || _MSC_VER >= 1910)
template <typename F, typename... Vs>
using fmatrix_t = decltype(base::make_fmatrix<F, Vs...>());

template <typename F, typename... Vs>
struct fmatrix {
  static constexpr fmatrix_t<F, Vs...> value = base::make_fmatrix<F, Vs...>();
};

template <typename F, typename... Vs>
constexpr fmatrix_t<F, Vs...> fmatrix<F, Vs...>::value;

template <typename F, typename... Vs>
using fdiagonal_t = decltype(base::make_fdiagonal<F, Vs...>());

template <typename F, typename... Vs>
struct fdiagonal {
  static constexpr fdiagonal_t<F, Vs...> value =
      base::make_fdiagonal<F, Vs...>();
};

template <typename F, typename... Vs>
constexpr fdiagonal_t<F, Vs...> fdiagonal<F, Vs...>::value;
#endif

struct alt {
  template <typename Visitor, typename... Vs>
  inline static constexpr DECLTYPE_AUTO visit_alt(Visitor&& visitor, Vs&&... vs)
#ifdef MPARK_VARIANT_SWITCH_VISIT
      DECLTYPE_AUTO_RETURN(base::dispatcher<
                           true,
                           base::dispatch_result_t<
                               Visitor,
                               decltype(as_base(lib::forward<Vs>(vs)))...>>::
                               template dispatch<0>(
                                   lib::forward<Visitor>(visitor),
                                   as_base(lib::forward<Vs>(vs))...))
#elif !defined(_MSC_VER) || _MSC_VER >= 1910
      DECLTYPE_AUTO_RETURN(base::at(
          fmatrix<Visitor&&, decltype(as_base(lib::forward<Vs>(vs)))...>::value,
          vs.index()...)(
          lib::forward<Visitor>(visitor),
          as_base(lib::forward<Vs>(vs))...))
#else
      DECLTYPE_AUTO_RETURN(base::at(
          base::make_fmatrix<
              Visitor&&,
              decltype(as_base(lib::forward<Vs>(vs)))...>(),
          vs.index()...)(
          lib::forward<Visitor>(visitor),
          as_base(lib::forward<Vs>(vs))...))
#endif

          template <typename Visitor, typename... Vs>
          inline static constexpr DECLTYPE_AUTO
      visit_alt_at(std::size_t index, Visitor&& visitor, Vs&&... vs)
#ifdef MPARK_VARIANT_SWITCH_VISIT
          DECLTYPE_AUTO_RETURN(
              base::dispatcher<
                  true,
                  base::dispatch_result_t<
                      Visitor,
                      decltype(as_base(lib::forward<Vs>(vs)))...>>::
                  template dispatch_at<0>(
                      index,
                      lib::forward<Visitor>(visitor),
                      as_base(lib::forward<Vs>(vs))...))
#elif !defined(_MSC_VER) || _MSC_VER >= 1910
          DECLTYPE_AUTO_RETURN(base::at(
              fdiagonal<Visitor&&, decltype(as_base(lib::forward<Vs>(vs)))...>::
                  value,
              index)(
              lib::forward<Visitor>(visitor),
              as_base(lib::forward<Vs>(vs))...))
#else
          DECLTYPE_AUTO_RETURN(base::at(
              base::make_fdiagonal<
                  Visitor&&,
                  decltype(as_base(lib::forward<Vs>(vs)))...>(),
              index)(
              lib::forward<Visitor>(visitor),
              as_base(lib::forward<Vs>(vs))...))
#endif
};

struct variant {
 private:
  template <typename Visitor>
  struct visitor {
    template <typename... Values>
    inline static constexpr bool does_not_handle() {
      return lib::is_invocable<Visitor, Values...>::value;
    }
  };

  template <typename Visitor, typename... Values>
  struct visit_exhaustiveness_check {
    static_assert(
        visitor<Visitor>::template does_not_handle<Values...>(),
        "`visit` requires the visitor to be exhaustive.");

    inline static constexpr DECLTYPE_AUTO invoke(
        Visitor&& visitor,
        Values&&... values)
        DECLTYPE_AUTO_RETURN(lib::invoke(
            lib::forward<Visitor>(visitor),
            lib::forward<Values>(values)...))
  };

  template <typename Visitor>
  struct value_visitor {
    Visitor&& visitor_;

    template <typename... Alts>
    inline constexpr DECLTYPE_AUTO operator()(Alts&&... alts) const
        DECLTYPE_AUTO_RETURN(visit_exhaustiveness_check<
                             Visitor,
                             decltype((lib::forward<Alts>(alts).value))...>::
                                 invoke(
                                     lib::forward<Visitor>(visitor_),
                                     lib::forward<Alts>(alts).value...))
  };

  template <typename Visitor>
  inline static constexpr AUTO make_value_visitor(Visitor&& visitor)
      AUTO_RETURN(value_visitor<Visitor>{lib::forward<Visitor>(visitor)})

          public
      : template <typename Visitor, typename... Vs>
        inline static constexpr DECLTYPE_AUTO
        visit_alt(Visitor&& visitor, Vs&&... vs)
            DECLTYPE_AUTO_RETURN(alt::visit_alt(
                lib::forward<Visitor>(visitor),
                lib::forward<Vs>(vs).impl_...))

                template <typename Visitor, typename... Vs>
                inline static constexpr DECLTYPE_AUTO
        visit_alt_at(std::size_t index, Visitor&& visitor, Vs&&... vs)
            DECLTYPE_AUTO_RETURN(alt::visit_alt_at(
                index,
                lib::forward<Visitor>(visitor),
                lib::forward<Vs>(vs).impl_...))

                template <typename Visitor, typename... Vs>
                inline static constexpr DECLTYPE_AUTO
        visit_value(Visitor&& visitor, Vs&&... vs)
            DECLTYPE_AUTO_RETURN(visit_alt(
                make_value_visitor(lib::forward<Visitor>(visitor)),
                lib::forward<Vs>(vs)...))

                template <typename Visitor, typename... Vs>
                inline static constexpr DECLTYPE_AUTO
        visit_value_at(std::size_t index, Visitor&& visitor, Vs&&... vs)
            DECLTYPE_AUTO_RETURN(visit_alt_at(
                index,
                make_value_visitor(lib::forward<Visitor>(visitor)),
                lib::forward<Vs>(vs)...))
};

} // namespace visitation

template <std::size_t Index, typename T>
struct alt {
  using value_type = T;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
  template <typename... Args>
  inline explicit constexpr alt(in_place_t, Args&&... args)
      : value(lib::forward<Args>(args)...) {}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  T value;
};

template <Trait DestructibleTrait, std::size_t Index, typename... Ts>
union recursive_union;

template <Trait DestructibleTrait, std::size_t Index>
union recursive_union<DestructibleTrait, Index> {};

#define MPARK_VARIANT_RECURSIVE_UNION(destructible_trait, destructor)      \
  template <std::size_t Index, typename T, typename... Ts>                 \
  union recursive_union<destructible_trait, Index, T, Ts...> {             \
   public:                                                                 \
    inline explicit constexpr recursive_union(valueless_t) noexcept        \
        : dummy_{} {}                                                      \
                                                                           \
    template <typename... Args>                                            \
    inline explicit constexpr recursive_union(                             \
        in_place_index_t<0>,                                               \
        Args&&... args)                                                    \
        : head_(in_place_t{}, lib::forward<Args>(args)...) {}              \
                                                                           \
    template <std::size_t I, typename... Args>                             \
    inline explicit constexpr recursive_union(                             \
        in_place_index_t<I>,                                               \
        Args&&... args)                                                    \
        : tail_(in_place_index_t<I - 1>{}, lib::forward<Args>(args)...) {} \
                                                                           \
    recursive_union(const recursive_union&) = default;                     \
    recursive_union(recursive_union&&) = default;                          \
                                                                           \
    destructor                                                             \
                                                                           \
        recursive_union&                                                   \
        operator=(const recursive_union&) = default;                       \
    recursive_union& operator=(recursive_union&&) = default;               \
                                                                           \
   private:                                                                \
    char dummy_;                                                           \
    alt<Index, T> head_;                                                   \
    recursive_union<destructible_trait, Index + 1, Ts...> tail_;           \
                                                                           \
    friend struct access::recursive_union;                                 \
  }

MPARK_VARIANT_RECURSIVE_UNION(Trait::TriviallyAvailable,
                              ~recursive_union() = default;);
MPARK_VARIANT_RECURSIVE_UNION(Trait::Available, ~recursive_union(){});
MPARK_VARIANT_RECURSIVE_UNION(Trait::Unavailable, ~recursive_union() = delete;);

#undef MPARK_VARIANT_RECURSIVE_UNION

using index_t = unsigned int;

template <Trait DestructibleTrait, typename... Ts>
class base {
 public:
  inline explicit constexpr base(valueless_t tag) noexcept
      : data_(tag), index_(static_cast<index_t>(-1)) {}

  template <std::size_t I, typename... Args>
  inline explicit constexpr base(in_place_index_t<I>, Args&&... args)
      : data_(in_place_index_t<I>{}, lib::forward<Args>(args)...), index_(I) {}

  inline constexpr bool valueless_by_exception() const noexcept {
    return index_ == static_cast<index_t>(-1);
  }

  inline constexpr std::size_t index() const noexcept {
    return valueless_by_exception() ? variant_npos : index_;
  }

 protected:
  using data_t = recursive_union<DestructibleTrait, 0, Ts...>;

  friend inline constexpr base& as_base(base& b) {
    return b;
  }
  friend inline constexpr const base& as_base(const base& b) {
    return b;
  }
  friend inline constexpr base&& as_base(base&& b) {
    return lib::move(b);
  }
  friend inline constexpr const base&& as_base(const base&& b) {
    return lib::move(b);
  }

  friend inline constexpr data_t& data(base& b) {
    return b.data_;
  }
  friend inline constexpr const data_t& data(const base& b) {
    return b.data_;
  }
  friend inline constexpr data_t&& data(base&& b) {
    return lib::move(b).data_;
  }
  friend inline constexpr const data_t&& data(const base&& b) {
    return lib::move(b).data_;
  }

  inline static constexpr std::size_t size() {
    return sizeof...(Ts);
  }

  data_t data_;
  index_t index_;

  friend struct access::base;
  friend struct visitation::base;
};

struct dtor {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
  template <typename Alt>
  inline void operator()(Alt& alt) const noexcept {
    alt.~Alt();
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
};

#if !defined(_MSC_VER) || _MSC_VER >= 1910
#define MPARK_INHERITING_CTOR(type, base) using base::base;
#else
#define MPARK_INHERITING_CTOR(type, base)        \
  template <typename... Args>                    \
  inline explicit constexpr type(Args&&... args) \
      : base(lib::forward<Args>(args)...) {}
#endif

template <typename Traits, Trait = Traits::destructible_trait>
class destructor;

#define MPARK_VARIANT_DESTRUCTOR(destructible_trait, definition, destroy) \
  template <typename... Ts>                                               \
  class destructor<traits<Ts...>, destructible_trait>                     \
      : public base<destructible_trait, Ts...> {                          \
    using super = base<destructible_trait, Ts...>;                        \
                                                                          \
   public:                                                                \
    MPARK_INHERITING_CTOR(destructor, super)                              \
    using super::operator=;                                               \
                                                                          \
    destructor(const destructor&) = default;                              \
    destructor(destructor&&) = default;                                   \
    definition destructor& operator=(const destructor&) = default;        \
    destructor& operator=(destructor&&) = default;                        \
                                                                          \
   protected:                                                             \
    destroy                                                               \
  }

MPARK_VARIANT_DESTRUCTOR(
    Trait::TriviallyAvailable, ~destructor() = default;
    , inline void destroy() noexcept {
      this->index_ = static_cast<index_t>(-1);
    });

MPARK_VARIANT_DESTRUCTOR(
    Trait::Available,
    ~destructor() { destroy(); },
    inline void destroy() noexcept {
      if (!this->valueless_by_exception()) {
        visitation::alt::visit_alt(dtor{}, *this);
      }
      this->index_ = static_cast<index_t>(-1);
    });

MPARK_VARIANT_DESTRUCTOR(Trait::Unavailable, ~destructor() = delete;
                         , inline void destroy() noexcept = delete;);

#undef MPARK_VARIANT_DESTRUCTOR

template <typename Traits>
class constructor : public destructor<Traits> {
  using super = destructor<Traits>;

 public:
  MPARK_INHERITING_CTOR(constructor, super)
  using super::operator=;

 protected:
#ifndef MPARK_GENERIC_LAMBDAS
  struct ctor {
    template <typename LhsAlt, typename RhsAlt>
    inline void operator()(LhsAlt& lhs_alt, RhsAlt&& rhs_alt) const {
      constructor::construct_alt(lhs_alt, lib::forward<RhsAlt>(rhs_alt).value);
    }
  };
#endif

  template <std::size_t I, typename T, typename... Args>
  inline static T& construct_alt(alt<I, T>& a, Args&&... args) {
    auto* result = ::new (static_cast<void*>(lib::addressof(a)))
        alt<I, T>(in_place_t{}, lib::forward<Args>(args)...);
    return result->value;
  }

  template <typename Rhs>
  inline static void generic_construct(constructor& lhs, Rhs&& rhs) {
    lhs.destroy();
    if (!rhs.valueless_by_exception()) {
      visitation::alt::visit_alt_at(
          rhs.index(),
#ifdef MPARK_GENERIC_LAMBDAS
          [](auto& lhs_alt, auto&& rhs_alt) {
            constructor::construct_alt(
                lhs_alt, lib::forward<decltype(rhs_alt)>(rhs_alt).value);
          }
#else
          ctor {}
#endif
          ,
          lhs,
          lib::forward<Rhs>(rhs));
      lhs.index_ = rhs.index_;
    }
  }
};

template <typename Traits, Trait = Traits::move_constructible_trait>
class move_constructor;

#define MPARK_VARIANT_MOVE_CONSTRUCTOR(move_constructible_trait, definition) \
  template <typename... Ts>                                                  \
  class move_constructor<traits<Ts...>, move_constructible_trait>            \
      : public constructor<traits<Ts...>> {                                  \
    using super = constructor<traits<Ts...>>;                                \
                                                                             \
   public:                                                                   \
    MPARK_INHERITING_CTOR(move_constructor, super)                           \
    using super::operator=;                                                  \
                                                                             \
    move_constructor(const move_constructor&) = default;                     \
    definition ~move_constructor() = default;                                \
    move_constructor& operator=(const move_constructor&) = default;          \
    move_constructor& operator=(move_constructor&&) = default;               \
  }

MPARK_VARIANT_MOVE_CONSTRUCTOR(
    Trait::TriviallyAvailable,
    move_constructor(move_constructor&& that) = default;);

MPARK_VARIANT_MOVE_CONSTRUCTOR(
    Trait::Available,
    move_constructor(move_constructor&& that) noexcept(
        lib::all<std::is_nothrow_move_constructible<Ts>::value...>::value)
    : move_constructor(valueless_t{}) {
      this->generic_construct(*this, lib::move(that));
    });

MPARK_VARIANT_MOVE_CONSTRUCTOR(Trait::Unavailable,
                               move_constructor(move_constructor&&) = delete;);

#undef MPARK_VARIANT_MOVE_CONSTRUCTOR

template <typename Traits, Trait = Traits::copy_constructible_trait>
class copy_constructor;

#define MPARK_VARIANT_COPY_CONSTRUCTOR(copy_constructible_trait, definition) \
  template <typename... Ts>                                                  \
  class copy_constructor<traits<Ts...>, copy_constructible_trait>            \
      : public move_constructor<traits<Ts...>> {                             \
    using super = move_constructor<traits<Ts...>>;                           \
                                                                             \
   public:                                                                   \
    MPARK_INHERITING_CTOR(copy_constructor, super)                           \
    using super::operator=;                                                  \
                                                                             \
    definition copy_constructor(copy_constructor&&) = default;               \
    ~copy_constructor() = default;                                           \
    copy_constructor& operator=(const copy_constructor&) = default;          \
    copy_constructor& operator=(copy_constructor&&) = default;               \
  }

MPARK_VARIANT_COPY_CONSTRUCTOR(
    Trait::TriviallyAvailable,
    copy_constructor(const copy_constructor& that) = default;);

MPARK_VARIANT_COPY_CONSTRUCTOR(
    Trait::Available, copy_constructor(const copy_constructor& that)
    : copy_constructor(valueless_t{}) {
      this->generic_construct(*this, that);
    });

MPARK_VARIANT_COPY_CONSTRUCTOR(
    Trait::Unavailable, copy_constructor(const copy_constructor&) = delete;);

#undef MPARK_VARIANT_COPY_CONSTRUCTOR

template <typename Traits>
class assignment : public copy_constructor<Traits> {
  using super = copy_constructor<Traits>;

 public:
  MPARK_INHERITING_CTOR(assignment, super)
  using super::operator=;

  template <std::size_t I, typename... Args>
  inline /* auto & */ auto emplace(Args&&... args)
      -> decltype(this->construct_alt(
          access::base::get_alt<I>(*this),
          lib::forward<Args>(args)...)) {
    this->destroy();
    auto& result = this->construct_alt(
        access::base::get_alt<I>(*this), lib::forward<Args>(args)...);
    this->index_ = I;
    return result;
  }

 protected:
#ifndef MPARK_GENERIC_LAMBDAS
  template <typename That>
  struct assigner {
    template <typename ThisAlt, typename ThatAlt>
    inline void operator()(ThisAlt& this_alt, ThatAlt&& that_alt) const {
      self->assign_alt(this_alt, lib::forward<ThatAlt>(that_alt).value);
    }
    assignment* self;
  };
#endif

  template <std::size_t I, typename T, typename Arg>
  inline void assign_alt(alt<I, T>& a, Arg&& arg) {
    if (this->index() == I) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
      a.value = lib::forward<Arg>(arg);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    } else {
      struct {
        void operator()(std::true_type) const {
          this_->emplace<I>(lib::forward<Arg>(arg_));
        }
        void operator()(std::false_type) const {
          this_->emplace<I>(T(lib::forward<Arg>(arg_)));
        }
        assignment* this_;
        Arg&& arg_;
      } impl{this, lib::forward<Arg>(arg)};
      impl(
          lib::bool_constant < std::is_nothrow_constructible<T, Arg>::value ||
          !std::is_nothrow_move_constructible<T>::value > {});
    }
  }

  template <typename That>
  inline void generic_assign(That&& that) {
    if (this->valueless_by_exception() && that.valueless_by_exception()) {
      // do nothing.
    } else if (that.valueless_by_exception()) {
      this->destroy();
    } else {
      visitation::alt::visit_alt_at(
          that.index(),
#ifdef MPARK_GENERIC_LAMBDAS
          [this](auto& this_alt, auto&& that_alt) {
            this->assign_alt(
                this_alt, lib::forward<decltype(that_alt)>(that_alt).value);
          }
#else
          assigner<That> { this }
#endif
          ,
          *this,
          lib::forward<That>(that));
    }
  }
};

template <typename Traits, Trait = Traits::move_assignable_trait>
class move_assignment;

#define MPARK_VARIANT_MOVE_ASSIGNMENT(move_assignable_trait, definition) \
  template <typename... Ts>                                              \
  class move_assignment<traits<Ts...>, move_assignable_trait>            \
      : public assignment<traits<Ts...>> {                               \
    using super = assignment<traits<Ts...>>;                             \
                                                                         \
   public:                                                               \
    MPARK_INHERITING_CTOR(move_assignment, super)                        \
    using super::operator=;                                              \
                                                                         \
    move_assignment(const move_assignment&) = default;                   \
    move_assignment(move_assignment&&) = default;                        \
    ~move_assignment() = default;                                        \
    move_assignment& operator=(const move_assignment&) = default;        \
    definition                                                           \
  }

MPARK_VARIANT_MOVE_ASSIGNMENT(
    Trait::TriviallyAvailable,
    move_assignment& operator=(move_assignment&& that) = default;);

MPARK_VARIANT_MOVE_ASSIGNMENT(
    Trait::Available,
    move_assignment&
    operator=(move_assignment&& that) noexcept(
        lib::all<
            (std::is_nothrow_move_constructible<Ts>::value &&
             std::is_nothrow_move_assignable<Ts>::value)...>::value) {
      this->generic_assign(lib::move(that));
      return *this;
    });

MPARK_VARIANT_MOVE_ASSIGNMENT(
    Trait::Unavailable,
    move_assignment& operator=(move_assignment&&) = delete;);

#undef MPARK_VARIANT_MOVE_ASSIGNMENT

template <typename Traits, Trait = Traits::copy_assignable_trait>
class copy_assignment;

#define MPARK_VARIANT_COPY_ASSIGNMENT(copy_assignable_trait, definition) \
  template <typename... Ts>                                              \
  class copy_assignment<traits<Ts...>, copy_assignable_trait>            \
      : public move_assignment<traits<Ts...>> {                          \
    using super = move_assignment<traits<Ts...>>;                        \
                                                                         \
   public:                                                               \
    MPARK_INHERITING_CTOR(copy_assignment, super)                        \
    using super::operator=;                                              \
                                                                         \
    copy_assignment(const copy_assignment&) = default;                   \
    copy_assignment(copy_assignment&&) = default;                        \
    ~copy_assignment() = default;                                        \
    definition copy_assignment& operator=(copy_assignment&&) = default;  \
  }

MPARK_VARIANT_COPY_ASSIGNMENT(
    Trait::TriviallyAvailable,
    copy_assignment& operator=(const copy_assignment& that) = default;);

MPARK_VARIANT_COPY_ASSIGNMENT(
    Trait::Available,
    copy_assignment&
    operator=(const copy_assignment& that) {
      this->generic_assign(that);
      return *this;
    });

MPARK_VARIANT_COPY_ASSIGNMENT(
    Trait::Unavailable,
    copy_assignment& operator=(const copy_assignment&) = delete;);

#undef MPARK_VARIANT_COPY_ASSIGNMENT

template <typename... Ts>
class impl : public copy_assignment<traits<Ts...>> {
  using super = copy_assignment<traits<Ts...>>;

 public:
  MPARK_INHERITING_CTOR(impl, super)
  impl& operator=(const impl& other) = default;

  template <std::size_t I, typename Arg>
  inline void assign(Arg&& arg) {
    this->assign_alt(access::base::get_alt<I>(*this), lib::forward<Arg>(arg));
  }

  inline void swap(impl& that) {
    if (this->valueless_by_exception() && that.valueless_by_exception()) {
      // do nothing.
    } else if (this->index() == that.index()) {
      visitation::alt::visit_alt_at(
          this->index(),
#ifdef MPARK_GENERIC_LAMBDAS
          [](auto& this_alt, auto& that_alt) {
            using std::swap;
            swap(this_alt.value, that_alt.value);
          }
#else
          swapper {}
#endif
          ,
          *this,
          that);
    } else {
      impl* lhs = this;
      impl* rhs = lib::addressof(that);
      if (lhs->move_nothrow() && !rhs->move_nothrow()) {
        std::swap(lhs, rhs);
      }
      impl tmp(lib::move(*rhs));
#ifdef MPARK_EXCEPTIONS
      // EXTENSION: When the move construction of `lhs` into `rhs` throws
      // and `tmp` is nothrow move constructible then we move `tmp` back
      // into `rhs` and provide the strong exception safety guarantee.
      try {
        this->generic_construct(*rhs, lib::move(*lhs));
      } catch (...) {
        if (tmp.move_nothrow()) {
          this->generic_construct(*rhs, lib::move(tmp));
        }
        throw;
      }
#else
      this->generic_construct(*rhs, lib::move(*lhs));
#endif
      this->generic_construct(*lhs, lib::move(tmp));
    }
  }

 private:
#ifndef MPARK_GENERIC_LAMBDAS
  struct swapper {
    template <typename ThisAlt, typename ThatAlt>
    inline void operator()(ThisAlt& this_alt, ThatAlt& that_alt) const {
      using std::swap;
      swap(this_alt.value, that_alt.value);
    }
  };
#endif

  inline constexpr bool move_nothrow() const {
    return this->valueless_by_exception() ||
        lib::array<bool, sizeof...(Ts)>{
            {std::is_nothrow_move_constructible<Ts>::value...}}[this->index()];
  }
};

#undef MPARK_INHERITING_CTOR

template <std::size_t I, typename T>
struct overload_leaf {
  using F = lib::size_constant<I> (*)(T);
  operator F() const {
    return nullptr;
  }
};

template <typename... Ts>
struct overload_impl {
 private:
  template <typename>
  struct impl;

  template <std::size_t... Is>
  struct impl<lib::index_sequence<Is...>> : overload_leaf<Is, Ts>... {};

 public:
  using type = impl<lib::index_sequence_for<Ts...>>;
};

template <typename... Ts>
using overload = typename overload_impl<Ts...>::type;

template <typename T, typename... Ts>
using best_match = lib::invoke_result_t<overload<Ts...>, T&&>;

template <typename T>
struct is_in_place_index : std::false_type {};

template <std::size_t I>
struct is_in_place_index<in_place_index_t<I>> : std::true_type {};

template <typename T>
struct is_in_place_type : std::false_type {};

template <typename T>
struct is_in_place_type<in_place_type_t<T>> : std::true_type {};

} // namespace detail_

template <typename... Ts>
class variant {
  static_assert(
      0 < sizeof...(Ts),
      "variant must consist of at least one alternative.");

  static_assert(
      lib::all<!std::is_array<Ts>::value...>::value,
      "variant can not have an array type as an alternative.");

  static_assert(
      lib::all<!std::is_reference<Ts>::value...>::value,
      "variant can not have a reference type as an alternative.");

  static_assert(
      lib::all<!std::is_void<Ts>::value...>::value,
      "variant can not have a void type as an alternative.");

 public:
  template <
      typename Front = lib::type_pack_element_t<0, Ts...>,
      lib::enable_if_t<std::is_default_constructible<Front>::value, int> = 0>
  inline constexpr variant() noexcept(
      std::is_nothrow_default_constructible<Front>::value)
      : impl_(in_place_index_t<0>{}) {}

  variant(const variant&) = default;
  variant(variant&&) = default;

  // NOTE [gcc 7.3.1 bug workaround]
  //
  // The original line `typename T = lib::type_pack_element_t<I, Ts...>`
  // throws the following compiler error on gcc 7.3.1:
  // ```
  // ../c10/util/variant.h:2250:9: internal compiler error:
  // unexpected expression ‘I’ of kind template_parm_index
  //          typename T = lib::type_pack_element_t<I, Ts...>,
  //          ^~~~~~~~
  // ```
  // As a workaround, `I` is changed to `detail_::best_match<Arg,
  // Ts...>::value`, which is the default value for `I` in this template. Note
  // that this workaround effectively disallows setting `I` to any other
  // non-default value, and we add a `static_assert` in the function body to
  // check for this.
  //
  // See the following issues for more context:
  // - https://github.com/mpark/variant/issues/43
  // - https://github.com/eggs-cpp/variant/issues/31
  template <
      typename Arg,
      typename Decayed = lib::decay_t<Arg>,
      lib::enable_if_t<!std::is_same<Decayed, variant>::value, int> = 0,
      lib::enable_if_t<!detail_::is_in_place_index<Decayed>::value, int> = 0,
      lib::enable_if_t<!detail_::is_in_place_type<Decayed>::value, int> = 0,
      std::size_t I = detail_::best_match<Arg, Ts...>::value,
      typename T = lib::
          type_pack_element_t<detail_::best_match<Arg, Ts...>::value, Ts...>,
      lib::enable_if_t<std::is_constructible<T, Arg>::value, int> = 0>
  inline constexpr variant(Arg&& arg) noexcept(
      std::is_nothrow_constructible<T, Arg>::value)
      : impl_(in_place_index_t<I>{}, lib::forward<Arg>(arg)) {
    static_assert(
        I == detail_::best_match<Arg, Ts...>::value,
        "Setting template parameter `I` to a custom non-default value is not supported. "
        "Please file a feature request if you see this.");
  }

  template <
      std::size_t I,
      typename... Args,
      typename T = lib::type_pack_element_t<I, Ts...>,
      lib::enable_if_t<std::is_constructible<T, Args...>::value, int> = 0>
  inline explicit constexpr variant(
      in_place_index_t<I>,
      Args&&... args) noexcept(std::is_nothrow_constructible<T, Args...>::value)
      : impl_(in_place_index_t<I>{}, lib::forward<Args>(args)...) {}

  template <
      std::size_t I,
      typename Up,
      typename... Args,
      typename T = lib::type_pack_element_t<I, Ts...>,
      lib::enable_if_t<
          std::is_constructible<T, std::initializer_list<Up>&, Args...>::value,
          int> = 0>
  inline explicit constexpr variant(
      in_place_index_t<I>,
      std::initializer_list<Up> il,
      Args&&... args) noexcept(std::
                                   is_nothrow_constructible<
                                       T,
                                       std::initializer_list<Up>&,
                                       Args...>::value)
      : impl_(in_place_index_t<I>{}, il, lib::forward<Args>(args)...) {}

  template <
      typename T,
      typename... Args,
      std::size_t I = detail_::find_index_sfinae<T, Ts...>::value,
      lib::enable_if_t<std::is_constructible<T, Args...>::value, int> = 0>
  inline explicit constexpr variant(
      in_place_type_t<T>,
      Args&&... args) noexcept(std::is_nothrow_constructible<T, Args...>::value)
      : impl_(in_place_index_t<I>{}, lib::forward<Args>(args)...) {}

  template <
      typename T,
      typename Up,
      typename... Args,
      std::size_t I = detail_::find_index_sfinae<T, Ts...>::value,
      lib::enable_if_t<
          std::is_constructible<T, std::initializer_list<Up>&, Args...>::value,
          int> = 0>
  inline explicit constexpr variant(
      in_place_type_t<T>,
      std::initializer_list<Up> il,
      Args&&... args) noexcept(std::
                                   is_nothrow_constructible<
                                       T,
                                       std::initializer_list<Up>&,
                                       Args...>::value)
      : impl_(in_place_index_t<I>{}, il, lib::forward<Args>(args)...) {}

  ~variant() = default;

  variant& operator=(const variant&) = default;
  variant& operator=(variant&&) = default;

  // NOTE: See NOTE [gcc 7.3.1 bug workaround] for the changes made to this
  // function.
  template <
      typename Arg,
      lib::enable_if_t<!std::is_same<lib::decay_t<Arg>, variant>::value, int> =
          0,
      std::size_t I = detail_::best_match<Arg, Ts...>::value,
      typename T = lib::
          type_pack_element_t<detail_::best_match<Arg, Ts...>::value, Ts...>,
      lib::enable_if_t<
          (std::is_assignable<T&, Arg>::value &&
           std::is_constructible<T, Arg>::value),
          int> = 0>
  inline variant& operator=(Arg&& arg) noexcept(
      (std::is_nothrow_assignable<T&, Arg>::value &&
       std::is_nothrow_constructible<T, Arg>::value)) {
    static_assert(
        I == detail_::best_match<Arg, Ts...>::value,
        "Setting template parameter `I` to a custom non-default value is not supported. "
        "Please file a feature request if you see this.");
    impl_.template assign<I>(lib::forward<Arg>(arg));
    return *this;
  }

  template <
      std::size_t I,
      typename... Args,
      typename T = lib::type_pack_element_t<I, Ts...>,
      lib::enable_if_t<std::is_constructible<T, Args...>::value, int> = 0>
  inline T& emplace(Args&&... args) {
    return impl_.template emplace<I>(lib::forward<Args>(args)...);
  }

  template <
      std::size_t I,
      typename Up,
      typename... Args,
      typename T = lib::type_pack_element_t<I, Ts...>,
      lib::enable_if_t<
          std::is_constructible<T, std::initializer_list<Up>&, Args...>::value,
          int> = 0>
  inline T& emplace(std::initializer_list<Up> il, Args&&... args) {
    return impl_.template emplace<I>(il, lib::forward<Args>(args)...);
  }

  template <
      typename T,
      typename... Args,
      std::size_t I = detail_::find_index_sfinae<T, Ts...>::value,
      lib::enable_if_t<std::is_constructible<T, Args...>::value, int> = 0>
  inline T& emplace(Args&&... args) {
    return impl_.template emplace<I>(lib::forward<Args>(args)...);
  }

  template <
      typename T,
      typename Up,
      typename... Args,
      std::size_t I = detail_::find_index_sfinae<T, Ts...>::value,
      lib::enable_if_t<
          std::is_constructible<T, std::initializer_list<Up>&, Args...>::value,
          int> = 0>
  inline T& emplace(std::initializer_list<Up> il, Args&&... args) {
    return impl_.template emplace<I>(il, lib::forward<Args>(args)...);
  }

  inline constexpr bool valueless_by_exception() const noexcept {
    return impl_.valueless_by_exception();
  }

  inline constexpr std::size_t index() const noexcept {
    return impl_.index();
  }

  template <
      bool Dummy = true,
      lib::enable_if_t<
          lib::all<
              Dummy,
              (lib::dependent_type<std::is_move_constructible<Ts>, Dummy>::
                   value &&
               lib::dependent_type<lib::is_swappable<Ts>, Dummy>::value)...>::
              value,
          int> = 0>
  inline void swap(variant& that) noexcept(
      lib::all<
          (std::is_nothrow_move_constructible<Ts>::value &&
           lib::is_nothrow_swappable<Ts>::value)...>::value) {
    impl_.swap(that.impl_);
  }

 private:
  detail_::impl<Ts...> impl_;

  friend struct detail_::access::variant;
  friend struct detail_::visitation::variant;
};

template <std::size_t I, typename... Ts>
inline constexpr bool holds_alternative(const variant<Ts...>& v) noexcept {
  return v.index() == I;
}

template <typename T, typename... Ts>
inline constexpr bool holds_alternative(const variant<Ts...>& v) noexcept {
  return holds_alternative<detail_::find_index_checked<T, Ts...>::value>(v);
}

namespace detail_ {
template <std::size_t I, typename V>
struct generic_get_impl {
  constexpr generic_get_impl(int) noexcept {}

  constexpr AUTO_REFREF operator()(V&& v) const
      AUTO_REFREF_RETURN(access::variant::get_alt<I>(lib::forward<V>(v)).value)
};

template <std::size_t I, typename V>
inline constexpr AUTO_REFREF generic_get(V&& v)
    AUTO_REFREF_RETURN(generic_get_impl<I, V>(
        holds_alternative<I>(v)
            ? 0
            : (throw_bad_variant_access(), 0))(lib::forward<V>(v)))
} // namespace detail_

template <std::size_t I, typename... Ts>
inline constexpr variant_alternative_t<I, variant<Ts...>>& get(
    variant<Ts...>& v) {
  return detail_::generic_get<I>(v);
}

template <std::size_t I, typename... Ts>
inline constexpr variant_alternative_t<I, variant<Ts...>>&& get(
    variant<Ts...>&& v) {
  return detail_::generic_get<I>(lib::move(v));
}

template <std::size_t I, typename... Ts>
inline constexpr const variant_alternative_t<I, variant<Ts...>>& get(
    const variant<Ts...>& v) {
  return detail_::generic_get<I>(v);
}

template <std::size_t I, typename... Ts>
inline constexpr const variant_alternative_t<I, variant<Ts...>>&& get(
    const variant<Ts...>&& v) {
  return detail_::generic_get<I>(lib::move(v));
}

template <typename T, typename... Ts>
inline constexpr T& get(variant<Ts...>& v) {
  return get<detail_::find_index_checked<T, Ts...>::value>(v);
}

template <typename T, typename... Ts>
inline constexpr T&& get(variant<Ts...>&& v) {
  return get<detail_::find_index_checked<T, Ts...>::value>(lib::move(v));
}

template <typename T, typename... Ts>
inline constexpr const T& get(const variant<Ts...>& v) {
  return get<detail_::find_index_checked<T, Ts...>::value>(v);
}

template <typename T, typename... Ts>
inline constexpr const T&& get(const variant<Ts...>&& v) {
  return get<detail_::find_index_checked<T, Ts...>::value>(lib::move(v));
}

namespace detail_ {

template <std::size_t I, typename V>
inline constexpr /* auto * */ AUTO generic_get_if(V* v) noexcept AUTO_RETURN(
    v&& holds_alternative<I>(*v)
        ? lib::addressof(access::variant::get_alt<I>(*v).value)
        : nullptr)

} // namespace detail_

template <std::size_t I, typename... Ts>
inline constexpr lib::add_pointer_t<variant_alternative_t<I, variant<Ts...>>>
get_if(variant<Ts...>* v) noexcept {
  return detail_::generic_get_if<I>(v);
}

template <std::size_t I, typename... Ts>
inline constexpr lib::add_pointer_t<
    const variant_alternative_t<I, variant<Ts...>>>
get_if(const variant<Ts...>* v) noexcept {
  return detail_::generic_get_if<I>(v);
}

template <typename T, typename... Ts>
inline constexpr lib::add_pointer_t<T> get_if(variant<Ts...>* v) noexcept {
  return get_if<detail_::find_index_checked<T, Ts...>::value>(v);
}

template <typename T, typename... Ts>
inline constexpr lib::add_pointer_t<const T> get_if(
    const variant<Ts...>* v) noexcept {
  return get_if<detail_::find_index_checked<T, Ts...>::value>(v);
}

namespace detail_ {
template <typename RelOp>
struct convert_to_bool {
  template <typename Lhs, typename Rhs>
  inline constexpr bool operator()(Lhs&& lhs, Rhs&& rhs) const {
    static_assert(
        std::is_convertible<lib::invoke_result_t<RelOp, Lhs, Rhs>, bool>::value,
        "relational operators must return a type"
        " implicitly convertible to bool");
    return lib::invoke(RelOp{}, lib::forward<Lhs>(lhs), lib::forward<Rhs>(rhs));
  }
};
} // namespace detail_

template <typename... Ts>
inline constexpr bool operator==(
    const variant<Ts...>& lhs,
    const variant<Ts...>& rhs) {
  using detail_::visitation::variant;
  using equal_to = detail_::convert_to_bool<lib::equal_to>;
#ifdef MPARK_CPP14_CONSTEXPR
  if (lhs.index() != rhs.index())
    return false;
  if (lhs.valueless_by_exception())
    return true;
  return variant::visit_value_at(lhs.index(), equal_to{}, lhs, rhs);
#else
  return lhs.index() == rhs.index() &&
      (lhs.valueless_by_exception() ||
       variant::visit_value_at(lhs.index(), equal_to{}, lhs, rhs));
#endif
}

template <typename... Ts>
inline constexpr bool operator!=(
    const variant<Ts...>& lhs,
    const variant<Ts...>& rhs) {
  using detail_::visitation::variant;
  using not_equal_to = detail_::convert_to_bool<lib::not_equal_to>;
#ifdef MPARK_CPP14_CONSTEXPR
  if (lhs.index() != rhs.index())
    return true;
  if (lhs.valueless_by_exception())
    return false;
  return variant::visit_value_at(lhs.index(), not_equal_to{}, lhs, rhs);
#else
  return lhs.index() != rhs.index() ||
      (!lhs.valueless_by_exception() &&
       variant::visit_value_at(lhs.index(), not_equal_to{}, lhs, rhs));
#endif
}

template <typename... Ts>
inline constexpr bool operator<(
    const variant<Ts...>& lhs,
    const variant<Ts...>& rhs) {
  using detail_::visitation::variant;
  using less = detail_::convert_to_bool<lib::less>;
#ifdef MPARK_CPP14_CONSTEXPR
  if (rhs.valueless_by_exception())
    return false;
  if (lhs.valueless_by_exception())
    return true;
  if (lhs.index() < rhs.index())
    return true;
  if (lhs.index() > rhs.index())
    return false;
  return variant::visit_value_at(lhs.index(), less{}, lhs, rhs);
#else
  return !rhs.valueless_by_exception() &&
      (lhs.valueless_by_exception() || lhs.index() < rhs.index() ||
       (lhs.index() == rhs.index() &&
        variant::visit_value_at(lhs.index(), less{}, lhs, rhs)));
#endif
}

template <typename... Ts>
inline constexpr bool operator>(
    const variant<Ts...>& lhs,
    const variant<Ts...>& rhs) {
  using detail_::visitation::variant;
  using greater = detail_::convert_to_bool<lib::greater>;
#ifdef MPARK_CPP14_CONSTEXPR
  if (lhs.valueless_by_exception())
    return false;
  if (rhs.valueless_by_exception())
    return true;
  if (lhs.index() > rhs.index())
    return true;
  if (lhs.index() < rhs.index())
    return false;
  return variant::visit_value_at(lhs.index(), greater{}, lhs, rhs);
#else
  return !lhs.valueless_by_exception() &&
      (rhs.valueless_by_exception() || lhs.index() > rhs.index() ||
       (lhs.index() == rhs.index() &&
        variant::visit_value_at(lhs.index(), greater{}, lhs, rhs)));
#endif
}

template <typename... Ts>
inline constexpr bool operator<=(
    const variant<Ts...>& lhs,
    const variant<Ts...>& rhs) {
  using detail_::visitation::variant;
  using less_equal = detail_::convert_to_bool<lib::less_equal>;
#ifdef MPARK_CPP14_CONSTEXPR
  if (lhs.valueless_by_exception())
    return true;
  if (rhs.valueless_by_exception())
    return false;
  if (lhs.index() < rhs.index())
    return true;
  if (lhs.index() > rhs.index())
    return false;
  return variant::visit_value_at(lhs.index(), less_equal{}, lhs, rhs);
#else
  return lhs.valueless_by_exception() ||
      (!rhs.valueless_by_exception() &&
       (lhs.index() < rhs.index() ||
        (lhs.index() == rhs.index() &&
         variant::visit_value_at(lhs.index(), less_equal{}, lhs, rhs))));
#endif
}

template <typename... Ts>
inline constexpr bool operator>=(
    const variant<Ts...>& lhs,
    const variant<Ts...>& rhs) {
  using detail_::visitation::variant;
  using greater_equal = detail_::convert_to_bool<lib::greater_equal>;
#ifdef MPARK_CPP14_CONSTEXPR
  if (rhs.valueless_by_exception())
    return true;
  if (lhs.valueless_by_exception())
    return false;
  if (lhs.index() > rhs.index())
    return true;
  if (lhs.index() < rhs.index())
    return false;
  return variant::visit_value_at(lhs.index(), greater_equal{}, lhs, rhs);
#else
  return rhs.valueless_by_exception() ||
      (!lhs.valueless_by_exception() &&
       (lhs.index() > rhs.index() ||
        (lhs.index() == rhs.index() &&
         variant::visit_value_at(lhs.index(), greater_equal{}, lhs, rhs))));
#endif
}

struct monostate {};

inline constexpr bool operator<(monostate, monostate) noexcept {
  return false;
}

inline constexpr bool operator>(monostate, monostate) noexcept {
  return false;
}

inline constexpr bool operator<=(monostate, monostate) noexcept {
  return true;
}

inline constexpr bool operator>=(monostate, monostate) noexcept {
  return true;
}

inline constexpr bool operator==(monostate, monostate) noexcept {
  return true;
}

inline constexpr bool operator!=(monostate, monostate) noexcept {
  return false;
}

#ifdef MPARK_CPP14_CONSTEXPR
namespace detail_ {

inline constexpr bool any(std::initializer_list<bool> bs) {
  for (bool b : bs) {
    if (b) {
      return true;
    }
  }
  return false;
}

} // namespace detail_

template <typename Visitor, typename... Vs>
inline constexpr decltype(auto) visit(Visitor&& visitor, Vs&&... vs) {
  return (!detail_::any({vs.valueless_by_exception()...})
              ? (void)0
              : throw_bad_variant_access()),
         detail_::visitation::variant::visit_value(
             lib::forward<Visitor>(visitor), lib::forward<Vs>(vs)...);
}
#else
namespace detail_ {

template <std::size_t N>
inline constexpr bool all_impl(const lib::array<bool, N>& bs, std::size_t idx) {
  return idx >= N || (bs[idx] && all_impl(bs, idx + 1));
}

template <std::size_t N>
inline constexpr bool all(const lib::array<bool, N>& bs) {
  return all_impl(bs, 0);
}

} // namespace detail_

template <typename Visitor, typename... Vs>
inline constexpr DECLTYPE_AUTO visit(Visitor&& visitor, Vs&&... vs)
    DECLTYPE_AUTO_RETURN(
        (detail_::all(lib::array<bool, sizeof...(Vs)>{
             {!vs.valueless_by_exception()...}})
             ? (void)0
             : throw_bad_variant_access()),
        detail_::visitation::variant::visit_value(
            lib::forward<Visitor>(visitor),
            lib::forward<Vs>(vs)...))
#endif

template <typename... Ts>
inline auto swap(variant<Ts...>& lhs, variant<Ts...>& rhs) noexcept(
    noexcept(lhs.swap(rhs))) -> decltype(lhs.swap(rhs)) {
  lhs.swap(rhs);
}

namespace detail_ {

template <typename T, typename...>
using enabled_type = T;

namespace hash {

template <typename H, typename K>
constexpr bool meets_requirements() noexcept {
  return std::is_copy_constructible<H>::value &&
      std::is_move_constructible<H>::value &&
      lib::is_invocable_r<std::size_t, H, const K&>::value;
}

template <typename K>
constexpr bool is_enabled() noexcept {
  using H = std::hash<K>;
  return meets_requirements<H, K>() &&
      std::is_default_constructible<H>::value &&
      std::is_copy_assignable<H>::value && std::is_move_assignable<H>::value;
}

} // namespace hash

} // namespace detail_

#undef AUTO
#undef AUTO_RETURN

#undef AUTO_REFREF
#undef AUTO_REFREF_RETURN

#undef DECLTYPE_AUTO
#undef DECLTYPE_AUTO_RETURN

} // namespace c10

namespace std {

template <typename... Ts>
struct hash<c10::detail_::enabled_type<
    c10::variant<Ts...>,
    c10::lib::enable_if_t<c10::lib::all<c10::detail_::hash::is_enabled<
        c10::lib::remove_const_t<Ts>>()...>::value>>> {
  using argument_type = c10::variant<Ts...>;
  using result_type = std::size_t;

  inline result_type operator()(const argument_type& v) const {
    using c10::detail_::visitation::variant;
    std::size_t result = v.valueless_by_exception()
        ? 299792458 // Random value chosen by the universe upon creation
        : variant::visit_alt(
#ifdef MPARK_GENERIC_LAMBDAS
              [](const auto& alt) {
                using alt_type = c10::lib::decay_t<decltype(alt)>;
                using value_type =
                    c10::lib::remove_const_t<typename alt_type::value_type>;
                return hash<value_type>{}(alt.value);
              }
#else
              hasher {}
#endif
              ,
              v);
    return hash_combine(result, hash<std::size_t>{}(v.index()));
  }

 private:
#ifndef MPARK_GENERIC_LAMBDAS
  struct hasher {
    template <typename Alt>
    inline std::size_t operator()(const Alt& alt) const {
      using alt_type = c10::lib::decay_t<Alt>;
      using value_type =
          c10::lib::remove_const_t<typename alt_type::value_type>;
      return hash<value_type>{}(alt.value);
    }
  };
#endif

  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  }
};

template <>
struct hash<c10::monostate> {
  using argument_type = c10::monostate;
  using result_type = std::size_t;

  inline result_type operator()(const argument_type&) const noexcept {
    return 66740831; // return a fundamentally attractive random value.
  }
};

} // namespace std

#endif // C10_UTIL_VARIANT_H_
