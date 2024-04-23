//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// type_traits.h
// -----------------------------------------------------------------------------
//
// This file contains C++11-compatible versions of standard <type_traits> API
// functions for determining the characteristics of types. Such traits can
// support type inference, classification, and transformation, as well as
// make it easier to write templates based on generic type behavior.
//
// See https://en.cppreference.com/w/cpp/header/type_traits
//
// WARNING: use of many of the constructs in this header will count as "complex
// template metaprogramming", so before proceeding, please carefully consider
// https://google.github.io/styleguide/cppguide.html#Template_metaprogramming
//
// WARNING: using template metaprogramming to detect or depend on API
// features is brittle and not guaranteed. Neither the standard library nor
// Abseil provides any guarantee that APIs are stable in the face of template
// metaprogramming. Use with caution.
#ifndef OTABSL_META_TYPE_TRAITS_H_
#define OTABSL_META_TYPE_TRAITS_H_

#include <stddef.h>
#include <functional>
#include <type_traits>

#include "../base/config.h"

// MSVC constructibility traits do not detect destructor properties and so our
// implementations should not use them as a source-of-truth.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__GNUC__)
#define OTABSL_META_INTERNAL_STD_CONSTRUCTION_TRAITS_DONT_CHECK_DESTRUCTION 1
#endif

namespace absl {
OTABSL_NAMESPACE_BEGIN

// Defined and documented later on in this file.
template <typename T>
struct is_trivially_destructible;

// Defined and documented later on in this file.
template <typename T>
struct is_trivially_move_assignable;

namespace type_traits_internal {

// Silence MSVC warnings about the destructor being defined as deleted.
#if defined(_MSC_VER) && !defined(__GNUC__)
#pragma warning(push)
#pragma warning(disable : 4624)
#endif  // defined(_MSC_VER) && !defined(__GNUC__)

template <class T>
union SingleMemberUnion {
  T t;
};

// Restore the state of the destructor warning that was silenced above.
#if defined(_MSC_VER) && !defined(__GNUC__)
#pragma warning(pop)
#endif  // defined(_MSC_VER) && !defined(__GNUC__)

template <class T>
struct IsTriviallyMoveConstructibleObject
    : std::integral_constant<
          bool, std::is_move_constructible<
                    type_traits_internal::SingleMemberUnion<T>>::value &&
                    absl::is_trivially_destructible<T>::value> {};

template <class T>
struct IsTriviallyCopyConstructibleObject
    : std::integral_constant<
          bool, std::is_copy_constructible<
                    type_traits_internal::SingleMemberUnion<T>>::value &&
                    absl::is_trivially_destructible<T>::value> {};

template <class T>
struct IsTriviallyMoveAssignableReference : std::false_type {};

template <class T>
struct IsTriviallyMoveAssignableReference<T&>
    : absl::is_trivially_move_assignable<T>::type {};

template <class T>
struct IsTriviallyMoveAssignableReference<T&&>
    : absl::is_trivially_move_assignable<T>::type {};

template <typename... Ts>
struct VoidTImpl {
  using type = void;
};

// This trick to retrieve a default alignment is necessary for our
// implementation of aligned_storage_t to be consistent with any implementation
// of std::aligned_storage.
template <size_t Len, typename T = std::aligned_storage<Len>>
struct default_alignment_of_aligned_storage;

template <size_t Len, size_t Align>
struct default_alignment_of_aligned_storage<Len,
                                            std::aligned_storage<Len, Align>> {
  static constexpr size_t value = Align;
};

////////////////////////////////
// Library Fundamentals V2 TS //
////////////////////////////////

// NOTE: The `is_detected` family of templates here differ from the library
// fundamentals specification in that for library fundamentals, `Op<Args...>` is
// evaluated as soon as the type `is_detected<Op, Args...>` undergoes
// substitution, regardless of whether or not the `::value` is accessed. That
// is inconsistent with all other standard traits and prevents lazy evaluation
// in larger contexts (such as if the `is_detected` check is a trailing argument
// of a `conjunction`. This implementation opts to instead be lazy in the same
// way that the standard traits are (this "defect" of the detection idiom
// specifications has been reported).

template <class Enabler, template <class...> class Op, class... Args>
struct is_detected_impl {
  using type = std::false_type;
};

template <template <class...> class Op, class... Args>
struct is_detected_impl<typename VoidTImpl<Op<Args...>>::type, Op, Args...> {
  using type = std::true_type;
};

template <template <class...> class Op, class... Args>
struct is_detected : is_detected_impl<void, Op, Args...>::type {};

template <class Enabler, class To, template <class...> class Op, class... Args>
struct is_detected_convertible_impl {
  using type = std::false_type;
};

template <class To, template <class...> class Op, class... Args>
struct is_detected_convertible_impl<
    typename std::enable_if<std::is_convertible<Op<Args...>, To>::value>::type,
    To, Op, Args...> {
  using type = std::true_type;
};

template <class To, template <class...> class Op, class... Args>
struct is_detected_convertible
    : is_detected_convertible_impl<void, To, Op, Args...>::type {};

template <typename T>
using IsCopyAssignableImpl =
    decltype(std::declval<T&>() = std::declval<const T&>());

template <typename T>
using IsMoveAssignableImpl = decltype(std::declval<T&>() = std::declval<T&&>());

}  // namespace type_traits_internal

// MSVC 19.10 and higher have a regression that causes our workarounds to fail, but their
// std forms now appear to be compliant.
#if defined(_MSC_VER) && !defined(__clang__) && (_MSC_VER >= 1910)

template <typename T>
using is_copy_assignable = std::is_copy_assignable<T>;

template <typename T>
using is_move_assignable = std::is_move_assignable<T>;

#else

template <typename T>
struct is_copy_assignable : type_traits_internal::is_detected<
                                type_traits_internal::IsCopyAssignableImpl, T> {
};

template <typename T>
struct is_move_assignable : type_traits_internal::is_detected<
                                type_traits_internal::IsMoveAssignableImpl, T> {
};

#endif

// void_t()
//
// Ignores the type of any its arguments and returns `void`. In general, this
// metafunction allows you to create a general case that maps to `void` while
// allowing specializations that map to specific types.
//
// This metafunction is designed to be a drop-in replacement for the C++17
// `std::void_t` metafunction.
//
// NOTE: `absl::void_t` does not use the standard-specified implementation so
// that it can remain compatible with gcc < 5.1. This can introduce slightly
// different behavior, such as when ordering partial specializations.
template <typename... Ts>
using void_t = typename type_traits_internal::VoidTImpl<Ts...>::type;

// conjunction
//
// Performs a compile-time logical AND operation on the passed types (which
// must have  `::value` members convertible to `bool`. Short-circuits if it
// encounters any `false` members (and does not compare the `::value` members
// of any remaining arguments).
//
// This metafunction is designed to be a drop-in replacement for the C++17
// `std::conjunction` metafunction.
template <typename... Ts>
struct conjunction;

template <typename T, typename... Ts>
struct conjunction<T, Ts...>
    : std::conditional<T::value, conjunction<Ts...>, T>::type {};

template <typename T>
struct conjunction<T> : T {};

template <>
struct conjunction<> : std::true_type {};

// disjunction
//
// Performs a compile-time logical OR operation on the passed types (which
// must have  `::value` members convertible to `bool`. Short-circuits if it
// encounters any `true` members (and does not compare the `::value` members
// of any remaining arguments).
//
// This metafunction is designed to be a drop-in replacement for the C++17
// `std::disjunction` metafunction.
template <typename... Ts>
struct disjunction;

template <typename T, typename... Ts>
struct disjunction<T, Ts...> :
      std::conditional<T::value, T, disjunction<Ts...>>::type {};

template <typename T>
struct disjunction<T> : T {};

template <>
struct disjunction<> : std::false_type {};

// negation
//
// Performs a compile-time logical NOT operation on the passed type (which
// must have  `::value` members convertible to `bool`.
//
// This metafunction is designed to be a drop-in replacement for the C++17
// `std::negation` metafunction.
template <typename T>
struct negation : std::integral_constant<bool, !T::value> {};

// is_function()
//
// Determines whether the passed type `T` is a function type.
//
// This metafunction is designed to be a drop-in replacement for the C++11
// `std::is_function()` metafunction for platforms that have incomplete C++11
// support (such as libstdc++ 4.x).
//
// This metafunction works because appending `const` to a type does nothing to
// function types and reference types (and forms a const-qualified type
// otherwise).
template <typename T>
struct is_function
    : std::integral_constant<
          bool, !(std::is_reference<T>::value ||
                  std::is_const<typename std::add_const<T>::type>::value)> {};

// is_trivially_destructible()
//
// Determines whether the passed type `T` is trivially destructible.
//
// This metafunction is designed to be a drop-in replacement for the C++11
// `std::is_trivially_destructible()` metafunction for platforms that have
// incomplete C++11 support (such as libstdc++ 4.x). On any platforms that do
// fully support C++11, we check whether this yields the same result as the std
// implementation.
//
// NOTE: the extensions (__has_trivial_xxx) are implemented in gcc (version >=
// 4.3) and clang. Since we are supporting libstdc++ > 4.7, they should always
// be present. These  extensions are documented at
// https://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html#Type-Traits.
template <typename T>
struct is_trivially_destructible
#ifdef OTABSL_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE
    : std::is_trivially_destructible<T> {
#else
    : std::integral_constant<bool, __has_trivial_destructor(T) &&
                                   std::is_destructible<T>::value> {
#endif
#ifdef OTABSL_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE
 private:
  static constexpr bool compliant = std::is_trivially_destructible<T>::value ==
                                    is_trivially_destructible::value;
  static_assert(compliant || std::is_trivially_destructible<T>::value,
                "Not compliant with std::is_trivially_destructible; "
                "Standard: false, Implementation: true");
  static_assert(compliant || !std::is_trivially_destructible<T>::value,
                "Not compliant with std::is_trivially_destructible; "
                "Standard: true, Implementation: false");
#endif  // OTABSL_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE
};

// is_trivially_default_constructible()
//
// Determines whether the passed type `T` is trivially default constructible.
//
// This metafunction is designed to be a drop-in replacement for the C++11
// `std::is_trivially_default_constructible()` metafunction for platforms that
// have incomplete C++11 support (such as libstdc++ 4.x). On any platforms that
// do fully support C++11, we check whether this yields the same result as the
// std implementation.
//
// NOTE: according to the C++ standard, Section: 20.15.4.3 [meta.unary.prop]
// "The predicate condition for a template specialization is_constructible<T,
// Args...> shall be satisfied if and only if the following variable
// definition would be well-formed for some invented variable t:
//
// T t(declval<Args>()...);
//
// is_trivially_constructible<T, Args...> additionally requires that the
// variable definition does not call any operation that is not trivial.
// For the purposes of this check, the call to std::declval is considered
// trivial."
//
// Notes from https://en.cppreference.com/w/cpp/types/is_constructible:
// In many implementations, is_nothrow_constructible also checks if the
// destructor throws because it is effectively noexcept(T(arg)). Same
// applies to is_trivially_constructible, which, in these implementations, also
// requires that the destructor is trivial.
// GCC bug 51452: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51452
// LWG issue 2116: http://cplusplus.github.io/LWG/lwg-active.html#2116.
//
// "T obj();" need to be well-formed and not call any nontrivial operation.
// Nontrivially destructible types will cause the expression to be nontrivial.
template <typename T>
struct is_trivially_default_constructible
#if defined(OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE)
    : std::is_trivially_default_constructible<T> {
#else
    : std::integral_constant<bool, __has_trivial_constructor(T) &&
                                   std::is_default_constructible<T>::value &&
                                   is_trivially_destructible<T>::value> {
#endif
#if defined(OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE) && \
    !defined(                                            \
        OTABSL_META_INTERNAL_STD_CONSTRUCTION_TRAITS_DONT_CHECK_DESTRUCTION)
 private:
  static constexpr bool compliant =
      std::is_trivially_default_constructible<T>::value ==
      is_trivially_default_constructible::value;
  static_assert(compliant || std::is_trivially_default_constructible<T>::value,
                "Not compliant with std::is_trivially_default_constructible; "
                "Standard: false, Implementation: true");
  static_assert(compliant || !std::is_trivially_default_constructible<T>::value,
                "Not compliant with std::is_trivially_default_constructible; "
                "Standard: true, Implementation: false");
#endif  // OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE
};

// is_trivially_move_constructible()
//
// Determines whether the passed type `T` is trivially move constructible.
//
// This metafunction is designed to be a drop-in replacement for the C++11
// `std::is_trivially_move_constructible()` metafunction for platforms that have
// incomplete C++11 support (such as libstdc++ 4.x). On any platforms that do
// fully support C++11, we check whether this yields the same result as the std
// implementation.
//
// NOTE: `T obj(declval<T>());` needs to be well-formed and not call any
// nontrivial operation.  Nontrivially destructible types will cause the
// expression to be nontrivial.
template <typename T>
struct is_trivially_move_constructible
#if defined(OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE)
    : std::is_trivially_move_constructible<T> {
#else
    : std::conditional<
          std::is_object<T>::value && !std::is_array<T>::value,
          type_traits_internal::IsTriviallyMoveConstructibleObject<T>,
          std::is_reference<T>>::type::type {
#endif
#if defined(OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE) && \
    !defined(                                            \
        OTABSL_META_INTERNAL_STD_CONSTRUCTION_TRAITS_DONT_CHECK_DESTRUCTION)
 private:
  static constexpr bool compliant =
      std::is_trivially_move_constructible<T>::value ==
      is_trivially_move_constructible::value;
  static_assert(compliant || std::is_trivially_move_constructible<T>::value,
                "Not compliant with std::is_trivially_move_constructible; "
                "Standard: false, Implementation: true");
  static_assert(compliant || !std::is_trivially_move_constructible<T>::value,
                "Not compliant with std::is_trivially_move_constructible; "
                "Standard: true, Implementation: false");
#endif  // OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE
};

// is_trivially_copy_constructible()
//
// Determines whether the passed type `T` is trivially copy constructible.
//
// This metafunction is designed to be a drop-in replacement for the C++11
// `std::is_trivially_copy_constructible()` metafunction for platforms that have
// incomplete C++11 support (such as libstdc++ 4.x). On any platforms that do
// fully support C++11, we check whether this yields the same result as the std
// implementation.
//
// NOTE: `T obj(declval<const T&>());` needs to be well-formed and not call any
// nontrivial operation.  Nontrivially destructible types will cause the
// expression to be nontrivial.
template <typename T>
struct is_trivially_copy_constructible
    : std::conditional<
          std::is_object<T>::value && !std::is_array<T>::value,
          type_traits_internal::IsTriviallyCopyConstructibleObject<T>,
          std::is_lvalue_reference<T>>::type::type {
#if defined(OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE) && \
    !defined(                                            \
        OTABSL_META_INTERNAL_STD_CONSTRUCTION_TRAITS_DONT_CHECK_DESTRUCTION)
 private:
  static constexpr bool compliant =
      std::is_trivially_copy_constructible<T>::value ==
      is_trivially_copy_constructible::value;
  static_assert(compliant || std::is_trivially_copy_constructible<T>::value,
                "Not compliant with std::is_trivially_copy_constructible; "
                "Standard: false, Implementation: true");
  static_assert(compliant || !std::is_trivially_copy_constructible<T>::value,
                "Not compliant with std::is_trivially_copy_constructible; "
                "Standard: true, Implementation: false");
#endif  // OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE
};

// is_trivially_move_assignable()
//
// Determines whether the passed type `T` is trivially move assignable.
//
// This metafunction is designed to be a drop-in replacement for the C++11
// `std::is_trivially_move_assignable()` metafunction for platforms that have
// incomplete C++11 support (such as libstdc++ 4.x). On any platforms that do
// fully support C++11, we check whether this yields the same result as the std
// implementation.
//
// NOTE: `is_assignable<T, U>::value` is `true` if the expression
// `declval<T>() = declval<U>()` is well-formed when treated as an unevaluated
// operand. `is_trivially_assignable<T, U>` requires the assignment to call no
// operation that is not trivial. `is_trivially_copy_assignable<T>` is simply
// `is_trivially_assignable<T&, T>`.
template <typename T>
struct is_trivially_move_assignable
    : std::conditional<
          std::is_object<T>::value && !std::is_array<T>::value &&
              std::is_move_assignable<T>::value,
          std::is_move_assignable<type_traits_internal::SingleMemberUnion<T>>,
          type_traits_internal::IsTriviallyMoveAssignableReference<T>>::type::
          type {
#ifdef OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE
 private:
  static constexpr bool compliant =
      std::is_trivially_move_assignable<T>::value ==
      is_trivially_move_assignable::value;
  static_assert(compliant || std::is_trivially_move_assignable<T>::value,
                "Not compliant with std::is_trivially_move_assignable; "
                "Standard: false, Implementation: true");
  static_assert(compliant || !std::is_trivially_move_assignable<T>::value,
                "Not compliant with std::is_trivially_move_assignable; "
                "Standard: true, Implementation: false");
#endif  // OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE
};

// is_trivially_copy_assignable()
//
// Determines whether the passed type `T` is trivially copy assignable.
//
// This metafunction is designed to be a drop-in replacement for the C++11
// `std::is_trivially_copy_assignable()` metafunction for platforms that have
// incomplete C++11 support (such as libstdc++ 4.x). On any platforms that do
// fully support C++11, we check whether this yields the same result as the std
// implementation.
//
// NOTE: `is_assignable<T, U>::value` is `true` if the expression
// `declval<T>() = declval<U>()` is well-formed when treated as an unevaluated
// operand. `is_trivially_assignable<T, U>` requires the assignment to call no
// operation that is not trivial. `is_trivially_copy_assignable<T>` is simply
// `is_trivially_assignable<T&, const T&>`.
template <typename T>
struct is_trivially_copy_assignable
#ifdef OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE
    : std::is_trivially_copy_assignable<T> {
#else
    : std::integral_constant<
          bool, __has_trivial_assign(typename std::remove_reference<T>::type) &&
                    absl::is_copy_assignable<T>::value> {
#endif
#ifdef OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE
 private:
  static constexpr bool compliant =
      std::is_trivially_copy_assignable<T>::value ==
      is_trivially_copy_assignable::value;
  static_assert(compliant || std::is_trivially_copy_assignable<T>::value,
                "Not compliant with std::is_trivially_copy_assignable; "
                "Standard: false, Implementation: true");
  static_assert(compliant || !std::is_trivially_copy_assignable<T>::value,
                "Not compliant with std::is_trivially_copy_assignable; "
                "Standard: true, Implementation: false");
#endif  // OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE
};

namespace type_traits_internal {
// is_trivially_copyable()
//
// Determines whether the passed type `T` is trivially copyable.
//
// This metafunction is designed to be a drop-in replacement for the C++11
// `std::is_trivially_copyable()` metafunction for platforms that have
// incomplete C++11 support (such as libstdc++ 4.x). We use the C++17 definition
// of TriviallyCopyable.
//
// NOTE: `is_trivially_copyable<T>::value` is `true` if all of T's copy/move
// constructors/assignment operators are trivial or deleted, T has at least
// one non-deleted copy/move constructor/assignment operator, and T is trivially
// destructible. Arrays of trivially copyable types are trivially copyable.
//
// We expose this metafunction only for internal use within absl.

#if defined(OTABSL_HAVE_STD_IS_TRIVIALLY_COPYABLE)
template <typename T>
struct is_trivially_copyable : std::is_trivially_copyable<T> {};
#else
template <typename T>
class is_trivially_copyable_impl {
  using ExtentsRemoved = typename std::remove_all_extents<T>::type;
  static constexpr bool kIsCopyOrMoveConstructible =
      std::is_copy_constructible<ExtentsRemoved>::value ||
      std::is_move_constructible<ExtentsRemoved>::value;
  static constexpr bool kIsCopyOrMoveAssignable =
      absl::is_copy_assignable<ExtentsRemoved>::value ||
      absl::is_move_assignable<ExtentsRemoved>::value;

 public:
  static constexpr bool kValue =
      (__has_trivial_copy(ExtentsRemoved) || !kIsCopyOrMoveConstructible) &&
      (__has_trivial_assign(ExtentsRemoved) || !kIsCopyOrMoveAssignable) &&
      (kIsCopyOrMoveConstructible || kIsCopyOrMoveAssignable) &&
      is_trivially_destructible<ExtentsRemoved>::value &&
      // We need to check for this explicitly because otherwise we'll say
      // references are trivial copyable when compiled by MSVC.
      !std::is_reference<ExtentsRemoved>::value;
};

template <typename T>
struct is_trivially_copyable
    : std::integral_constant<
          bool, type_traits_internal::is_trivially_copyable_impl<T>::kValue> {};
#endif
}  // namespace type_traits_internal

// -----------------------------------------------------------------------------
// C++14 "_t" trait aliases
// -----------------------------------------------------------------------------

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_const_t = typename std::remove_const<T>::type;

template <typename T>
using remove_volatile_t = typename std::remove_volatile<T>::type;

template <typename T>
using add_cv_t = typename std::add_cv<T>::type;

template <typename T>
using add_const_t = typename std::add_const<T>::type;

template <typename T>
using add_volatile_t = typename std::add_volatile<T>::type;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using add_lvalue_reference_t = typename std::add_lvalue_reference<T>::type;

template <typename T>
using add_rvalue_reference_t = typename std::add_rvalue_reference<T>::type;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

template <typename T>
using add_pointer_t = typename std::add_pointer<T>::type;

template <typename T>
using make_signed_t = typename std::make_signed<T>::type;

template <typename T>
using make_unsigned_t = typename std::make_unsigned<T>::type;

template <typename T>
using remove_extent_t = typename std::remove_extent<T>::type;

template <typename T>
using remove_all_extents_t = typename std::remove_all_extents<T>::type;

template <size_t Len, size_t Align = type_traits_internal::
                          default_alignment_of_aligned_storage<Len>::value>
using aligned_storage_t = typename std::aligned_storage<Len, Align>::type;

template <typename T>
using decay_t = typename std::decay<T>::type;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <bool B, typename T, typename F>
using conditional_t = typename std::conditional<B, T, F>::type;

template <typename... T>
using common_type_t = typename std::common_type<T...>::type;

template <typename T>
using underlying_type_t = typename std::underlying_type<T>::type;

namespace type_traits_internal {

#if (!defined(_MSVC_LANG) && (__cplusplus >= 201703L)) || \
    (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L))
// std::result_of is deprecated (C++17) or removed (C++20)
template<typename> struct result_of;
template<typename F, typename... Args>
struct result_of<F(Args...)> : std::invoke_result<F, Args...> {};
#else
template<typename F> using result_of = std::result_of<F>;
#endif

}  // namespace type_traits_internal

template<typename F>
using result_of_t = typename type_traits_internal::result_of<F>::type;

namespace type_traits_internal {
// In MSVC we can't probe std::hash or stdext::hash because it triggers a
// static_assert instead of failing substitution. Libc++ prior to 4.0
// also used a static_assert.
//
#if defined(_MSC_VER) || (defined(_LIBCPP_VERSION) && \
                          _LIBCPP_VERSION < 4000 && _LIBCPP_STD_VER > 11)
#define OTABSL_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_ 0
#else
#define OTABSL_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_ 1
#endif

#if !OTABSL_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_
template <typename Key, typename = size_t>
struct IsHashable : std::true_type {};
#else   // OTABSL_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_
template <typename Key, typename = void>
struct IsHashable : std::false_type {};

template <typename Key>
struct IsHashable<
    Key,
    absl::enable_if_t<std::is_convertible<
        decltype(std::declval<std::hash<Key>&>()(std::declval<Key const&>())),
        std::size_t>::value>> : std::true_type {};
#endif  // !OTABSL_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_

struct AssertHashEnabledHelper {
 private:
  static void Sink(...) {}
  struct NAT {};

  template <class Key>
  static auto GetReturnType(int)
      -> decltype(std::declval<std::hash<Key>>()(std::declval<Key const&>()));
  template <class Key>
  static NAT GetReturnType(...);

  template <class Key>
  static std::nullptr_t DoIt() {
    static_assert(IsHashable<Key>::value,
                  "std::hash<Key> does not provide a call operator");
    static_assert(
        std::is_default_constructible<std::hash<Key>>::value,
        "std::hash<Key> must be default constructible when it is enabled");
    static_assert(
        std::is_copy_constructible<std::hash<Key>>::value,
        "std::hash<Key> must be copy constructible when it is enabled");
    static_assert(absl::is_copy_assignable<std::hash<Key>>::value,
                  "std::hash<Key> must be copy assignable when it is enabled");
    // is_destructible is unchecked as it's implied by each of the
    // is_constructible checks.
    using ReturnType = decltype(GetReturnType<Key>(0));
    static_assert(std::is_same<ReturnType, NAT>::value ||
                      std::is_same<ReturnType, size_t>::value,
                  "std::hash<Key> must return size_t");
    return nullptr;
  }

  template <class... Ts>
  friend void AssertHashEnabled();
};

template <class... Ts>
inline void AssertHashEnabled() {
  using Helper = AssertHashEnabledHelper;
  Helper::Sink(Helper::DoIt<Ts>()...);
}

}  // namespace type_traits_internal

// An internal namespace that is required to implement the C++17 swap traits.
// It is not further nested in type_traits_internal to avoid long symbol names.
namespace swap_internal {

// Necessary for the traits.
using std::swap;

// This declaration prevents global `swap` and `absl::swap` overloads from being
// considered unless ADL picks them up.
void swap();

template <class T>
using IsSwappableImpl = decltype(swap(std::declval<T&>(), std::declval<T&>()));

// NOTE: This dance with the default template parameter is for MSVC.
template <class T,
          class IsNoexcept = std::integral_constant<
              bool, noexcept(swap(std::declval<T&>(), std::declval<T&>()))>>
using IsNothrowSwappableImpl = typename std::enable_if<IsNoexcept::value>::type;

// IsSwappable
//
// Determines whether the standard swap idiom is a valid expression for
// arguments of type `T`.
template <class T>
struct IsSwappable
    : absl::type_traits_internal::is_detected<IsSwappableImpl, T> {};

// IsNothrowSwappable
//
// Determines whether the standard swap idiom is a valid expression for
// arguments of type `T` and is noexcept.
template <class T>
struct IsNothrowSwappable
    : absl::type_traits_internal::is_detected<IsNothrowSwappableImpl, T> {};

// Swap()
//
// Performs the swap idiom from a namespace where valid candidates may only be
// found in `std` or via ADL.
template <class T, absl::enable_if_t<IsSwappable<T>::value, int> = 0>
void Swap(T& lhs, T& rhs) noexcept(IsNothrowSwappable<T>::value) {
  swap(lhs, rhs);
}

// StdSwapIsUnconstrained
//
// Some standard library implementations are broken in that they do not
// constrain `std::swap`. This will effectively tell us if we are dealing with
// one of those implementations.
using StdSwapIsUnconstrained = IsSwappable<void()>;

}  // namespace swap_internal

namespace type_traits_internal {

// Make the swap-related traits/function accessible from this namespace.
using swap_internal::IsNothrowSwappable;
using swap_internal::IsSwappable;
using swap_internal::Swap;
using swap_internal::StdSwapIsUnconstrained;

}  // namespace type_traits_internal
OTABSL_NAMESPACE_END
}  // namespace absl

#endif  // OTABSL_META_TYPE_TRAITS_H_
