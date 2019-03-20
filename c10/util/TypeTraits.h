#pragma once

#include <c10/util/C++17.h>
#include <functional>

namespace c10 {
namespace guts {


/**
 * is_equality_comparable<T> is true_type iff the equality operator is defined for T.
 */
template<class T, class Enable = void> struct is_equality_comparable : std::false_type {};
template<class T> struct is_equality_comparable<T, void_t<decltype(std::declval<T&>() == std::declval<T&>())>> : std::true_type {};
template<class T> using is_equality_comparable_t = typename is_equality_comparable<T>::type;



/**
 * is_hashable<T> is true_type iff std::hash is defined for T
 */
template<class T, class Enable = void> struct is_hashable : std::false_type {};
template<class T> struct is_hashable<T, void_t<decltype(std::hash<T>()(std::declval<T&>()))>> : std::true_type {};
template<class T> using is_hashable_t = typename is_hashable<T>::type;



/**
 * is_function_type<T> is true_type iff T is a plain function type (i.e. "Result(Args...)")
 */
template<class T>
struct is_function_type : std::false_type {};
template<class Result, class... Args>
struct is_function_type<Result (Args...)> : std::true_type {};
template<class T> using is_function_type_t = typename is_function_type<T>::type;



/**
 * is_instantiation_of<T, I> is true_type iff I is a template instantiation of T (e.g. vector<int> is an instantiation of vector)
 *  Example:
 *    is_instantiation_of_t<vector, vector<int>> // true
 *    is_instantiation_of_t<pair, pair<int, string>> // true
 *    is_instantiation_of_t<vector, pair<int, string>> // false
 */
template <template <class...> class Template, class T>
struct is_instantiation_of : std::false_type {};
template <template <class...> class Template, class... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};
template<template<class...> class Template, class T> using is_instantiation_of_t = typename is_instantiation_of<Template, T>::type;



/**
 * is_type_condition<C> is true_type iff C<...> is a type trait representing a condition (i.e. has a constexpr static bool ::value member)
 * Example:
 *   is_type_condition<std::is_reference>  // true
 */
template<template<class> class C, class Enable = void>
struct is_type_condition : std::false_type {};
template<template<class> class C>
struct is_type_condition<C, guts::enable_if_t<std::is_same<bool, guts::remove_cv_t<decltype(C<int>::value)>>::value>> : std::true_type {};


}
}
