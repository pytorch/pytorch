#pragma once

#include <type_traits>
#include <array>
#include <functional>
#include <c10/util/TypeList.h>
#include <c10/util/Array.h>
#include <tuple>

namespace c10 { namespace guts {

/**
 **** function_traits ****
 * 
 * Access information about result type or arguments from a function type.
 * Example:
 * using A = function_traits<int (float, double)>::return_type // A == int
 * using A = function_traits<int (float, double)>::parameter_types::tuple_type // A == tuple<float, double>
 */

// Modified from https://stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda

// Fallback, anything with an operator()
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {
};

// Pointers to class members that are themselves functors.
// For example, in the following code:
// template <typename func_t>
// struct S {
//     func_t f;
// };
// template <typename func_t>
// S<func_t> make_s(func_t f) {
//     return S<func_t> { .f = f };
// }
//
// auto s = make_s([] (int, float) -> double { /* ... */ });
//
// function_traits<decltype(&s::f)> traits;
template <typename ClassType, typename T>
struct function_traits<T ClassType::*> : public function_traits<T> {
};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const> : public function_traits<ReturnType(Args...)> {
};

// Reference types
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T*> : public function_traits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  using func_type = ReturnType (Args...);
  using return_type = ReturnType;
  using parameter_types = typelist::typelist<Args...>;

  // arity is the number of arguments.
  constexpr static int arity() { return sizeof...(Args); }

  typedef std::tuple<Args...> ArgsTuple;

  template <size_t i>
  struct arg
  {
      typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
      // the i-th argument is equivalent to the i-th tuple element of a tuple
      // composed of those arguments.
  };
};

/**
 * Use extract_arg_by_filtered_index to return the i-th argument whose
 * type fulfills a given type trait. The argument itself is perfectly forwarded.
 *
 * Example:
 * std::string arg1 = "Hello";
 * std::string arg2 = "World";
 * std::string&& result = extract_arg_by_filtered_index<is_string, 1>(0, arg1, 2.0, std::move(arg2));
 *
 * Warning: Taking the result by rvalue reference can cause segfaults because ownership will not be passed on
 *          from the original reference. The original reference dies after the expression and the resulting
 */
namespace detail {
template<template <class> class Condition, size_t index, class Enable, class... Args> struct extract_arg_by_filtered_index_;
template<template <class> class Condition, size_t index, class Head, class... Tail>
struct extract_arg_by_filtered_index_<Condition, index, std::enable_if_t<!Condition<Head>::value>, Head, Tail...> {
  static decltype(auto) call(Head&& /*head*/, Tail&&... tail) {
    return extract_arg_by_filtered_index_<Condition, index, void, Tail...>::call(std::forward<Tail>(tail)...);
  }
};
template<template <class> class Condition, size_t index, class Head, class... Tail>
struct extract_arg_by_filtered_index_<Condition, index, std::enable_if_t<Condition<Head>::value && index != 0>, Head, Tail...> {
  static decltype(auto) call(Head&& /*head*/, Tail&&... tail) {
    return extract_arg_by_filtered_index_<Condition, index-1, void, Tail...>::call(std::forward<Tail>(tail)...);
  }
};
template<template <class> class Condition, size_t index>
struct extract_arg_by_filtered_index_<Condition, index, void> {
  static void call() {
    static_assert(index != index, "extract_arg_by_filtered_index out of range.");
  }
};
template<template <class> class Condition, size_t index, class Head, class... Tail>
struct extract_arg_by_filtered_index_<Condition, index, std::enable_if_t<Condition<Head>::value && index == 0>, Head, Tail...> {
  static decltype(auto) call(Head&& head, Tail&&... /*tail*/) {
    return std::forward<Head>(head);
  }
};
}
template<template <class> class Condition, size_t index, class... Args>
decltype(auto) extract_arg_by_filtered_index(Args&&... args) {
  static_assert(is_type_condition<Condition>::value, "In extract_arg_by_filtered_index, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");
  return detail::extract_arg_by_filtered_index_<Condition, index, void, Args...>::call(std::forward<Args>(args)...);
}



/**
 * Use filter_map to map a subset of the arguments to values.
 * The subset is defined by type traits, and will be evaluated at compile time.
 * At runtime, it will just loop over the pre-filtered arguments to create an std::array.
 *
 * Example:
 *  std::array<double, 2> result = filter_map<double, std::is_integral>([] (auto a) {return (double)a;}, 3, "bla", 4);
 *  // result == {3.0, 4.0}
 */
namespace detail {

template<class ResultType, size_t num_results> struct filter_map_ {
   template<template <class> class Condition, class Mapper, class... Args, size_t... INDEX>
   static guts::array<ResultType, num_results> call(const Mapper& mapper, std::index_sequence<INDEX...>, Args&&... args) {
     return guts::array<ResultType, num_results> { mapper(extract_arg_by_filtered_index<Condition, INDEX>(std::forward<Args>(args)...))... };
   }
};
template<class ResultType> struct filter_map_<ResultType, 0> {
  template<template <class> class Condition, class Mapper, class... Args, size_t... INDEX>
  static guts::array<ResultType, 0> call(const Mapper& /*mapper*/, std::index_sequence<INDEX...>, Args&&... /*args*/) {
    return guts::array<ResultType, 0> { };
  }
};
}

template<class ResultType, template <class> class Condition, class Mapper, class... Args>
decltype(auto) filter_map(const Mapper& mapper, Args&&... args) {
  static_assert(is_type_condition<Condition>::value, "In filter_map<Result, Condition>, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");

  static constexpr size_t num_results = typelist::count_if<Condition, typelist::typelist<Args...>>::value;
  return detail::filter_map_<ResultType, num_results>::template call<Condition, Mapper, Args...>(mapper, std::make_index_sequence<num_results>(), std::forward<Args>(args)...);
}

}}
