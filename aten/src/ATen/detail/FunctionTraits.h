#pragma once

#include <tuple>

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

// Free functions
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  // arity is the number of arguments.
  enum { arity = sizeof...(Args) };

  typedef std::tuple<Args...> ArgsTuple;
  typedef ReturnType result_type;

  template <size_t i>
  struct arg
  {
      typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
      // the i-th argument is equivalent to the i-th tuple element of a tuple
      // composed of those arguments.
  };
};

template <typename T>
struct nullary_function_traits {
  using traits = function_traits<T>;
  using result_type = typename traits::result_type;
};

template <typename T>
struct unary_function_traits {
  using traits = function_traits<T>;
  using result_type = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
};

template <typename T>
struct binary_function_traits {
  using traits = function_traits<T>;
  using result_type = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
};
