#pragma once

#include <tuple>

// Modified from https://stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda

// For generic types, directly use the result of the signature of its 'operator()'
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {
};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const> {
  // arity is the number of arguments.
  enum { arity = sizeof...(Args) };

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
