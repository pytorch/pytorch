#pragma once

/**
 * This file contains functionality to take a C++ function and infer its
 * c10::FunctionSchema.
 */

#include <ATen/core/function_schema.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>

namespace c10 {

namespace detail {
/// Checks the static C++ type `T` for correctness to catch common error cases.
template <typename T>
void checkStaticTypes() {
 // Give nice error messages for some of the common error cases.
 // Use a LOUD ERROR MESSAGE SO USERS SEE THE STATIC_ASSERT
 static_assert(
     !std::is_integral<T>::value || std::is_same<T, int64_t>::value,
     "INVALID TYPE: Only int64_t is supported as an integral argument type");
 static_assert(
     !std::is_same<T, float>::value,
     "INVALID TYPE: float is not supported as an argument type, use double instead");
}

template <typename First, typename Second, typename... Rest>
void checkStaticTypes() {
 checkStaticTypes<First>();
 checkStaticTypes<Second, Rest...>();
}

template <typename... Ts, size_t... Is>
::std::vector<Argument> createArgumentVectorFromTypes(guts::index_sequence<Is...>) {
  checkStaticTypes<guts::decay_t<Ts>...>();
  // Arguments are named "_<index>"
  return {Argument("_" + std::to_string(Is), getTypePtr<guts::decay_t<Ts>>())...};
}

template <typename... Ts, size_t... Is>
::std::vector<Argument> createReturns(guts::index_sequence<Is...>) {
  return createArgumentVectorFromTypes<Ts..., Is...>();
}

/// Unpack a tuple return type into a vector of return types, one per tuple
/// element.
template <typename... Ts>
::std::vector<Argument> createReturns(std::tuple<Ts...>* tuple) {
  return createReturns<Ts...>(guts::make_index_sequence<sizeof...(Ts)>());
}

/// Create a single-element `vector` for simple (non-tuple) return types.
template <typename ReturnType>
::std::vector<Argument> createReturns(ReturnType*) {
  checkStaticTypes<guts::decay_t<ReturnType>>();
  return {Argument("_1", getTypePtr<guts::decay_t<ReturnType>>())};
}

/// Creates a vector of `Argument` from `FunctionTraits` and a pack of indices
/// into the argument list.
template <typename FunctionTraits, size_t... Is>
::std::vector<Argument> createArgumentVectorFromTraits(guts::index_sequence<Is...> indices) {
 using ArgumentTypes = typename FunctionTraits::parameter_types;
 return createArgumentVectorFromTypes<
     c10::guts::typelist::element_t<Is, ArgumentTypes>...>(indices);
}

/// Creates a `FunctionSchema` object from a `FunctionTraits` type for a
/// function.
template <typename FunctionTraits>
FunctionSchema createFunctionSchemaFromTraits(std::string name, std::string overload_name) {
 using ReturnType = typename FunctionTraits::return_type;

 auto arguments = createArgumentVectorFromTraits<FunctionTraits>(
     guts::make_index_sequence<FunctionTraits::number_of_parameters>());
 auto returns = createReturns(static_cast<ReturnType*>(nullptr));

 return {std::move(name), std::move(overload_name), std::move(arguments), std::move(returns)};
}
}

template<class FuncType>
FunctionSchema inferFunctionSchema(std::string name, std::string overload_name) {
  return detail::createFunctionSchemaFromTraits<guts::infer_function_traits_t<FuncType>>(std::move(name), std::move(overload_name));
}

}
