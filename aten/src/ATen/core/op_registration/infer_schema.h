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
     !std::is_integral<T>::value || std::is_same<T, int64_t>::value || std::is_same<T, bool>::value,
     "INVALID TYPE: Only int64_t and bool are supported as an integral argument type");
 static_assert(
     !std::is_same<T, float>::value,
     "INVALID TYPE: float is not supported as an argument type, use double instead");
}

template <typename... Ts, size_t... Is>
::std::vector<Argument> createArgumentVectorFromTypes(guts::index_sequence<Is...>) {
  // Check types for common errors
  (void)std::initializer_list<int>{(
    checkStaticTypes<Ts>()
  , 0)...};

  // Arguments are named "_<index>"
  return {Argument("_" + std::to_string(Is), getTypePtr<guts::decay_t<Ts>>())...};
}

template<class ParameterTypes> struct createArguments final {};

template<class... ParameterTypes>
struct createArguments<guts::typelist::typelist<ParameterTypes...>> final {
  static std::vector<Argument> call() {
    return createArgumentVectorFromTypes<ParameterTypes...>(
        guts::make_index_sequence<sizeof...(ParameterTypes)>()
    );
  }
};

template<class ReturnType, class Enable = void> struct createReturns final {};

template<class... ReturnTypes>
struct createReturns<std::tuple<ReturnTypes...>, void> final {
  static std::vector<Argument> call() {
    return createArgumentVectorFromTypes<ReturnTypes...>(
        guts::make_index_sequence<sizeof...(ReturnTypes)>()
    );
  }
};

template<class ReturnType>
struct createReturns<ReturnType, guts::enable_if_t<!std::is_same<void, ReturnType>::value && !guts::is_instantiation_of<std::tuple, ReturnType>::value>> final {
  static std::vector<Argument> call() {
    return createReturns<std::tuple<ReturnType>>::call();
  }
};

template<>
struct createReturns<void, void> final {
  static std::vector<Argument> call() {
    return createReturns<std::tuple<>>::call();
  }
};

/// Creates a `FunctionSchema` object from a `FunctionTraits` type for a
/// function.
template <typename FunctionTraits>
FunctionSchema createFunctionSchemaFromTraits(std::string name, std::string overload_name) {
 using ReturnType = typename FunctionTraits::return_type;
 using ParameterTypes = typename FunctionTraits::parameter_types;

 auto arguments = createArguments<ParameterTypes>::call();
 auto returns = createReturns<ReturnType>::call();

 return {std::move(name), std::move(overload_name), std::move(arguments), std::move(returns)};
}
}

template<class FuncType>
FunctionSchema inferFunctionSchema(std::string name, std::string overload_name) {
  return detail::createFunctionSchemaFromTraits<guts::infer_function_traits_t<FuncType>>(std::move(name), std::move(overload_name));
}

void assertSchemasHaveSameSignature(const FunctionSchema& inferred, const FunctionSchema& specified);

}
