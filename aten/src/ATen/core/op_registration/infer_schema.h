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

namespace infer_schema {

/// The templated inference code creates `ArgumentDef` instead of `Argument`,
/// because that can be constructed at compile time and has a much smaller
/// binary size than having calls to `Argument` constructors in the template.
/// Creating `Argument` objects from `ArgumentDef` can then be done at
/// runtime in a non-templated way.
struct ArgumentDef final {
  using GetTypeFn = TypePtr();
  GetTypeFn* getTypeFn;
};

template<bool V>
struct bool_t {};
template<> struct bool_t<true> : std::true_type {};
template<> struct bool_t<false> : std::false_type {};

/// Checks the static C++ types `Types` for correctness to catch common error cases.
template <class... Types>
constexpr int checkStaticTypes() {
 // Give nice error messages for some of the common error cases.
 // Use a LOUD ERROR MESSAGE SO USERS SEE THE STATIC_ASSERT
 static_assert(guts::conjunction<
     bool_t<!std::is_integral<Types>::value || std::is_same<Types, int64_t>::value || std::is_same<Types, bool>::value>...
   >::value, "INVALID TYPE: Only int64_t and bool are supported as an integral argument type");
 static_assert(guts::conjunction<
     bool_t<!std::is_same<Types, float>::value>...
   >::value, "INVALID TYPE: float is not supported as an argument type, use double instead");
 return 0;
}

template <typename... Ts, size_t... Is>
constexpr std::array<ArgumentDef, sizeof...(Ts)> createArgumentVectorFromTypes(guts::index_sequence<Is...>) {
  return (
    // Check types for common errors
    checkStaticTypes<Ts...>(),

    // Create the return value
    std::array<ArgumentDef, sizeof...(Ts)>{{ArgumentDef{&getTypePtr_<guts::decay_t<Ts>>::call}...}}
  );
}

/// Creates a vector of `ArgumentDef` from a list of C++ types that are specified
/// as template arguments.
template<class ParameterTypes> struct createArguments final {};
template<class... ParameterTypes>
struct createArguments<guts::typelist::typelist<ParameterTypes...>> final {
  static constexpr std::array<ArgumentDef, sizeof...(ParameterTypes)> call() {
    return createArgumentVectorFromTypes<ParameterTypes...>(
        guts::make_index_sequence<sizeof...(ParameterTypes)>()
    );
  }
};

/// Creates a vector of `ArgumentDef` from a list of C++ types that are specified
/// as a tuple (i.e. in the way c10 kernels return values).
/// It can be a tuple<A, B, C> if there's three output arguments with types A, B, C.
/// It can be an empty tuple<>, or void for kernels that don't return anything.
/// It can be a single type A (i.e. no tuple) for the case where a kernel just
/// returns one value.
template<class ReturnTypeTuple, class Enable = void> struct createReturns final {};

template<class... ReturnTypes>
struct createReturns<std::tuple<ReturnTypes...>, void> final {
  static constexpr std::array<ArgumentDef, sizeof...(ReturnTypes)> call() {
    return createArgumentVectorFromTypes<ReturnTypes...>(
        guts::make_index_sequence<sizeof...(ReturnTypes)>()
    );
  }
};

template<class ReturnType>
struct createReturns<ReturnType, guts::enable_if_t<!std::is_same<void, ReturnType>::value && !guts::is_instantiation_of<std::tuple, ReturnType>::value>> final {
  static constexpr std::array<ArgumentDef, 1> call() {
    return createReturns<std::tuple<ReturnType>>::call();
  }
};

template<>
struct createReturns<void, void> final {
  static constexpr std::array<ArgumentDef, 0> call() {
    return createReturns<std::tuple<>>::call();
  }
};

template<size_t NumArgs>
std::vector<Argument> createArgumentVector(const std::array<ArgumentDef, NumArgs>& args) {
  std::vector<Argument> result;
  result.reserve(NumArgs);
  for (size_t i = 0; i < args.size(); ++i) {
    // Arguments are named "_<index>"
    result.push_back(Argument("_" + c10::guts::to_string(i), (*args[i].getTypeFn)()));
  }
  return result;
}

// This is intentionally a separate function
// because then the template is smaller and that benefits binary size
inline FunctionSchema make_function_schema(std::string&& name, std::string&& overload_name, std::vector<Argument>&& arguments, std::vector<Argument>&& returns) {
  return FunctionSchema(std::move(name), std::move(overload_name), std::move(arguments), std::move(returns));
}

template<size_t NumArgs, size_t NumReturns>
inline FunctionSchema make_function_schema(std::string&& name, std::string&& overload_name, const std::array<ArgumentDef, NumArgs>& arguments, const std::array<ArgumentDef, NumReturns>& returns) {
  return make_function_schema(std::move(name), std::move(overload_name), createArgumentVector(arguments), createArgumentVector(returns));
}

/// Creates a `FunctionSchema` object from a `FunctionTraits` type for a
/// function.
template <typename FunctionTraits>
FunctionSchema createFunctionSchemaFromTraits(std::string&& name, std::string&& overload_name) {
 using ReturnType = typename FunctionTraits::return_type;
 using ParameterTypes = typename FunctionTraits::parameter_types;

 constexpr auto arguments = createArguments<ParameterTypes>::call();
 constexpr auto returns = createReturns<ReturnType>::call();

 return make_function_schema(std::move(name), std::move(overload_name), arguments, returns);
}
}
}

template<class FuncType>
FunctionSchema inferFunctionSchema(std::string&& name, std::string&& overload_name) {
  return detail::infer_schema::createFunctionSchemaFromTraits<guts::infer_function_traits_t<FuncType>>(std::move(name), std::move(overload_name));
}

CAFFE2_API c10::optional<std::string> findSchemaDifferences(const FunctionSchema& inferred, const FunctionSchema& specified);

}
