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
constexpr std::array<ArgumentDef, sizeof...(Ts)> createArgumentVectorFromTypes(std::index_sequence<Is...>) {
  return (
    // Check types for common errors
    checkStaticTypes<Ts...>(),

    // Create the return value
    std::array<ArgumentDef, sizeof...(Ts)>{{ArgumentDef{&getTypePtr_<std::decay_t<Ts>>::call}...}}
  );
}

/// Creates a vector of `ArgumentDef` from a list of C++ types that are specified
/// as template arguments.
template<class ParameterTypes> struct createArguments final {};
template<class... ParameterTypes>
struct createArguments<guts::typelist::typelist<ParameterTypes...>> final {
  static constexpr std::array<ArgumentDef, sizeof...(ParameterTypes)> call() {
    return createArgumentVectorFromTypes<ParameterTypes...>(
        std::make_index_sequence<sizeof...(ParameterTypes)>()
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
        std::make_index_sequence<sizeof...(ReturnTypes)>()
    );
  }
};

template<class ReturnType>
struct createReturns<ReturnType, std::enable_if_t<!std::is_same<void, ReturnType>::value && !guts::is_instantiation_of<std::tuple, ReturnType>::value>> final {
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

template <typename ReturnType>
struct createSingleReturn {
  static constexpr std::array<ArgumentDef, 1> call() {
    return createArgumentVectorFromTypes<ReturnType>(std::make_index_sequence<1>());
  }
};

C10_API FunctionSchema make_function_schema(std::string&& name, std::string&& overload_name, c10::ArrayRef<ArgumentDef> arguments, c10::ArrayRef<ArgumentDef> returns);

/// Creates a `FunctionSchema` object from a `FunctionTraits` type for a
/// function. Flattens std::tuple returns into multiple return types
template <typename FunctionTraits>
FunctionSchema createFunctionSchemaFromTraitsFlattenedReturns(std::string&& name, std::string&& overload_name) {
 using ReturnType = typename FunctionTraits::return_type;
 using ParameterTypes = typename FunctionTraits::parameter_types;

 // arguments and returns are computed into a std::array at compile time and embedded into the binary.
 // The only code executed at runtime here is the one that creates a std::vector
 // of the arguments/returns from the std::array.
 constexpr auto arguments = createArguments<ParameterTypes>::call();
 constexpr auto returns = createReturns<ReturnType>::call();

 return make_function_schema(std::move(name), std::move(overload_name), arguments, returns);
}

/// Creates a `FunctionSchema` object from a `FunctionTraits` type for a
/// function. Preserves std::tuple returns as a Tuple return type
template <typename FunctionTraits>
FunctionSchema createFunctionSchemaFromTraitsSingleReturn(std::string&& name, std::string&& overload_name) {
 using ReturnType = typename FunctionTraits::return_type;
 using ParameterTypes = typename FunctionTraits::parameter_types;

 // arguments and returns are computed into a std::array at compile time and embedded into the binary.
 // The only code executed at runtime here is the one that creates a std::vector
 // of the arguments/returns from the std::array.
 constexpr auto arguments = createArguments<ParameterTypes>::call();
 constexpr auto returns = createSingleReturn<ReturnType>::call();

 return make_function_schema(std::move(name), std::move(overload_name), arguments, returns);
}

}
}

template<class FuncType>
FunctionSchema inferFunctionSchemaFlattenedReturns(std::string&& name, std::string&& overload_name) {
  return detail::infer_schema::createFunctionSchemaFromTraitsFlattenedReturns<guts::infer_function_traits_t<FuncType>>(std::move(name), std::move(overload_name));
}

template<class FuncType>
FunctionSchema inferFunctionSchemaSingleReturn(std::string&& name, std::string&& overload_name) {
  return detail::infer_schema::createFunctionSchemaFromTraitsSingleReturn<guts::infer_function_traits_t<FuncType>>(std::move(name), std::move(overload_name));
}

CAFFE2_API c10::optional<std::string> findSchemaDifferences(const FunctionSchema& inferred, const FunctionSchema& specified);

}
