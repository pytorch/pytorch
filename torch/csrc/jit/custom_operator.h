#pragma once

#include <torch/csrc/jit/function_schema.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/stack.h>
#include <torch/csrc/utils/variadic.h>

#include <caffe2/utils/Metaprogramming.h>
#include <caffe2/utils/TypeList.h>

namespace torch { namespace jit {
namespace detail {
template <typename... Ts, size_t... Is>
std::vector<Argument> createArgumentVectorFromTypes(Indices<Is...> indices) {
  // Arguments are named "_<index>"
  return {Argument("_" + std::to_string(Is), getTypePtr<Ts>())...};
}

template <typename... Ts, size_t... Is>
std::vector<Argument> createReturns(Indices<Is...> indices) {
  return createArgumentVectorFromTypes<Ts..., Is...>();
}

/// Unpack a tuple return type into a vector of return types, one per tuple
/// element.
template <typename... Ts>
std::vector<Argument> createReturns(std::tuple<Ts...>* tuple) {
  // Create an index pack so we can call `get<Indices>` on the tuple next.
  return createReturns<Ts...>(typename MakeIndices<sizeof...(Ts)>::indices{});
}

/// Create a single-element `vector` for simple (non-tuple) return types.
template <typename ReturnType>
std::vector<Argument> createReturns(ReturnType*) {
  return {Argument("_1", getTypePtr<ReturnType>())};
}

/// Creates a vector of `Argument` from `FunctionTraits` and a pack of indices
/// into the argument list.
template <typename FunctionTraits, size_t... Is>
std::vector<Argument> createArgumentVectorFromTraits(Indices<Is...> indices) {
  using ArgumentTypes = typename FunctionTraits::parameter_types;
  return createArgumentVectorFromTypes<
      c10::guts::typelist::element_t<Is, ArgumentTypes>...>(indices);
}

/// Creates a `FunctionSchema` object from a `FunctionTraits` type for a
/// function.
template <typename FunctionTraits>
FunctionSchema createFunctionSchemaFromTraits(const std::string& name) {
  using ReturnType = typename FunctionTraits::return_type;
  auto arguments = createArgumentVectorFromTraits<FunctionTraits>(
      typename MakeIndices<FunctionTraits::number_of_parameters>::indices{});
  auto returns = createReturns(static_cast<ReturnType*>(nullptr));
  return {name, arguments, returns};
}

/// Does two things for an operator implementation and a tuple of arguments:
/// 1. Pops all necessary arguments off the stack into the tuple's elements,
/// 2. Unpacks the tuple and calls the operator implementation.
/// The result of the implementation call is returned.
template <
    typename ReturnType,
    typename Implementation,
    typename... Types,
    size_t... Is>
ReturnType callOperatorWithTuple(
    Implementation&& implementation,
    Stack& stack,
    std::tuple<Types...>& tuple,
    Indices<Is...>) {
  pop(stack, std::get<Is>(tuple)...);
  return std::forward<Implementation>(implementation)(std::get<Is>(tuple)...);
}
} // namespace detail

/// Low-level interface to register an operator with a parsed `FunctionSchema`
/// and a stack-based operator implementation (the `operation`). The `operation`
/// must pop its arguments from the stack, perform some operation on those
/// arguments, and then push the return value back onto the stack.
inline Operator createOperatorWithStack(
    FunctionSchema schema,
    Operation operation) {
  return {schema, [operation](Node*) { return operation; }};
}

/// Registers a custom operator with a schema and an implementation function.
/// The implementation function can be a function pointer or a functor
/// (including a lambda object). The function (or `operator()`) can take any
/// number of arguments with a type from the subset accepted by the PyTorch
/// JIT/Script backend, and return a single type or a tuple of types.
/// Example invocation:
/// ```
/// registerOperator(
///    parseSchema("foo::bar(float a, Tensor b)"),
///    [](float a, at::Tensor b) { return a + b; });
/// ```
template <typename Implementation>
Operator createOperator(FunctionSchema schema, Implementation&& implementation) {
  using Traits = c10::guts::infer_function_traits_t<Implementation>;
  using ArgumentTypes = typename Traits::parameter_types;
  using ArgumentTuple =
      typename c10::guts::typelist::to_tuple<ArgumentTypes>::type;
  using ReturnType = typename Traits::return_type;
  return createOperatorWithStack(schema, [implementation](Stack& stack) {
    ArgumentTuple tuple;
    auto result = torch::jit::detail::callOperatorWithTuple<ReturnType>(
        std::move(implementation),
        stack,
        tuple,
        typename MakeIndices<std::tuple_size<ArgumentTuple>::value>::indices{});
    pack(stack, std::move(result));
    return 0;
  });
}

/// Registers a custom operator with a name and an implementation function. The
/// schema, including the type of each argument and the return type, is inferred
/// from the function signature. The argument names are not automatically
/// inferred and by default take on sequential placeholder names like `_1`, `_2`
/// and so on. If you want function names to be preserved, use the overload of
/// this function that takes an explicit schema. The implementation function (or
/// `operator()`) can take any number of arguments with a type from the subset
/// accepted by the PyTorch JIT/Script backend, and return a single type or a
/// tuple of types. Example invocation:
/// ```
/// registerOperator("foo::bar", [](float a, at::Tensor b) { return a + b; });
/// ```
template <typename Implementation>
Operator createOperator(
    const std::string& name,
    Implementation&& implementation) {
  using Traits = c10::guts::infer_function_traits_t<Implementation>;
  auto schema =
      torch::jit::detail::createFunctionSchemaFromTraits<Traits>(name);
  return createOperator(schema, std::forward<Implementation>(implementation));
}

} // namespace jit
} // namespace torch
