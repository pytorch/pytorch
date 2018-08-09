#pragma once

#include <torch/csrc/jit/function_schema.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/stack.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/variadic.h>

#include <caffe2/utils/Metaprogramming.h>
#include <caffe2/utils/TypeList.h>

namespace torch { namespace jit {
namespace detail {
template <typename... Ts, size_t... Is>
std::vector<Argument> createArgumentVectorFromTypes(Indices<Is...> indices) {
  // Arguments are named "_<index>"
  return {Argument("_" + std::to_string(Is), getTypePtr<decay_t<Ts>>())...};
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
  return {Argument("_1", getTypePtr<decay_t<ReturnType>>())};
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

template <size_t... Is, typename... Types>
Node* getTracedNode(
    const FunctionSchema& schema,
    const std::tuple<Types...>& tuple) {
  auto symbol = Symbol::fromQualString(schema.name);
  const auto& graph = tracer::getTracingState()->graph;
  Node* node = graph->create(std::move(symbol), /*outputs=*/0);
  tracer::recordSourceLocation(node);

  // Hack to call addInputs for the parameter pack in a sequenced fashion.
  // https://stackoverflow.com/questions/12030538/calling-a-function-for-each-variadic-template-argument-and-an-array
  int _[] = {(tracer::addInputs(node, schema.arguments[Is].name.c_str(), std::get<Is>(tuple)), 0)...};
  (void)_;

  graph->appendNode(node);

  return node;
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
    const FunctionSchema& schema,
    Implementation&& implementation,
    Stack& stack,
    std::tuple<Types...>& tuple,
    Indices<Is...>) {
  Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    node = getTracedNode<Is...>(schema, tuple);
  }

  pop(stack, std::get<Is>(tuple)...);
  auto result =
      std::forward<Implementation>(implementation)(std::get<Is>(tuple)...);

  if (jit::tracer::isTracing()) {
    jit::tracer::postRecordTrace(node, result);
  }

  return result;
}

void checkArgumentVector(
    const char* what,
    const std::vector<Argument>& inferred,
    const std::vector<Argument>& provided,
    const FunctionSchema& inferredSchema,
    const FunctionSchema& providedSchema) {
  AT_CHECK(
      inferred.size() == provided.size(),
      "Inferred ", inferred.size(), " ", what,
      "(s) for operator implementation, but the provided schema specified ",
      provided.size(), " ", what, "(s). Inferred schema: ",
      inferredSchema, " | Provided schema: ", providedSchema);
  for (size_t i = 0; i < provided.size(); ++i) {
    AT_CHECK(
        provided[i].type->isSubtypeOf(inferred[i].type),
        "Inferred type for ", what, " #", i, " was ",
        *inferred[i].type, ", but the provided schema specified type ",
        *provided[i].type, " for the ", what,
        " in that position. Inferred schema: ",
        inferredSchema, " | Provided schema: ", providedSchema);
  }
}

/// If `schemaOrName` contains a `(`, it is assumed it specifies a schema, else
/// it is assumed it only specifies the name. In the case where it is a full
/// schema (assumed), we nevertheless infer the schema and verify that the user
/// made no mistakes. Either way, this function returns the final schema.
template <typename Traits>
FunctionSchema inferAndCheckSchema(const std::string& schemaOrName) {
  // If there is no '(' in the schema, we assume this is only the name (e.g.
  // "foo::bar").
  const auto bracketIndex = schemaOrName.find('(');
  if (bracketIndex == std::string::npos) {
    // Infer the full schema and we're good.
    return torch::jit::detail::createFunctionSchemaFromTraits<Traits>(
        /*name=*/schemaOrName);
  }

  // If the user provided her own schema, we need to infer it nevertheless and
  // check that it's correct. We return the user provided schema in the end
  // because it has proper argument names.

  auto providedSchema = parseSchema(schemaOrName);

  const auto inferredSchema =
      torch::jit::detail::createFunctionSchemaFromTraits<Traits>(
          providedSchema.name);
  checkArgumentVector(
      "argument",
      inferredSchema.arguments,
      providedSchema.arguments,
      inferredSchema,
      providedSchema);
  checkArgumentVector(
      "return value",
      inferredSchema.returns,
      providedSchema.returns,
      inferredSchema,
      providedSchema);
  return providedSchema;
}
} // namespace detail

/// Registers a custom operator with a name or schema, and an implementation
/// function.
///
/// If the first argument specifies only the function name like `foo::bar`, the
/// schema, including the type of each argument and the return type, is inferred
/// from the function signature. Otherwise, the string should specify the whole
/// schema, like `foo::bar(Tensor a, double b) -> Tensor`. In that case, the
/// schema will still be inferred from the function and checked against this
/// provided schema.
///
/// If the schema is left to be inferred, the argument names will take on
/// sequential placeholder names like `_0`, `_1`, '_2' and so on. If you want
/// argument names to be preserved, you should provide the schema yourself.
///
/// The implementation function can be a function pointer or a functor
/// (including a lambda object). The function (or `operator()`) can take any
/// number of arguments with a type from the subset accepted by the PyTorch
/// JIT/Script backend, and return a single type or a tuple of types.
///
/// Example invocation:
/// ```
/// createOperator(
///    "foo::bar(float a, Tensor b)",
///    [](float a, at::Tensor b) { return a + b; });
/// ```
template <typename Implementation>
Operator createOperator(
    const std::string& schemaOrName,
    Implementation&& implementation) {
  using Traits = c10::guts::infer_function_traits_t<Implementation>;
  using ArgumentTypes =
      c10::guts::typelist::map_t<decay_t, typename Traits::parameter_types>;
  using ArgumentTuple =
      typename c10::guts::typelist::to_tuple<ArgumentTypes>::type;
  using ReturnType = decay_t<typename Traits::return_type>;

  auto schema = torch::jit::detail::inferAndCheckSchema<Traits>(schemaOrName);

  return Operator(schema, [implementation, schema](Stack& stack) {
    ArgumentTuple tuple;
    auto result = torch::jit::detail::callOperatorWithTuple<ReturnType>(
        schema,
        std::move(implementation),
        stack,
        tuple,
        typename MakeIndices<std::tuple_size<ArgumentTuple>::value>::indices{});
    pack(stack, std::move(result));
    return 0;
  });
}
} // namespace jit
} // namespace torch
