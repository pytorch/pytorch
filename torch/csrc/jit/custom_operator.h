#pragma once

#include <torch/csrc/jit/function_schema.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/stack.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/variadic.h>

#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>

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

/// Adds the elements of the `tuple` as input nodes to the traced graph.
template <size_t... Is, typename... Types>
Node* getTracedNode(
    const FunctionSchema& schema,
    const std::tuple<Types...>& tuple) {
  auto symbol = Symbol::fromQualString(schema.name());
  const auto& graph = tracer::getTracingState()->graph;
  Node* node = graph->create(std::move(symbol), /*num_outputs=*/0);
  tracer::recordSourceLocation(node);

  // Hack to call addInputs for the parameter pack in a sequenced fashion.
  // https://stackoverflow.com/questions/12030538/calling-a-function-for-each-variadic-template-argument-and-an-array
  int _[] = {
      (tracer::addInputs(
           node, schema.arguments()[Is].name().c_str(), std::get<Is>(tuple)),
       0)...};
  (void)_; // ignore

  graph->appendNode(node);

  return node;
}

/// Does two things for an operator implementation and a tuple of arguments:
/// 1. Pops all necessary arguments off the stack into the tuple's elements,
/// 2. Unpacks the tuple and calls the operator implementation.
/// If tracing is currently enabled, this function will also take care of
/// tracing the operator call.
template <typename Implementation, typename... Types, size_t... Is>
void callOperatorWithTuple(
    const FunctionSchema& schema,
    Implementation&& implementation,
    Stack& stack,
    std::tuple<Types...>& arguments,
    Indices<Is...>) {
  AT_ASSERT(stack.size() == sizeof...(Is));

  // Pop values from the stack into the elements of the tuple.
  pop(stack, std::get<Is>(arguments)...);

  Node* node = nullptr;
  if (jit::tracer::isTracing()) {
    node = getTracedNode<Is...>(schema, arguments);
  }

  // Call into the actual, original, user-supplied function.
  auto return_value =
      std::forward<Implementation>(implementation)(std::get<Is>(arguments)...);

  if (jit::tracer::isTracing()) {
    jit::tracer::addOutput(node, return_value);
  }

  // Push the return value back onto the stack.
  push(stack, IValue(std::move(return_value)));
}

inline void checkArgumentVector(
    const char* what,
    const std::vector<Argument>& inferred,
    const std::vector<Argument>& provided,
    const FunctionSchema& inferredSchema,
    const FunctionSchema& providedSchema) {
  // clang-format off
  AT_CHECK(
      inferred.size() == provided.size(),
      "Inferred ", inferred.size(), " ", what,
      "(s) for operator implementation, but the provided schema specified ",
      provided.size(), " ", what, "(s). Inferred schema: ", inferredSchema,
      " | Provided schema: ", providedSchema);
  // clang-format on
  for (size_t i = 0; i < provided.size(); ++i) {
    // clang-format off
    AT_CHECK(
        provided[i].type()->isSubtypeOf(inferred[i].type()),
        "Inferred type for ", what, " #", i, " was ", *inferred[i].type(),
        ", but the provided schema specified type ", *provided[i].type(),
        " for the ", what, " in that position. Inferred schema: ",
        inferredSchema, " | Provided schema: ", providedSchema);
    // clang-format on
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
          providedSchema.name());
  checkArgumentVector(
      "argument",
      inferredSchema.arguments(),
      providedSchema.arguments(),
      inferredSchema,
      providedSchema);
  checkArgumentVector(
      "return value",
      inferredSchema.returns(),
      providedSchema.returns(),
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
  static constexpr auto kNumberOfArguments =
      std::tuple_size<ArgumentTuple>::value;

  auto schema = torch::jit::detail::inferAndCheckSchema<Traits>(schemaOrName);

  return Operator(schema, [implementation, schema](Stack& stack) {
    ArgumentTuple tuple;
    torch::jit::detail::callOperatorWithTuple(
        schema,
        std::move(implementation), // NOLINT(bugprone-move-forwarding-reference)
        stack,
        tuple,
        typename MakeIndices<kNumberOfArguments>::indices{});
    return 0;
  });
}

/// Registration class for new operators. Effectively calls
/// `torch::jit::registerOperator` for every supplied operator, but allows doing
/// so in the global scope when a `RegisterOperators` object is assigned to a
/// static variable. Also handles registration of user-defined, "custom"
/// operators.
struct TORCH_API RegisterOperators {
  RegisterOperators() = default;

  /// Registers a vector of already created `Operator`s.
  RegisterOperators(std::vector<Operator> operators) {
    for (Operator& o : operators) {
      registerOperator(std::move(o));
    }
  }

  /// Calls `op(...)` with the given operator name and implementation.
  template <typename Implementation>
  RegisterOperators(const std::string& name, Implementation&& implementation) {
    op(name, std::forward<Implementation>(implementation));
  }

  /// Creates a new operator from a name and implementation function (function
  /// pointer or function object/lambda) using `torch::jit::createOperator`, and
  /// then registers the operator.
  template <typename Implementation>
  RegisterOperators& op(
      const std::string& name,
      Implementation&& implementation) {
    registerOperator(
        createOperator(name, std::forward<Implementation>(implementation)));
    return *this;
  }
};

} // namespace jit
} // namespace torch
