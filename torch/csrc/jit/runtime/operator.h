// in memory description of all ATen Ops similar to Caffe2 schema
// once C10 exists this can be removed, or stubbed out, but we need
// it now to implement correct semantic checking for script
#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/op_registration/op_whitelist.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/library.h>

#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/interned_strings.h>

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

struct Node;
using ::c10::FunctionSchema;
using ::c10::Symbol;

using OperationCreator = Operation (*)(const Node*);

/*
 * Note: JIT relies on Operator instances having static lifetime, because
 * it for example stores a non-owning FunctionSchema* pointer in the Node class,
 * which points to the function shema stored in the Operator instance.
 * Also, jit::Operator is meant to store more operator related information like
 * symbolic derivatives, which also requires them to have static lifetime
 * so that changes to symbolic derivatives are remembered.
 *
 * Currently, the JIT operator library contains a jit::Operator instance
 * with a wrapper for each c10 operator. The c10 operator library registers
 * those wrappers using listeners in register_c10_ops.cpp.
 * TODO Instead of doing it this way, we should only have pure-jit ops in
 * the jit library but have the JIT operator lookup look into the c10 library
 * too.
 */

// An Operator is a thin wrapper around either a pure JIT operator (e.g. prim
// ops) or a c10 operator, allowing some common operations and abstracting away
// the concrete operator nature.
struct TORCH_API Operator {
 private:
  struct C10Operator final {
    c10::OperatorHandle handle_;
    Operation op_;
  };
  struct UnparsedFunctionSchema final {
    std::string schema_string_;
    mutable c10::optional<c10::AliasAnalysisKind> alias_analysis_;
  };
  struct JitOnlyOperator final {
    // The only valid transition for schema_ is from right->left, i.e.
    // when the schema gets parsed.
    mutable c10::either<FunctionSchema, UnparsedFunctionSchema> schema_;

    c10::either<Operation, OperationCreator> op_;
  };

 public:
  Operator(c10::OperatorHandle opHandle, Operation operation)
      : op_(c10::make_left<C10Operator, JitOnlyOperator>(
            C10Operator{std::move(opHandle), std::move(operation)})) {}

  Operator(
      std::string schema,
      Operation op,
      c10::AliasAnalysisKind alias_analysis)
      : op_(c10::make_right<C10Operator, JitOnlyOperator>(JitOnlyOperator{
            c10::make_right<FunctionSchema, UnparsedFunctionSchema>(
                UnparsedFunctionSchema{std::move(schema), alias_analysis}),
            c10::make_left<Operation, OperationCreator>(std::move(op))})) {}

  C10_DEPRECATED_MESSAGE(
      "Please define your operator as taking a `Stack*` argument instead of `Stack&` and as returning `void` instead of `int`.")
  Operator(
      std::string schema,
      std::function<int(Stack&)> op,
      c10::AliasAnalysisKind alias_analysis)
      : Operator(
            std::move(schema),
            [op = std::move(op)](Stack* stack) { op(*stack); },
            alias_analysis) {}

  Operator(
      std::string schema,
      OperationCreator op_creator,
      c10::AliasAnalysisKind alias_analysis)
      : op_(c10::make_right<C10Operator, JitOnlyOperator>(JitOnlyOperator{
            c10::make_right<FunctionSchema, UnparsedFunctionSchema>(
                UnparsedFunctionSchema{std::move(schema), alias_analysis}),
            c10::make_right<Operation, OperationCreator>(
                std::move(op_creator))})) {}

  // Helper constructor to register `op` to run
  // run for _every_ IR Node where n.kind() == name, regardless of arguments.
  // This is accomplished by marking the schema varargs and having no required
  // arguments.
  Operator(
      Symbol name,
      OperationCreator op_creator,
      c10::AliasAnalysisKind alias_analysis)
      : op_(c10::make_right<C10Operator, JitOnlyOperator>(JitOnlyOperator{
            c10::make_left<FunctionSchema, UnparsedFunctionSchema>(
                varArgSchemaWithName(name, alias_analysis)),
            c10::make_right<Operation, OperationCreator>(
                std::move(op_creator))})) {}

  Operation getOperation(const Node* node = nullptr) const {
    return op_.fold<Operation>(
        [](const C10Operator& op) { return op.op_; },
        [node](const JitOnlyOperator& op) {
          return op.op_.fold<Operation>(
              [](const Operation& op) { return op; },
              [node](const OperationCreator& op_creator) {
                return op_creator(node);
              });
        });
  }

  const FunctionSchema& schema() const {
    return op_.fold<const FunctionSchema&>(
        [](const C10Operator& op) -> const FunctionSchema& {
          return op.handle_.schema();
        },
        [](const JitOnlyOperator& op) -> const FunctionSchema& {
          // we lazily parse schema initialized from strings so that
          // we do less work during static operator registration
          if (op.schema_.is_right()) {
            auto& unmaterializedSchema = op.schema_.right();
            FunctionSchema schema =
                parseSchema(unmaterializedSchema.schema_string_);
            if (unmaterializedSchema.alias_analysis_.has_value()) {
              // TODO What if it gets set later?
              schema.setAliasAnalysis(*unmaterializedSchema.alias_analysis_);
            }
            op.schema_ = c10::make_left<FunctionSchema, UnparsedFunctionSchema>(
                std::move(schema));
          }
          return op.schema_.left();
        });
  }

  bool isC10Op() const {
    return op_.is_left();
  }

  c10::AliasAnalysisKind aliasAnalysisKind() const {
    const FunctionSchema& schemaRef = schema();
    c10::AliasAnalysisKind alias_analysis = schemaRef.aliasAnalysis();

    TORCH_CHECK(
        alias_analysis == AliasAnalysisKind::FROM_SCHEMA ||
            !schemaRef.hasAnyAliasInfo(),
        "In operator registration: Tried to register operator ",
        schemaRef,
        " with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA.");
    return alias_analysis;
  }

  bool hasOperation() const {
    return op_.fold<bool>(
        [](const C10Operator&) { return true; },
        [](const JitOnlyOperator& op) { return op.op_.is_left(); });
  }

 private:
  static FunctionSchema varArgSchemaWithName(
      Symbol name,
      AliasAnalysisKind alias_analysis) {
    auto result = FunctionSchema(
        name,
        "",
        {},
        {},
        /*is_vararg*/ true,
        /*is_varret*/ true);
    result.setAliasAnalysis(alias_analysis);
    return result;
  }

  c10::either<C10Operator, JitOnlyOperator> op_;
};

TORCH_API std::string canonicalSchemaString(const FunctionSchema& schema);

TORCH_API const std::vector<std::shared_ptr<Operator>> getAllOperators();
TORCH_API const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(
    Symbol name);

// given a operator with an overload name, find the specific operator related to
// it, may return nullptr if no operator exists.
TORCH_API std::shared_ptr<Operator> findOperatorFor(
    const c10::OperatorName& full_name);

TORCH_API std::vector<Symbol> findSimilarOperators(Symbol input_op);

TORCH_API void registerOperator(Operator&& op);
TORCH_API void deregisterOperator(const FunctionSchema& schema);

// XXX: this function is meant to be used with string literals only!
TORCH_API std::shared_ptr<Operator> getOperatorForLiteral(
    const char* signature);

// Ensure the thing that registers c10 ops is defined.
// Otherwise, our registry will not have c10 ops. You can run into this
// scenario if you're querying registered ops during static init.
//
// This fn is defined in register_c10_ops.cpp
TORCH_API void ensure_c10_registerer_defined();

// Used to assert that unschematized operators have an analysis method written
TORCH_API bool aliasAnalysisHasSpecialCaseFor(c10::Symbol sym);

// A factory function to generate an optional operator. It has two
// instantiations depending on the template bool arg value. The arg can be a
// compile-time function for the selective op registration based on schema
// string.
template <typename Func>
c10::optional<Operator> OperatorGenerator(
    torch::detail::SelectiveStr<true> schema_str,
    Func&& op,
    AliasAnalysisKind alias_analysis) {
  return c10::optional<Operator>(Operator(
      std::string(schema_str), std::forward<Func>(op), alias_analysis));
}

template <typename Func>
c10::optional<Operator> OperatorGenerator(
    torch::detail::SelectiveStr<false> schema_str,
    Func&& op,
    AliasAnalysisKind alias_analysis) {
  return c10::nullopt;
}

} // namespace jit
} // namespace torch
