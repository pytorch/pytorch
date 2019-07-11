// in memory description of all ATen Ops similar to Caffe2 schema
// once C10 exists this can be removed, or stubbed out, but we need
// it now to implement correct semantic checking for script
#pragma once

#include <ATen/core/stack.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/function_schema_parser.h>
#include <torch/csrc/jit/operator_options.h>
#include <ATen/core/stack.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/OperatorOptions.h>

#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

using ::c10::FunctionSchema;

using OperationCreator = std::function<Operation(const Node*)>;

/*
 * Note: JIT relies on Operator instances having static lifetime, because
 * it for example stores a non-owning FunctionSchema* pointer in the Node class,
 * which points to the function shema stored in the Operator instance.
 * Also, jit::Operator is meant to store more operator related information like
 * symbolic derivatives, which also requires them to have static lifetime
 * so that changes to symbolic derivatives are remembered.
 *
 * Now, currently, the c10 operator library doesn't store jit::Operator instances,
 * but we use a listener pattern that notifies JIT about changes in the
 * c10 operator library and then registers jit::Operator instances to the JIT
 * operator registry, acting as wrappers to the c10 operators.
 *
 * However, that results in code duplication as JIT and c10 will likely get
 * their own mechanisms for storing derivatives and other operator related
 * information, and all of this would have to be wrapped from c10 into JIT.
 *
 * We should consider merging the JIT and c10 registries, moving jit::Operator
 * to c10 and storing these jit::Operator instances in the c10 operator library
 * instead, allowing us to have these mechanisms only implemented once.
 * However, the current jit::Operator implementation has additional features
 * like OperationCreator that aren't needed in c10 (they're only used for
 * prim ops like If/Else or While which wouldn't be in the c10 operator library),
 * and which depend on other JIT features which we don't want to move to c10
 * (notably jit/ir.h). We might, however, be able, to split jit::Operator into
 * a c10::Operator with the core features and a jit::Operator that adds the
 * JIT-only features like OperationCreator, and then use c10::Operator in the
 * c10 operator library.
 */

struct TORCH_API Operator {
  Operator(c10::OperatorHandle opHandle, Operation operation)
      : schema_(std::make_shared<FunctionSchema>(opHandle.schema())),
        op_(std::make_shared<Operation>(std::move(operation))),
        c10Handle_(opHandle),
        options_(c10Handle_->options()) {}

  Operator(
      FunctionSchema schema,
      OperationCreator op_creator,
      c10::OperatorOptions options = c10::OperatorOptions())
      : schema_(std::make_shared<FunctionSchema>(std::move(schema))),
        op_creator_(std::move(op_creator)),
        options_(std::move(options)) {}

  Operator(
      const std::string& schema,
      OperationCreator op_creator,
      c10::OperatorOptions options = c10::OperatorOptions())
      : schema_string_(schema),
        op_creator_(std::move(op_creator)),
        options_(std::move(options)) {}

  // Helper constructor to register `op` to run
  // run for _every_ IR Node where n.kind() == name, regardless of arguments.
  // This is accomplished by marking the schema varargs and having no required
  // arguments. This is used for things like prim::While or prim::If that can
  // take a number of different valid input types and lengths.
  Operator(
      Symbol name,
      OperationCreator op_creator,
      c10::OperatorOptions options = c10::OperatorOptions())
      : Operator(
            FunctionSchema(
                name,
                "",
                {},
                {},
                /*is_vararg*/ true,
                /*is_varret*/ true),
            std::move(op_creator),
            std::move(options)) {}

  Operator(
      FunctionSchema schema,
      Operation op)
      : schema_(std::make_shared<FunctionSchema>(std::move(schema))),
        op_(std::make_shared<Operation>(std::move(op))) {}

  Operator(
      const std::string& schema,
      Operation op)
      : schema_string_(schema),
        op_(std::make_shared<Operation>(std::move(op))) {}

  bool matches(const Node* node) const;

  Operation getOperation(const Node* node = nullptr) const {
    if (op_) {
      return *op_;
    }
    AT_ASSERT(node != nullptr);
    return op_creator_(node);
  }

  const FunctionSchema& schema() const {
    // we lazily parse schema initialized from strings so that
    // we do less work during static operator registration
    if (!schema_) {
      schema_ =
          std::make_shared<FunctionSchema>(parseSchema(schema_string_.value()));
      schema_string_ = c10::nullopt;
    }
    return *schema_;
  }

  bool isC10Op() const {
    return c10Handle_.has_value();
  }

  c10::AliasAnalysisKind aliasAnalysisKind() const {
    return options_.aliasAnalysis();
  }

 private:
  mutable c10::optional<std::string> schema_string_;
  // cannot use c10::optional because windows has issues that require an
  // assignment operator to be generated cannot use std::unique_ptr because
  // initializer lists of Operators end up copying the Operator
  mutable std::shared_ptr<FunctionSchema> schema_;

  // Essentially a variant<Operation, OperationCreator>.
  // NB: std::function has a default state (where it == nullptr).
  std::shared_ptr<Operation> op_;
  OperationCreator op_creator_;
  c10::optional<c10::OperatorHandle> c10Handle_;
  c10::OperatorOptions options_;
};

TORCH_API std::string canonicalSchemaString(const FunctionSchema& schema);

TORCH_API const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(
    Symbol name);

std::shared_ptr<Operator> findOperatorFor(const Node* node);
const Operator& getOperatorFor(const Node* node);

inline Operation getOperation(const Node* node) {
  // note: getOperatorFor ensures that getOperatorFor(node).matches(node) ==
  // true so the call to selectVariant is always valid.
  return getOperatorFor(node).getOperation(node);
}


TORCH_API std::vector<Symbol> findSimilarOperators(Symbol input_op);

TORCH_API void registerOperator(Operator&& op);

// XXX: this function is meant to be used with string literals only!
Operator& sig(const char* signature_literal);

struct OperatorSet {
  OperatorSet(std::initializer_list<const char*> sig_literals);
  // XXX: Returns a nullptr if no Operator in the set matches n
  Operator* find(const Node* n) const;

 private:
  std::unordered_map<Symbol, std::vector<std::shared_ptr<Operator>>> ops;
};

} // namespace jit
} // namespace torch
