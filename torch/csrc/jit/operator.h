// in memory description of all ATen Ops similar to Caffe2 schema
// once C10 exists this can be removed, or stubbed out, but we need
// it now to implement correct semantic checking for script
#pragma once

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/function_schema.h"
#include "torch/csrc/jit/stack.h"

#include "ATen/ATen.h"

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch { namespace jit {

TORCH_API FunctionSchema parseSchema(const std::string& schema);

using OperationCreator = std::function<Operation(const Node*)>;

struct TORCH_API Operator {
  Operator(FunctionSchema schema, OperationCreator op_creator)
      : schema_(std::make_shared<FunctionSchema>(std::move(schema))),
        op_creator_(std::move(op_creator)) {}

  Operator(const std::string& schema, OperationCreator op_creator)
      : schema_string_(schema), op_creator_(std::move(op_creator)) {}

  // Helper constructor to register `op` to run
  // run for _every_ IR Node where n.kind() == name, regardless of arguments.
  // This is accomplished by marking the schema varargs and having no required
  // arguments. This is used for things like prim::While or prim::If that can
  // take a number of different valid input types and lengths.
  Operator(Symbol name, OperationCreator op_creator)
      : Operator(FunctionSchema(name, {}, {}, /*is_vararg*/true, /*is_varret*/true), std::move(op_creator)) {}

  Operator(FunctionSchema schema, Operation op)
      : schema_(std::make_shared<FunctionSchema>(std::move(schema))),
        op_(std::make_shared<Operation>(std::move(op))) {}

  Operator(const std::string& schema, Operation op)
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

  const FunctionSchema & schema() const {
    // we lazily parse schema initialized from strings so that
    // we do less work during static operator registration
    if(!schema_) {
      schema_ = std::make_shared<FunctionSchema>(parseSchema(schema_string_.value()));
      schema_string_ = c10::nullopt;
    }
    return *schema_;
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
};

TORCH_API const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol name);
std::shared_ptr<Operator> findOperatorFor(const Node* node);
const Operator& getOperatorFor(const Node* node);

inline Operation getOperation(const Node* node) {
  // note: getOperatorFor ensures that getOperatorFor(node).matches(node) == true
  // so the call to selectVariant is always valid.
  return getOperatorFor(node).getOperation(node);
}

TORCH_API void registerOperator(Operator&& op);

// XXX: this function is meant to be used with string literals only!
Operator& sig(const char *signature_literal);

struct OperatorSet {
  OperatorSet(std::initializer_list<const char *> sig_literals);
  // XXX: Returns a nullptr if no Operator in the set matches n
  Operator* find(const Node *n) const;
private:
  std::unordered_map<Symbol, std::vector<std::shared_ptr<Operator>>> ops;
};


}}
