// in memory description of all ATen Ops similar to Caffe2 schema
// once C10 exists this can be removed, or stubbed out, but we need
// it now to implement correct semantic checking for script
#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/function_schema.h"
#include "torch/csrc/jit/stack.h"

namespace torch { namespace jit {

FunctionSchema parseSchema(const std::string& decl);

using OperationCreator = std::function<Operation(Node*)>;

struct Operator {
  Operator(FunctionSchema schema, OperationCreator op, OperationCreator op_const_attributes = nullptr)
    : schema(std::move(schema))
    , op(std::move(op))
    , op_const_attributes(std::move(op_const_attributes)) {}

  Operator(const std::string& schema, OperationCreator op, OperationCreator op_const_attributes = nullptr)
    : Operator(parseSchema(schema), std::move(op), std::move(op_const_attributes)) {}

  // Helper constructor to regsiter `op` to run
  // run for _every_ IR Node where n.kind() == name, regardless of arguments.
  // This is accomplished by marking the schema varargs and having no required arguments.
  // This is used for things like prim::While or prim::If that can take a number
  // of different valid input types and lengths.
  Operator(Symbol name, OperationCreator op)
  : Operator(FunctionSchema(name, {}, {}, true), op, op) {}

  FunctionSchema schema;

  bool matchesNode(Node* n) const;
  // Operators have different versions depending on if some inputs are encoded
  // as attributes or inputs. This function returns the right Operation function,
  // given a node encoded for one variant.
  // Behavior is undefined if matchesNode(n) == false
  Operation selectVariant(Node* n) const {
    if(n->hasAttributes()) {
      JIT_ASSERT(op_const_attributes != nullptr);
      return op_const_attributes(n);
    } else {
      return op(n);
    }
  }
private:
  OperationCreator op;
  OperationCreator op_const_attributes;
};

const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol name);
std::shared_ptr<Operator> findOperatorFor(Node* node);
const Operator& getOperatorFor(Node* node);

inline Operation getOperation(Node* node) {
  // note: getOperatorFor ensures that getOperatorFor(node).matchesNode(node) == true
  // so the call to selectVariant is always valid.
  return getOperatorFor(node).selectVariant(node);
}

void registerOperator(Operator&& op);

struct RegisterOperators {
  RegisterOperators(std::vector<Operator> operators) {
    for(Operator& o : operators) {
      registerOperator(std::move(o));
    }
  }
};

}}
