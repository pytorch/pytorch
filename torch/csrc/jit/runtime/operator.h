// in memory description of all ATen Ops similar to Caffe2 schema
// once C10 exists this can be removed, or stubbed out, but we need
// it now to implement correct semantic checking for script
#pragma once

#include <ATen/core/stack.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <ATen/core/stack.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/OperatorOptions.h>

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
using ::c10::Symbol;
using ::c10::FunctionSchema;

using OperationCreator = Operation (*)(const Node*);

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
private:
  struct C10Operator final {
    c10::OperatorHandle handle_;
    Operation op_;
  };
  struct JitOnlyOperator final {
    mutable c10::either<FunctionSchema, std::string> schema_;

    mutable c10::OperatorOptions options_;

    c10::either<Operation, OperationCreator> op_;
  };
public:

  Operator(c10::OperatorHandle opHandle, Operation operation)
      : op_(c10::make_left<C10Operator, JitOnlyOperator>(C10Operator {
        std::move(opHandle), std::move(operation)
      })) {}


  Operator(
      std::string schema,
      Operation op,
      c10::OperatorOptions options = c10::OperatorOptions())
      : op_(c10::make_right<C10Operator, JitOnlyOperator>(JitOnlyOperator {
          c10::make_right<FunctionSchema, std::string>(std::move(schema)),
          std::move(options),
          c10::make_left<Operation, OperationCreator>(std::move(op))
      })) {}


  Operator(
      std::string schema,
      OperationCreator op_creator,
      c10::OperatorOptions options = c10::OperatorOptions())
      : op_(c10::make_right<C10Operator, JitOnlyOperator>(JitOnlyOperator {
          c10::make_right<FunctionSchema, std::string>(std::move(schema)),
          std::move(options),
          c10::make_right<Operation, OperationCreator>(std::move(op_creator))
      })) {}

  // Helper constructor to register `op` to run
  // run for _every_ IR Node where n.kind() == name, regardless of arguments.
  // This is accomplished by marking the schema varargs and having no required
  // arguments.
  Operator(
      Symbol name,
      OperationCreator op_creator,
      c10::OperatorOptions options = c10::OperatorOptions())
      : op_(c10::make_right<C10Operator, JitOnlyOperator>(JitOnlyOperator{
          c10::make_left<FunctionSchema, std::string>(varArgSchemaWithName(name)),
          std::move(options),
          c10::make_right<Operation, OperationCreator>(std::move(op_creator))
      })) {}

  Operation getOperation(const Node* node = nullptr) const {
    return op_.fold<Operation>([] (const C10Operator& op) {
      return op.op_;
    }, [node] (const JitOnlyOperator& op) {
      return op.op_.fold<Operation>([] (const Operation& op) {
        return op;
      }, [node] (const OperationCreator& op_creator) {
        return op_creator(node);
      });
    });
  }

  const FunctionSchema& schema() const {
    return op_.fold<const FunctionSchema&>([] (const C10Operator& op) -> const FunctionSchema& {
      return op.handle_.schema();
    }, [] (const JitOnlyOperator& op) -> const FunctionSchema& {
      // we lazily parse schema initialized from strings so that
      // we do less work during static operator registration
      if (op.schema_.is_right()) {
        op.schema_ = c10::make_left<FunctionSchema, std::string>(parseSchema(op.schema_.right()));
      }
      return op.schema_.left();
    });
  }

  bool isC10Op() const {
    return op_.is_left();
  }

  c10::AliasAnalysisKind aliasAnalysisKind() const {
    c10::AliasAnalysisKind alias_analysis =
      op_.fold<c10::AliasAnalysisKind>([] (const C10Operator& op) {
        if (op.handle_.isValid()) {
          return op.handle_.options().aliasAnalysis();
        } else {
          // This op is already deregistered, we're likely currently shutting down PyTorch.
          // Just return an arbitrary value (CONSERVATIVE).
          // TODO We're doing an isValid check because the c10 operator might already be deregistered.
          //      Instead, we should automatically deregister the JIT wrapper when the c10 op
          //      gets deregistered and remove this isValid() check.
          return c10::AliasAnalysisKind::CONSERVATIVE;
        }
      }, [] (const JitOnlyOperator& op) {
        return op.options_.aliasAnalysis();
      });

    const FunctionSchema& schemaRef = schema();
    TORCH_CHECK(
        alias_analysis == AliasAnalysisKind::FROM_SCHEMA ||
            !schemaRef.hasAnyAliasInfo(),
        "In operator registration: Tried to register operator ",
        schemaRef,
        " with aliasing information in the schema but without AliasAnalysisKind::FROM_SCHEMA.");
    return alias_analysis;
  }
  bool hasOperation() const {
    return op_.fold<bool>([] (const C10Operator&) {
      return true;
    }, [] (const JitOnlyOperator& op) {
      return op.op_.is_left();
    });
  }
 private:
  static FunctionSchema varArgSchemaWithName(Symbol name) {
    return FunctionSchema(
        name,
        "",
        {},
        {},
        /*is_vararg*/ true,
        /*is_varret*/ true);
  }
  
  c10::either<C10Operator, JitOnlyOperator> op_;
};

TORCH_API std::string canonicalSchemaString(const FunctionSchema& schema);

TORCH_API const std::vector<std::shared_ptr<Operator>> getAllOperators();
TORCH_API const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(
    Symbol name);

// given a operator with an overload name, find the specific operator related to it,
// may return nullptr if no operator exists.
TORCH_API std::shared_ptr<Operator> findOperatorFor(const c10::OperatorName& full_name);

TORCH_API std::vector<Symbol> findSimilarOperators(Symbol input_op);

TORCH_API void registerOperator(Operator&& op);

// XXX: this function is meant to be used with string literals only!
std::shared_ptr<Operator> getOperatorForLiteral(const char* signature);

// Ensure the thing that registers c10 ops is defined.
// Otherwise, our registry will not have c10 ops. You can run into this
// scenario if you're querying registered ops during static init.
//
// This fn is defined in register_c10_ops.cpp
TORCH_API void ensure_c10_registerer_defined();

// Used to assert that unschematized operators have an analysis method written
TORCH_API bool aliasAnalysisHasSpecialCaseFor(c10::Symbol sym);

} // namespace jit
} // namespace torch
