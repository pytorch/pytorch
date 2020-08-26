
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <c10/util/Optional.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

// Encapsulates a set of value bindings on top of a Fusion IR
// (used to provide known values to ExpressionEvaluator)
//
// NOTE: currently it only supports Int values
//
class TORCH_CUDA_API EvaluationContext {
 public:
  explicit EvaluationContext(Fusion* fusion) : fusion_(fusion) {}

  // Set the concrete value for a Int*
  void bind(const Val* value, Int::ScalarType concrete_value);

  // Retrieves the concrete value, or nullopt if not set
  c10::optional<Int::ScalarType> concreteValue(const Val* value) const;

  Fusion* fusion() const {
    return fusion_;
  }

  // Debugging helper, prints all the currently set values
  void print() const;

 private:
  std::unordered_map<const Val*, Int::ScalarType> bindings_;
  Fusion* fusion_ = nullptr;
};

// Evaluates expressions in a Fusion IR, using the passed in
// context (EvaluationContext) to query for concrete_values. The
// evaluation context may override concrete values in the IR as well.
class TORCH_CUDA_API ExpressionEvaluator : private IterVisitor {
 public:
  // Returns the result of the specified expression, or nullopt if
  // the result cannot be evaluated
  static c10::optional<Int::ScalarType> evaluate(
      Val* val,
      const EvaluationContext* context);

 private:
  explicit ExpressionEvaluator(const EvaluationContext* context)
      : context_(context) {}

  ~ExpressionEvaluator() override = default;

  c10::optional<Int::ScalarType> value(const Statement* stmt) const;

  using IterVisitor::handle;

  void handle(NamedScalar*) override;
  void handle(Int*) override;
  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;

  // TODO(kir): remove this
  void handle(kir::NamedScalar*) override;
  void handle(kir::Int*) override;
  void handle(kir::UnaryOp*) override;
  void handle(kir::BinaryOp*) override;

 private:
  const EvaluationContext* context_ = nullptr;
  std::unordered_map<const Statement*, Int::ScalarType> values_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
