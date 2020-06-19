
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>

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
  explicit EvaluationContext(const Fusion* fusion) : fusion_(fusion) {}

  // Set the concrete value for a Int*
  void bind(const Val* value, Int::ScalarType concrete_value);

  // Retrieves the concrete value, or nullopt if not set
  c10::optional<Int::ScalarType> concreteValue(const Val* value) const;

  const Fusion* fusion() const {
    return fusion_;
  }

  // Debugging helper, prints all the currently set values
  void print() const;

 private:
  std::unordered_map<const Val*, Int::ScalarType> bindings_;
  const Fusion* fusion_ = nullptr;
};

// Evaluates expressions in a Fusion IR, using the passed in
// context (EvaluationContext) to query for concrete_values. The
// evaluation context may override concrete values in the IR as well.
class TORCH_CUDA_API ExpressionEvaluator : private OptInConstDispatch {
 public:
  // Returns the result of the specified expression, or nullopt if
  // the result cannot be evaluated
  static c10::optional<Int::ScalarType> evaluate(
      const Statement* expr,
      const EvaluationContext* context);

 private:
  explicit ExpressionEvaluator(const EvaluationContext* context)
      : context_(context) {}

  ~ExpressionEvaluator() override = default;

  void handle(const Int*) override;
  void handle(const NamedScalar*) override;

  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;

 private:
  const EvaluationContext* context_ = nullptr;
  c10::optional<Int::ScalarType> result_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
