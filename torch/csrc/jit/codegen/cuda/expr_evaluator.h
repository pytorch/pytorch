#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/jit/codegen/cuda/dynamic_type.h>
#include <torch/csrc/jit/codegen/cuda/evaluator_common.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <c10/util/Optional.h>

#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class PrecomputedValues;

//! Calculate Fusion IR expressions
class TORCH_CUDA_CU_API ExpressionEvaluator : private OptInConstDispatch {
  void bind_(const Val* value, const EvaluatorValue& concrete_value);
  void bind_(const std::string& name, const EvaluatorValue& concrete_value);

 public:
  //! Bind a concrete value to an IR variable
  template <typename T>
  void bind(const Val* value, const T& concrete_value) {
    bind_(value, EvaluatorValue(concrete_value));
  }

  //! Bind a concrete value to a named scalar
  template <typename T>
  void bind(const std::string& name, const T& concrete_value) {
    bind_(name, EvaluatorValue(concrete_value));
  }

  //! Set a concrete value for a parallel dimension
  void bind(ParallelType pt, Int::ScalarType concrete_value);

  //! Try to evaluate a Fusion IR value
  c10::optional<EvaluatorValue> evaluate(const Val* value);

  //! Debugging helper, prints all the currently known values
  void print() const;

  void bindPrecomputedValues(PrecomputedValues* precomputed_values) {
    precomputed_values_ = precomputed_values;
  }

  auto& precomputedValues() {
    return precomputed_values_;
  }

 private:
  c10::optional<EvaluatorValue> getValue(const Val* value);

  void handle(const UnaryOp* unary_op) final;
  void handle(const BinaryOp* binary_op) final;
  void handle(const TernaryOp* binary_op) final;

 private:
  PrecomputedValues* precomputed_values_ = nullptr;
  std::unordered_map<const Val*, EvaluatorValue> known_values_;
  std::unordered_map<std::string, EvaluatorValue> known_named_scalars_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
