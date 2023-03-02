#pragma once

#include <c10/macros/Export.h>
#include <dynamic_type.h>
#include <ir_interface_nodes.h>
#include <iter_visitor.h>

#include <c10/util/Optional.h>

#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class FusionPrecomputedValues;

//! Calculate Fusion IR expressions
class TORCH_CUDA_CU_API ExpressionEvaluator : private OptOutDispatch {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit ExpressionEvaluator(Fusion* fusion) : fusion_(fusion) {}

  //! Returns the associated fusion object
  Fusion* fusion() const {
    return fusion_;
  }

  //! Bind a concrete value to an IR variable
  void bind(Val* value, const IntOrDouble& concrete_value);

  //! Bind a concrete value to a named scalar
  void bind(const std::string& name, const IntOrDouble& concrete_value);

  //! Try to evaluate a Fusion IR value
  c10::optional<IntOrDouble> evaluate(Val* value);

  //! Debugging helper, prints all the currently known values
  void print() const;

  void bindPrecomputedValues(FusionPrecomputedValues* precomputed_values) {
    evaluator_precomputed_values_ = precomputed_values;
  }

  auto precomputedValues() {
    return evaluator_precomputed_values_;
  }

 private:
  c10::optional<IntOrDouble> getValue(Val* value);

  void handle(UnaryOp*) final;
  void handle(BinaryOp*) final;
  // TODO: handle swizzle

 private:
  std::unordered_map<const Val*, IntOrDouble> known_values_;
  std::unordered_map<std::string, IntOrDouble> known_named_scalars_;
  Fusion* fusion_ = nullptr;
  FusionPrecomputedValues* evaluator_precomputed_values_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
