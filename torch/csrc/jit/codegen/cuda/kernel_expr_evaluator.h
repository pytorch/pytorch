
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <c10/util/Optional.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

//! Calculate Kernel IR expressions
//!
//! How to evaluate Kernel IR expressions:
//!
//! ```cpp
//!   kir::ExpressionEvaluator eval;
//!   eval.bind(symbolic_value, concrete_value);
//!   ... bind more values ...
//!   const auto result = eval.evaluate(interesting_value);
//!   if (result.has_value()) {
//!     ... we have successfully calculated the result ...
//!   } else {
//!     ... expression can't be evaluated ...
//!   }
//! ```
//!
class TORCH_CUDA_CU_API ExpressionEvaluator : private IrVisitor {
 public:
  //! Set a concrete value for a symbolic value
  void bind(const Val* value, Int::ScalarType concrete_value);

  //! Try to evaluate a Kernel IR value
  c10::optional<Int::ScalarType> evaluate(const Val* value);

  //! Returns true if `value` is known before binding kernel inputs
  static bool isConst(const Val* value);

  //! Debugging helper, prints all the currently known values
  void print() const;

 private:
  void unhandled(const void*) final;
  void visit(const Int* value) final;
  void visit(const NamedScalar* named_scalar) final;
  void visit(const UnaryOp* unary_op) final;
  void visit(const BinaryOp* binary_op) final;

 private:
  std::unordered_map<const Val*, Int::ScalarType> known_values_;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
