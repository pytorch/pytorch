#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <c10/util/Optional.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API StatefulExpressionEvaluator : private OptOutDispatch {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit StatefulExpressionEvaluator(Fusion* fusion) : fusion_(fusion) {}

  Fusion* fusion() const {
    return fusion_;
  }

  void safeBind(
      Val* value,
      Int::ScalarType concrete_value,
      GpuLower* lower = nullptr);

  // Returns value if found in mapping, otherwise returns c10::nullopt
  c10::optional<Int::ScalarType> getValue(Val* value);

  // Checks if value is already infered, returns infered value if so, otherwise
  // runs traversal on value. Warning: should not be called in traversal.
  c10::optional<Int::ScalarType> inferValue(Val* value);

  // Debugging helper, prints all the currently set values
  void print() const;

 private:
  using OptOutDispatch::handle;

  void handle(Expr* expr) override {
    switch (expr->getExprType().value()) {
      case ExprType::UnaryOp:
        handle(expr->as<UnaryOp>());
        break;
      case ExprType::BinaryOp:
        handle(expr->as<BinaryOp>());
        break;
      case ExprType::KirUnaryOp:
        handle(expr->as<kir::UnaryOp>());
        break;
      case ExprType::KirBinaryOp:
        handle(expr->as<kir::BinaryOp>());
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false,
            "Cannot handle Expr type: ",
            expr->getExprType().value(),
            " in stateful expression evaluator.");
    }
  }

  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;

  // TODO(kir): remove this
  void handle(kir::UnaryOp*) override;
  void handle(kir::BinaryOp*) override;

  c10::optional<Int::ScalarType> maybeHandle(Val*);

 private:
  std::unordered_map<const Val*, Int::ScalarType> bindings_;
  Fusion* fusion_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
