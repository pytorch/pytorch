#pragma once

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

/* Simplify the IR by combining arithmetic expressions over a common term.
 */
class TORCH_API IRSimplifier : public IRMutator {
 public:
  const Expr* mutate(const Add* v) override;
  const Expr* mutate(const Sub* v) override;
  const Expr* mutate(const Mul* v) override;
  const Expr* mutate(const Div* v) override;
  const Expr* mutate(const Mod* v) override;
  const Expr* mutate(const And* v) override;
  const Expr* mutate(const Xor* v) override;
  const Expr* mutate(const Lshift* v) override;
  const Expr* mutate(const Rshift* v) override;
  const Expr* mutate(const Max* v) override;
  const Expr* mutate(const Min* v) override;
  const Expr* mutate(const Intrinsics* v) override;
  const Expr* mutate(const Cast* v) override;

  static const Expr* simplify(const Expr* e);
  static ExprHandle simplify(const ExprHandle& e);
  static Stmt* simplify(Stmt* s);

 private:
  /* Expands lhs and rhs if they are LinearTerms, creating a new op to hold
   * them. If either side expands to a constant term, attempt simplification of
   * the new op. */
  const Expr* expandAndRecurse(
      IRNodeType expr_type,
      const Expr* lhs,
      const Expr* rhs);

  /* Handles optimization cases for Broadcast() + Other */
  const Expr* handleBroadcastAdd(const Broadcast* bc, const Expr* other);

  /* Handles optimization cases for Broadcast() * Other */
  const Expr* handleBroadcastMul(const Broadcast* bc, const Expr* other);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
