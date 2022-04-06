#pragma once

#include <iostream>

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
class ExprHandle;
class Mod;
class And;
class Or;
class Xor;
class Lshift;
class Rshift;
class CompareSelect;
class Ramp;
class Load;
class IfThenElse;
class Intrinsics;

class Stmt;
class ExternalCall;
class Store;
class For;
class Block;

class TORCH_API IRVerifier : public IRVisitor {
 public:
  IRVerifier() = default;

  void visit(ModPtr v) override;
  void visit(AndPtr v) override;
  void visit(OrPtr v) override;
  void visit(XorPtr v) override;
  void visit(LshiftPtr v) override;
  void visit(RshiftPtr v) override;
  void visit(CompareSelectPtr v) override;
  void visit(RampPtr v) override;
  void visit(LoadPtr v) override;
  void visit(IfThenElsePtr v) override;
  void visit(IntrinsicsPtr v) override;

  void visit(ExternalCallPtr v) override;
  void visit(StorePtr v) override;
  void visit(ForPtr v) override;
  void visit(BlockPtr v) override;
};

TORCH_API void verify(StmtPtr);
TORCH_API void verify(ExprPtr);
TORCH_API void verify(ExprHandle);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
