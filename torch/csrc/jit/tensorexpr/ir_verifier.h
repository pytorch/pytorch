#pragma once

#include <iostream>

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

  void visit(Mod* v) override;
  void visit(And* v) override;
  void visit(Or* v) override;
  void visit(Xor* v) override;
  void visit(Lshift* v) override;
  void visit(Rshift* v) override;
  void visit(CompareSelect* v) override;
  void visit(Ramp* v) override;
  void visit(Load* v) override;
  void visit(IfThenElse* v) override;
  void visit(Intrinsics* v) override;

  void visit(ExternalCall* v) override;
  void visit(Store* v) override;
  void visit(For* v) override;
  void visit(Block* v) override;
};

TORCH_API void verify(Stmt*);
TORCH_API void verify(Expr*);
TORCH_API void verify(ExprHandle);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
