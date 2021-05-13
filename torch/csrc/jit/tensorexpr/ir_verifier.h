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

  void visit(const Mod* v) override;
  void visit(const And* v) override;
  void visit(const Or* v) override;
  void visit(const Xor* v) override;
  void visit(const Lshift* v) override;
  void visit(const Rshift* v) override;
  void visit(const CompareSelect* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const Intrinsics* v) override;

  void visit(const ExternalCall* v) override;
  void visit(const Store* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;
};

TORCH_API void verify(Stmt*);
TORCH_API void verify(const Expr*);
TORCH_API void verify(ExprHandle);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
