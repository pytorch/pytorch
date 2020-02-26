#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Add;
class Sub;
class Mul;
class Div;
class Mod;
class Max;
class Min;
class And;
class Xor;
class Lshift;
class Rshift;
class CompareSelect;
class IntImm;
class FloatImm;
class Cast;
class Var;
class Let;
class LetStmt;
class Ramp;
class Load;
class For;
class Block;
class Store;
class Broadcast;
class IfThenElse;
class ExprHandle;
class Expr;
class BaseCallNode;
class Intrinsics;
class FunctionCall;
class Allocate;
class Free;
class Cond;
class Stmt;

class TORCH_API IRMutator {
 public:
  virtual ~IRMutator() {}
  virtual const Expr* mutate(const Add* v);
  virtual const Expr* mutate(const Sub* v);
  virtual const Expr* mutate(const Mul* v);
  virtual const Expr* mutate(const Div* v);
  virtual const Expr* mutate(const Mod* v);
  virtual const Expr* mutate(const Max* v);
  virtual const Expr* mutate(const Min* v);
  virtual const Expr* mutate(const And* v);
  virtual const Expr* mutate(const Xor* v);
  virtual const Expr* mutate(const Lshift* v);
  virtual const Expr* mutate(const Rshift* v);
  virtual const Expr* mutate(const CompareSelect* v);
  virtual const Expr* mutate(const IntImm* v);
  virtual const Expr* mutate(const FloatImm* v);
  virtual const Expr* mutate(const Cast* v);
  virtual const Expr* mutate(const Var* v);
  virtual const Expr* mutate(const Let* v);
  virtual Stmt* mutate(const LetStmt* v);
  virtual const Expr* mutate(const Ramp* v);
  virtual const Expr* mutate(const Load* v);
  virtual const Expr* mutate(const Broadcast* v);
  virtual const Expr* mutate(const IfThenElse* v);
  // BaseCallNode is the base class for all call nodes.
  // For any visitors that only needs the common behavior, only override this
  // function is enough. This is because all derived class handlers will call
  // this function by default.
  // Override the derived class handler only if the logic is more specific to
  // that.
  virtual const Expr* mutate(const BaseCallNode* v);
  virtual const Expr* mutate(const Intrinsics* v);
  virtual const Expr* mutate(const FunctionCall* v);

  virtual Stmt* mutate(const For* v);
  virtual Stmt* mutate(const Block* v);
  virtual Stmt* mutate(const Store* v);

  virtual Stmt* mutate(const Allocate* v);
  virtual Stmt* mutate(const Free* v);
  virtual Stmt* mutate(const Cond* v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
