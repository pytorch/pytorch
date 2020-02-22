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
class CompareSelect;
class IntImm;
class FloatImm;
class Cast;
class Variable;
class Let;
class Ramp;
class Load;
class For;
class Block;
class Store;
class Broadcast;
class IfThenElse;
class Expr;
class Stmt;
class BaseCallNode;
class FunctionCall;
class Allocate;
class Free;
class Cond;

class TORCH_API IRMutator {
 public:
  virtual ~IRMutator() {}
  virtual Expr mutate(const Add* v);
  virtual Expr mutate(const Sub* v);
  virtual Expr mutate(const Mul* v);
  virtual Expr mutate(const Div* v);
  virtual Expr mutate(const Mod* v);
  virtual Expr mutate(const Max* v);
  virtual Expr mutate(const Min* v);
  virtual Expr mutate(const CompareSelect* v);
  virtual Expr mutate(const IntImm* v);
  virtual Expr mutate(const FloatImm* v);
  virtual Expr mutate(const Cast* v);
  virtual Expr mutate(const Variable* v);
  virtual Expr mutate(const Let* v);
  virtual Expr mutate(const Ramp* v);
  virtual Expr mutate(const Load* v);
  virtual Expr mutate(const Broadcast* v);
  virtual Expr mutate(const IfThenElse* v);

  virtual Stmt mutate(const For* v);
  virtual Stmt mutate(const Block* v);
  virtual Stmt mutate(const Store* v);

  virtual Stmt mutate(const Allocate* v);
  virtual Stmt mutate(const Free* v);
  virtual Stmt mutate(const Cond* v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
