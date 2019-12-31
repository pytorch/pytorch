#pragma once

namespace torch {
namespace jit {
namespace compiler {

class Add;
class Sub;
class Mul;
class Div;
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
class Expr;
class Stmt;

class IRMutator {
 public:
  virtual Expr mutate(const Add* v);
  virtual Expr mutate(const Sub* v);
  virtual Expr mutate(const Mul* v);
  virtual Expr mutate(const Div* v);
  virtual Expr mutate(const IntImm* v);
  virtual Expr mutate(const FloatImm* v);
  virtual Expr mutate(const Cast* v);
  virtual Expr mutate(const Variable* v);
  virtual Expr mutate(const Let* v);
  virtual Expr mutate(const Ramp* v);
  virtual Expr mutate(const Load* v);
  virtual Expr mutate(const Broadcast* v);

  virtual Stmt mutate(const For* v);
  virtual Stmt mutate(const Block* v);
  virtual Stmt mutate(const Store* v);
};

} // namespace compiler
} // namespace jit
} // namespace torch
