#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <vector>

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
class Or;
class Xor;
class Lshift;
class Rshift;
class CompareSelect;

#define IMM_DECLARE(Type, Name) class Name##Imm;
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_DECLARE);
#undef IMM_DECLARE

class Cast;
class BitCast;
class Var;
class Buf;
class Ramp;
class Load;
class For;
class Block;
class Store;
class Broadcast;
class IfThenElse;
class ExprHandle;
class Expr;
class Intrinsics;
class Allocate;
class Free;
class Let;
class Cond;
class Stmt;
class Term;
class Polynomial;
class RoundOff;
class MaxTerm;
class MinTerm;
class ReduceOp;
class AtomicAdd;
class SyncThreads;
class ExternalCall;

class TORCH_API IRMutator {
 public:
  virtual ~IRMutator() = default;
  virtual Expr* mutate(Add* v);
  virtual Expr* mutate(Sub* v);
  virtual Expr* mutate(Mul* v);
  virtual Expr* mutate(Div* v);
  virtual Expr* mutate(Mod* v);
  virtual Expr* mutate(Max* v);
  virtual Expr* mutate(Min* v);
  virtual Expr* mutate(And* v);
  virtual Expr* mutate(Or* v);
  virtual Expr* mutate(Xor* v);
  virtual Expr* mutate(Lshift* v);
  virtual Expr* mutate(Rshift* v);
  virtual Expr* mutate(CompareSelect* v);
#define IMM_MUTATE_DECLARE(Type, Name) virtual Expr* mutate(Name##Imm* v);
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE
  virtual Expr* mutate(Cast* v);
  virtual Expr* mutate(BitCast* v);
  virtual Expr* mutate(Var* v);
  virtual Expr* mutate(Buf* v);
  virtual Expr* mutate(Ramp* v);
  virtual Expr* mutate(Load* v);
  virtual Expr* mutate(Broadcast* v);
  virtual Expr* mutate(IfThenElse* v);
  virtual Expr* mutate(Intrinsics* v);

  virtual Expr* mutate(Term* v);
  virtual Expr* mutate(Polynomial* v);
  virtual Expr* mutate(RoundOff* v);
  virtual Expr* mutate(MaxTerm* v);
  virtual Expr* mutate(MinTerm* v);

  virtual Expr* mutate(ReduceOp* v);

  virtual Stmt* mutate(For* v);
  virtual Stmt* mutate(Block* v);
  virtual Stmt* mutate(Store* v);
  virtual Stmt* mutate(AtomicAdd* v);
  virtual Stmt* mutate(SyncThreads* v);
  virtual Stmt* mutate(ExternalCall* v);

  virtual Stmt* mutate(Allocate* v);
  virtual Stmt* mutate(Free* v);
  virtual Stmt* mutate(Let* v);
  virtual Stmt* mutate(Cond* v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
