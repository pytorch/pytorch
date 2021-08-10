#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <vector>

#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

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

class TORCH_API IRCloner : public IRMutator {
 public:
  ~IRCloner() override = default;
  Expr* mutate(Add* v) override;
  Expr* mutate(Sub* v) override;
  Expr* mutate(Mul* v) override;
  Expr* mutate(Div* v) override;
  Expr* mutate(Mod* v) override;
  Expr* mutate(Max* v) override;
  Expr* mutate(Min* v) override;
  Expr* mutate(And* v) override;
  Expr* mutate(Or* v) override;
  Expr* mutate(Xor* v) override;
  Expr* mutate(Lshift* v) override;
  Expr* mutate(Rshift* v) override;
  Expr* mutate(CompareSelect* v) override;
#define IMM_MUTATE_DECLARE(Type, Name) Expr* mutate(Name##Imm* v) override;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE
  Expr* mutate(Cast* v) override;
  Expr* mutate(BitCast* v) override;
  Expr* mutate(Var* v) override;
  Expr* mutate(Buf* v) override;
  Expr* mutate(Ramp* v) override;
  Expr* mutate(Load* v) override;
  Expr* mutate(Broadcast* v) override;
  Expr* mutate(IfThenElse* v) override;
  Expr* mutate(Intrinsics* v) override;

  Expr* mutate(Term* v) override;
  Expr* mutate(Polynomial* v) override;
  Expr* mutate(RoundOff* v) override;
  Expr* mutate(MaxTerm* v) override;
  Expr* mutate(MinTerm* v) override;

  Expr* mutate(ReduceOp* v) override;

  Stmt* mutate(For* v) override;
  Stmt* mutate(Block* v) override;
  Stmt* mutate(Store* v) override;
  Stmt* mutate(AtomicAdd* v) override;
  Stmt* mutate(SyncThreads* v) override;
  Stmt* mutate(ExternalCall* v) override;

  Stmt* mutate(Allocate* v) override;
  Stmt* mutate(Free* v) override;
  Stmt* mutate(Let* v) override;
  Stmt* mutate(Cond* v) override;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
