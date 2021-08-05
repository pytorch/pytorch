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
  ~IRCloner() = default;
  Expr* mutate(Add* v);
  Expr* mutate(Sub* v);
  Expr* mutate(Mul* v);
  Expr* mutate(Div* v);
  Expr* mutate(Mod* v);
  Expr* mutate(Max* v);
  Expr* mutate(Min* v);
  Expr* mutate(And* v);
  Expr* mutate(Or* v);
  Expr* mutate(Xor* v);
  Expr* mutate(Lshift* v);
  Expr* mutate(Rshift* v);
  Expr* mutate(CompareSelect* v);
#define IMM_MUTATE_DECLARE(Type, Name) Expr* mutate(Name##Imm* v);
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_MUTATE_DECLARE);
#undef IMM_MUTATE_DECLARE
  Expr* mutate(Cast* v);
  Expr* mutate(BitCast* v);
  Expr* mutate(Var* v);
  Expr* mutate(Buf* v);
  Expr* mutate(Ramp* v);
  Expr* mutate(Load* v);
  Expr* mutate(Broadcast* v);
  Expr* mutate(IfThenElse* v);
  Expr* mutate(Intrinsics* v);

  Expr* mutate(Term* v);
  Expr* mutate(Polynomial* v);
  Expr* mutate(RoundOff* v);
  Expr* mutate(MaxTerm* v);
  Expr* mutate(MinTerm* v);

  Expr* mutate(ReduceOp* v);

  Stmt* mutate(For* v);
  Stmt* mutate(Block* v);
  Stmt* mutate(Store* v);
  Stmt* mutate(AtomicAdd* v);
  Stmt* mutate(SyncThreads* v);
  Stmt* mutate(ExternalCall* v);

  Stmt* mutate(Allocate* v);
  Stmt* mutate(Free* v);
  Stmt* mutate(Let* v);
  Stmt* mutate(Cond* v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
