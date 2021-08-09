#pragma once
#include <c10/core/ScalarType.h>
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
class Or;
class Xor;
class Lshift;
class Rshift;
class CompareSelect;

#define IMM_DECLARE(Type, Name) class Name##Imm;

AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_DECLARE)
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
class Intrinsics;
class Allocate;
class Free;
class Let;
class Cond;
class Term;
class Polynomial;
class RoundOff;
class MaxTerm;
class MinTerm;
class ReduceOp;
class AtomicAdd;
class SyncThreads;
class ExternalCall;

class TORCH_API IRVisitor {
 public:
  virtual ~IRVisitor() = default;
  virtual void visit(Add* v);
  virtual void visit(Sub* v);
  virtual void visit(Mul* v);
  virtual void visit(Div* v);
  virtual void visit(Mod* v);
  virtual void visit(Max* v);
  virtual void visit(Min* v);
  virtual void visit(And* v);
  virtual void visit(Or* v);
  virtual void visit(Xor* v);
  virtual void visit(Lshift* v);
  virtual void visit(Rshift* v);
  virtual void visit(CompareSelect* v);

#define IMM_PRINT_VISIT(Type, Name) virtual void visit(const Name##Imm* v);

  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_PRINT_VISIT)
#undef IMM_PRINT_VISIT

  virtual void visit(Cast* v);
  virtual void visit(BitCast* v);
  virtual void visit(Var* v);
  virtual void visit(Buf* v);
  virtual void visit(Ramp* v);
  virtual void visit(Load* v);
  virtual void visit(For* v);
  virtual void visit(Block* v);
  virtual void visit(Store* v);
  virtual void visit(Broadcast* v);
  virtual void visit(IfThenElse* v);
  virtual void visit(Intrinsics* v);
  virtual void visit(Allocate* v);
  virtual void visit(Free* v);
  virtual void visit(Let* v);
  virtual void visit(Cond* v);
  virtual void visit(Term* v);
  virtual void visit(Polynomial* v);
  virtual void visit(RoundOff* v);
  virtual void visit(MaxTerm* v);
  virtual void visit(MinTerm* v);
  virtual void visit(ReduceOp* v);
  virtual void visit(AtomicAdd* v);
  virtual void visit(SyncThreads* v);
  virtual void visit(ExternalCall* v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
