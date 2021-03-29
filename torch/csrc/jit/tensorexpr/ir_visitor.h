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
class BaseCallNode;
class Intrinsics;
class FunctionCall;
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
  virtual ~IRVisitor() {}
  virtual void visit(const Add* v);
  virtual void visit(const Sub* v);
  virtual void visit(const Mul* v);
  virtual void visit(const Div* v);
  virtual void visit(const Mod* v);
  virtual void visit(const Max* v);
  virtual void visit(const Min* v);
  virtual void visit(const And* v);
  virtual void visit(const Or* v);
  virtual void visit(const Xor* v);
  virtual void visit(const Lshift* v);
  virtual void visit(const Rshift* v);
  virtual void visit(const CompareSelect* v);

#define IMM_PRINT_VISIT(Type, Name) virtual void visit(const Name##Imm* v);

  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_PRINT_VISIT)
#undef IMM_PRINT_VISIT

  virtual void visit(const Cast* v);
  virtual void visit(const BitCast* v);
  virtual void visit(const Var* v);
  virtual void visit(const Buf* v);
  virtual void visit(const Ramp* v);
  virtual void visit(const Load* v);
  virtual void visit(const For* v);
  virtual void visit(const Block* v);
  virtual void visit(const Store* v);
  virtual void visit(const Broadcast* v);
  virtual void visit(const IfThenElse* v);

  // BaseCallNode is the base class for all call nodes.
  // For any visitors that only needs the common behavior, only override this
  // function is enough. This is because all derived class handlers will call
  // this function by default.
  // Override the derived class handler only if the logic is more specific to
  // that.
  virtual void visit(const BaseCallNode* v);
  virtual void visit(const Intrinsics* v);
  virtual void visit(const FunctionCall* v);
  virtual void visit(const Allocate* v);
  virtual void visit(const Free* v);
  virtual void visit(const Let* v);
  virtual void visit(const Cond* v);
  virtual void visit(const Term* v);
  virtual void visit(const Polynomial* v);
  virtual void visit(const RoundOff* v);
  virtual void visit(const MaxTerm* v);
  virtual void visit(const MinTerm* v);
  virtual void visit(const ReduceOp* v);
  virtual void visit(const AtomicAdd* v);
  virtual void visit(const SyncThreads* v);
  virtual void visit(const ExternalCall* v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
