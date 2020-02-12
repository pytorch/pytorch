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
class BaseCallNode;
class FunctionCall;
class Allocate;
class Free;
class Cond;

class TORCH_API IRVisitor {
 public:
  TORCH_API virtual ~IRVisitor() {}
  TORCH_API virtual void visit(const Add* v);
  TORCH_API virtual void visit(const Sub* v);
  TORCH_API virtual void visit(const Mul* v);
  TORCH_API virtual void visit(const Div* v);
  TORCH_API virtual void visit(const Mod* v);
  TORCH_API virtual void visit(const Max* v);
  TORCH_API virtual void visit(const Min* v);
  TORCH_API virtual void visit(const CompareSelect* v);
  TORCH_API virtual void visit(const IntImm* v);
  TORCH_API virtual void visit(const FloatImm* v);
  TORCH_API virtual void visit(const Cast* v);
  TORCH_API virtual void visit(const Variable* v);
  TORCH_API virtual void visit(const Let* v);
  TORCH_API virtual void visit(const Ramp* v);
  TORCH_API virtual void visit(const Load* v);
  TORCH_API virtual void visit(const For* v);
  TORCH_API virtual void visit(const Block* v);
  TORCH_API virtual void visit(const Store* v);
  TORCH_API virtual void visit(const Broadcast* v);
  TORCH_API virtual void visit(const IfThenElse* v);

  // BaseCallNode is the base class for all call nodes.
  // For any visitors that only needs the common behavior, only override this
  // function is enough. This is because all derived class handlers will call
  // this function by default.
  // Override the derived class handler only if the logic is more specific to
  // that.
  TORCH_API virtual void visit(const Allocate* v);
  TORCH_API virtual void visit(const Free* v);
  TORCH_API virtual void visit(const Cond* v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
