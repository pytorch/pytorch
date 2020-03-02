#pragma once

#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_API IRPrinter : public IterVisitor {
public:
  IRPrinter(std::ostream& os) : 
    IterVisitor(),
    irstream_(os),
    cnt_(0)   { }

protected:
  std::ostream& irstream_;
  int cnt_;
};

class TORCH_API IROperExprPrinter : public IRPrinter {
public:
  IROperExprPrinter(std::ostream& os) :
    IRPrinter(os)
  { } 
  
  void print(const Fusion* const fusion);
 
  // High Level Handles 
  void handle(Statement* s) override;
  void handle(Expr* e) override;

  // Tensor Related Vals
  void handle(Tensor*) override;
  void handle(TensorView*) override;
  void handle(TensorDomain*) override; 
  void handle(IterDomain*) override;

  // Scalar Related Vals
  void handle(Float*) override;
  void handle(Int*) override;

  // Operator Expressions
  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;
};             

class TORCH_API IRTensorExprPrinter : public IRPrinter {
public:
  IRTensorExprPrinter(std::ostream& os) :
	IRPrinter(os) 
  { } 
  
  void print(const Fusion* const fusion);
 
  // High Level Handles 
  virtual void handle(Statement* s);
  virtual void handle(Expr* e);
  virtual void handle(Val* v);

  // Tensor Related Vals
  virtual void handle(TensorDomain*) {}
  virtual void handle(TensorView*) {}
  virtual void handle(IterDomain*) {}
  virtual void handle(Tensor*) {}

  // Scalar Related Vals
  virtual void handle(Float*) {}
  virtual void handle(Int*) {}

  // Tensor Expressions
  virtual void handle(UnaryOp*) {}
  virtual void handle(BinaryOp*) {}
  virtual void handle(Split*) {}
  virtual void handle(Merge*) {}
  virtual void handle(Reorder*) {}
};             

} // namespace fuser
} // namespace jit
} // namespace torch
