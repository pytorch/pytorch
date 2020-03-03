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
 
  using IterVisitor::handle;
 
  // Scalar Related Vals
  virtual void handle(Float*) override;
  virtual void handle(Int*) override;

protected:
  std::ostream& irstream_;
  int cnt_;
};

class TORCH_API IRMathPrinter : public IRPrinter {
public:
  IRMathPrinter(std::ostream& os) :
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

  // Operator Expressions
  void handle(UnaryOp*) override;
  void handle(BinaryOp*) override;
};             

class TORCH_API IRTransformPrinter : public IRPrinter {
public:
  IRTransformPrinter(std::ostream& os) :
	IRPrinter(os) 
  { } 
  
  void print(const Fusion* const fusion);
 
  // High Level Handles 
  void handle(Statement* s) override;
  void handle(Expr* e) override;

  // Tensor Related Vals
  void handle(TensorDomain*) override;
  void handle(IterDomain*) override;

  // Tensor Expressions
  void handle(UnaryOp*) override {}
  void handle(BinaryOp*) override;
  void handle(Split*) override;
  void handle(Merge*) override {}
  void handle(Reorder*) override {}
};             

} // namespace fuser
} // namespace jit
} // namespace torch
