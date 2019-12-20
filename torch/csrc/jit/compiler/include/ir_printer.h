#pragma once

#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/ir_visitor.h"

#include <ostream>

namespace torch {
namespace jit {
namespace compiler {

class IRPrinter : public IRVisitor {
 public:
  IRPrinter(std::ostream&);
  void print(Expr);
  void print(Stmt);
  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const IntImm* v) override;
  void visit(const FloatImm* v) override;
  void visit(const Cast* v) override;
  void visit(const Variable* v) override;
  void visit(const Let* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;
  void visit(const Store* v) override;
  void visit(const Broadcast* v) override;

 private:
  std::ostream& os;
};

std::ostream& operator<<(std::ostream& stream, const Expr&);
std::ostream& operator<<(std::ostream& stream, const Stmt&);

} // namespace compiler
} // namespace jit
} // namespace torch
