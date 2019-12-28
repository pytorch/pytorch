#pragma once

#include "ir.h"
#include "ir_visitor.h"

#include <ostream>

namespace torch {
namespace jit {
namespace fuser {

class IRPrinter : public IRVisitor {
  
 public:
  IRPrinter(std::ostream&);
  void print(Expr);
  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const IntImm* v) override;
  void visit(const FloatImm* v) override;
  void visit(const Cast* v) override;
  void visit(const Variable* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;
private:
  void indent();
  std::ostream& os;
  int indent_count;
  
};

std::ostream& operator<<(std::ostream& stream, const Expr&);

} // namespace fuser
} // namespace jit
} // namespace torch
