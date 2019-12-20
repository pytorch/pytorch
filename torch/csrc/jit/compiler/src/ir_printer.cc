#include "torch/csrc/jit/compiler/include/ir_printer.h"

namespace torch {
namespace jit {
namespace compiler {

IRPrinter::IRPrinter(std::ostream& os) : os(os) {}

void IRPrinter::print(Expr expr) {
  expr.accept(this);
}

void IRPrinter::print(Stmt stmt) {
  stmt.accept(this);
}

#define BINARY_ACCEPT(os, v, op_str) \
  os << "(";                         \
  v->lhs().accept(this);             \
  os << " " << op_str << " ";        \
  v->rhs().accept(this);             \
  os << ")";

void IRPrinter::visit(const Add* v) {
  BINARY_ACCEPT(os, v, "+");
}

void IRPrinter::visit(const Sub* v) {
  BINARY_ACCEPT(os, v, "-");
}

void IRPrinter::visit(const Mul* v) {
  BINARY_ACCEPT(os, v, "*");
}

void IRPrinter::visit(const Div* v) {
  BINARY_ACCEPT(os, v, "/");
}

void IRPrinter::visit(const IntImm* v) {
  os << v->value();
}

void IRPrinter::visit(const FloatImm* v) {
  os << v->value();
}

void IRPrinter::visit(const Cast* v) {
  auto dtype = v->dtype();
  os << dtype << "(";
  v->src_value().accept(this);
  os << ")";
}

void IRPrinter::visit(const Variable* v) {
  os << v->name_hint();
}

void IRPrinter::visit(const Let* v) {
  os << "(let ";
  v->var().accept(this);
  os << " = ";
  v->value().accept(this);
  os << " in ";
  v->body().accept(this);
  os << ")";
}

void IRPrinter::visit(const Ramp* v) {
  throw std::runtime_error("NYI");
}

void IRPrinter::visit(const Load* v) {
  throw std::runtime_error("NYI");
}

void IRPrinter::visit(const For* v) {
  throw std::runtime_error("NYI");
}

void IRPrinter::visit(const Block* v) {
  throw std::runtime_error("NYI");
}

void IRPrinter::visit(const Store* v) {
  throw std::runtime_error("NYI");
}

void IRPrinter::visit(const Broadcast* v) {
  throw std::runtime_error("NYI");
}

std::ostream& operator<<(std::ostream& stream, const Expr& expr) {
  IRPrinter p(stream);
  p.print(expr);
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Stmt& stmt) {
  IRPrinter p(stream);
  p.print(stmt);
  return stream;
}

} // namespace compiler
} // namespace jit
} // namespace torch
