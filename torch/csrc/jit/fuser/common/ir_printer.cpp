#include "ir_printer.h"

namespace torch {
namespace jit {
namespace fuser {

IRPrinter::IRPrinter(std::ostream& os) : os(os), indent_count(0) {}

void IRPrinter::indent(){
  for (int i = 0; i < indent_count; i++)
      os << "  ";
}

void IRPrinter::print(Expr expr) {
  expr.accept(this);
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

void IRPrinter::visit(const For* v) {
  indent();
  os << "For (";
  v->var().accept(this);
  os << " in ";
  v->start().accept(this);
  os << " : ";
  v->stop().accept(this);
  os << "){\n";
  indent_count++;
  v->body().accept(this);
  indent_count--;
  indent();
  os << "}\n";
}

void IRPrinter::visit(const Block* v) {
  for(int i=0; i<v->nexprs(); i++)
    v->expr(i).accept(this);
}

std::ostream& operator<<(std::ostream& stream, const Expr& expr) {
  IRPrinter p(stream);
  p.print(expr);
  return stream;
}

} // namespace fuser
} // namespace jit
} // namespace torch
