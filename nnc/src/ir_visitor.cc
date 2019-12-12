#include "ir.h"

namespace nnc {

template <typename Op>
static void visit_binary_op(const BinaryOpNode<Op>* v, IRVisitor* visitor) {
  v->lhs().accept(visitor);
  v->rhs().accept(visitor);
}

void IRVisitor::visit(const Add* v) { visit_binary_op(v, this); }

void IRVisitor::visit(const Sub* v) { visit_binary_op(v, this); }

void IRVisitor::visit(const Mul* v) { visit_binary_op(v, this); }

void IRVisitor::visit(const Div* v) { visit_binary_op(v, this); }

void IRVisitor::visit(const IntImm* v) {}
void IRVisitor::visit(const FloatImm* v) {}

}  // namespace nnc
