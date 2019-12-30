#include <torch/csrc/jit/fuser/common/ir_visitor.h>
// #include <torch/csrc/jit/fuser/common/ir.h>

namespace torch {
namespace jit {
namespace fuser {

IRVisitor::~IRVisitor() {}

// template <typename Op>
// static void visit_binary_op(const BinaryOpNode<Op>* v, IRVisitor* visitor) {
//   v->lhs().accept(visitor);
//   v->rhs().accept(visitor);
// }

// void IRVisitor::visit(const Add* v) { visit_binary_op(v, this); }

// void IRVisitor::visit(const Sub* v) { visit_binary_op(v, this); }

// void IRVisitor::visit(const Mul* v) { visit_binary_op(v, this); }

// void IRVisitor::visit(const Div* v) { visit_binary_op(v, this); }

// void IRVisitor::visit(const IntImm* v) {}
// void IRVisitor::visit(const FloatImm* v) {}
// void IRVisitor::visit(const Cast* v) { v->src_value().accept(this); }
// void IRVisitor::visit(const Variable* v) {}

// void IRVisitor::visit(const For* v) {
//   v->var().accept(this);
//   v->start().accept(this);
//   v->stop().accept(this);
//   v->body().accept(this);
// }

// void IRVisitor::visit(const Block* v) {
//   for (int i = 0; i < v->nexprs(); i++) {
//     v->expr(i).accept(this);
//   }
// }

// void IRVisitor::visit(const EmptyExpr* v) {}

} // namespace fuser
} // namespace jit
} // namespace torch
