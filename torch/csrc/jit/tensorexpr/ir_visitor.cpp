#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Op>
static void visit_binary_op(const BinaryOpNode<Op>* v, IRVisitor* visitor) {
  v->lhs()->accept(visitor);
  v->rhs()->accept(visitor);
}

void IRVisitor::visit(const Add* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Sub* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Mul* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Div* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Mod* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Max* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Min* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const And* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Xor* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Lshift* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const Rshift* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const CompareSelect* v) {
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  v->ret_val1()->accept(this);
  v->ret_val2()->accept(this);
}

void IRVisitor::visit(const IntImm* v) {}
void IRVisitor::visit(const FloatImm* v) {}
void IRVisitor::visit(const Cast* v) {
  v->src_value()->accept(this);
}
void IRVisitor::visit(const Var* v) {}
void IRVisitor::visit(const Let* v) {
  v->var()->accept(this);
  v->value()->accept(this);
  v->body()->accept(this);
}

void IRVisitor::visit(const LetStmt* v) {
  v->var()->accept(this);
  v->value()->accept(this);
  v->body()->accept(this);
}

void IRVisitor::visit(const Ramp* v) {
  v->base()->accept(this);
  v->stride()->accept(this);
}

void IRVisitor::visit(const Load* v) {
  v->base_handle()->accept(this);
  v->index()->accept(this);
  v->mask()->accept(this);
}

void IRVisitor::visit(const Store* v) {
  v->base_handle()->accept(this);
  v->index()->accept(this);
  v->value()->accept(this);
  v->mask()->accept(this);
}

void IRVisitor::visit(const Block* v) {
  for (int i = 0; i < v->nstmts(); i++) {
    v->stmt(i)->accept(this);
  }
}

void IRVisitor::visit(const For* v) {
  v->var()->accept(this);
  v->start()->accept(this);
  v->stop()->accept(this);
  if (v->body()) {
    v->body()->accept(this);
  }
}

void IRVisitor::visit(const Broadcast* v) {
  v->value()->accept(this);
}

void IRVisitor::visit(const IfThenElse* v) {
  v->condition()->accept(this);
  v->true_value()->accept(this);
  v->false_value()->accept(this);
}

void IRVisitor::visit(const BaseCallNode* v) {
  for (int i = 0; i < v->nparams(); i++) {
    v->param(i)->accept(this);
  }
}

void IRVisitor::visit(const Intrinsics* v) {
  const BaseCallNode* base = v;
  this->visit(base);
}

void IRVisitor::visit(const FunctionCall* v) {
  const BaseCallNode* base = v;
  this->visit(base);
}

void IRVisitor::visit(const Allocate* v) {
  const Var* buffer_var = v->buffer_var();
  buffer_var->accept(this);
  std::vector<const Expr*> dims = v->dims();
  for (const Expr* dim : dims) {
    dim->accept(this);
  }
}

void IRVisitor::visit(const Free* v) {
  const Var* buffer_var = v->buffer_var();
  buffer_var->accept(this);
}

void IRVisitor::visit(const Cond* v) {
  const Expr* condition = v->condition();
  Stmt* true_stmt = v->true_stmt();
  Stmt* false_stmt = v->false_stmt();
  condition->accept(this);
  if (true_stmt) {
    true_stmt->accept(this);
  }
  if (false_stmt) {
    false_stmt->accept(this);
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
