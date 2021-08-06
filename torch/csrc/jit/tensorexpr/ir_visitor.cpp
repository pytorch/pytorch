#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Op>
static void visit_binary_op(BinaryOpNode<Op>* v, IRVisitor* visitor) {
  v->lhs()->accept(visitor);
  v->rhs()->accept(visitor);
}

void IRVisitor::visit(Add* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Sub* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Mul* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Div* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Mod* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Max* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Min* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(And* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Or* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Xor* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Lshift* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(Rshift* v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(CompareSelect* v) {
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  v->ret_val1()->accept(this);
  v->ret_val2()->accept(this);
}

// NOLINTNEXTLINE
#define IMM_VISIT(Type, Name) \
  void IRVisitor::visit(const Name##Imm* v) {}
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_VISIT);
#undef IMM_VISIT

void IRVisitor::visit(Cast* v) {
  v->src_value()->accept(this);
}
void IRVisitor::visit(BitCast* v) {
  v->src_value()->accept(this);
}
void IRVisitor::visit(Var* v) {}

void IRVisitor::visit(Ramp* v) {
  v->base()->accept(this);
  v->stride()->accept(this);
}

void IRVisitor::visit(Load* v) {
  v->buf()->accept(this);
  for (Expr* ind : v->indices()) {
    ind->accept(this);
  }
}

void IRVisitor::visit(Buf* v) {
  v->base_handle()->accept(this);
}

void IRVisitor::visit(Store* v) {
  v->buf()->accept(this);
  for (Expr* ind : v->indices()) {
    ind->accept(this);
  }
  v->value()->accept(this);
}

void IRVisitor::visit(AtomicAdd* v) {
  v->buf()->accept(this);
  for (Expr* ind : v->indices()) {
    ind->accept(this);
  }
  v->value()->accept(this);
}

void IRVisitor::visit(SyncThreads* v) {}

void IRVisitor::visit(ExternalCall* v) {
  v->buf()->accept(this);
  for (Buf* buf_arg : v->buf_args()) {
    buf_arg->accept(this);
  }
  for (Expr* arg : v->args()) {
    arg->accept(this);
  }
}

void IRVisitor::visit(Block* v) {
  for (Stmt* s : *v) {
    s->accept(this);
  }
}

void IRVisitor::visit(For* v) {
  v->var()->accept(this);
  v->start()->accept(this);
  v->stop()->accept(this);
  if (v->body()) {
    v->body()->accept(this);
  }
}

void IRVisitor::visit(Broadcast* v) {
  v->value()->accept(this);
}

void IRVisitor::visit(IfThenElse* v) {
  v->condition()->accept(this);
  v->true_value()->accept(this);
  v->false_value()->accept(this);
}

void IRVisitor::visit(Intrinsics* v) {
  for (auto i : c10::irange(v->nparams())) {
    v->param(i)->accept(this);
  }
}

void IRVisitor::visit(Allocate* v) {
  v->buffer_var()->accept(this);
  std::vector<Expr*> dims = v->dims();
  for (Expr* dim : dims) {
    dim->accept(this);
  }
}

void IRVisitor::visit(Free* v) {
  v->buffer_var()->accept(this);
}

void IRVisitor::visit(Let* v) {
  v->var()->accept(this);
  v->value()->accept(this);
}

void IRVisitor::visit(Cond* v) {
  Expr* condition = v->condition();
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

void IRVisitor::visit(Term* v) {
  v->scalar()->accept(this);
  for (auto* t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(Polynomial* v) {
  v->scalar()->accept(this);
  for (auto* t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(RoundOff* v) {
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}

void IRVisitor::visit(MaxTerm* v) {
  if (v->scalar()) {
    v->scalar()->accept(this);
  }
  for (auto* t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(MinTerm* v) {
  if (v->scalar()) {
    v->scalar()->accept(this);
  }
  for (auto* t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(ReduceOp* v) {
  v->body()->accept(this);

  for (auto* r : v->reduce_args()) {
    r->accept(this);
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
