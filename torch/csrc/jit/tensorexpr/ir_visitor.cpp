#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <c10/util/irange.h>

namespace torch::jit::tensorexpr {

template <
    typename Op,
    std::enable_if_t<std::is_same_v<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>>* = nullptr>
static void visit_binary_op(const NodePtr<Op>& v, IRVisitor* visitor) {
  v->lhs()->accept(visitor);
  v->rhs()->accept(visitor);
}

void IRVisitor::visit(const AddPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const SubPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const MulPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const DivPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const ModPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const MaxPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const MinPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const AndPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const OrPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const XorPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const LshiftPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const RshiftPtr& v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(const CompareSelectPtr& v) {
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  v->ret_val1()->accept(this);
  v->ret_val2()->accept(this);
}

#define IMM_VISIT(Type, Name) \
  void IRVisitor::visit(const Name##ImmPtr& v) {}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_VISIT)
#undef IMM_VISIT

void IRVisitor::visit(const CastPtr& v) {
  v->src_value()->accept(this);
}
void IRVisitor::visit(const BitCastPtr& v) {
  v->src_value()->accept(this);
}
void IRVisitor::visit(const VarPtr& v) {}

void IRVisitor::visit(const RampPtr& v) {
  v->base()->accept(this);
  v->stride()->accept(this);
}

void IRVisitor::visit(const LoadPtr& v) {
  v->buf()->accept(this);
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
  }
}

void IRVisitor::visit(const BufPtr& v) {
  v->base_handle()->accept(this);
  if (v->qscale()) {
    v->qscale()->accept(this);
  }
  if (v->qzero()) {
    v->qzero()->accept(this);
  }
}

void IRVisitor::visit(const StorePtr& v) {
  v->buf()->accept(this);
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
  }
  v->value()->accept(this);
}

void IRVisitor::visit(const AtomicAddPtr& v) {
  v->buf()->accept(this);
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
  }
  v->value()->accept(this);
}

void IRVisitor::visit(const SyncThreadsPtr& v) {}

void IRVisitor::visit(const ExternalCallPtr& v) {
  v->buf()->accept(this);
  for (const BufPtr& buf_arg : v->buf_args()) {
    buf_arg->accept(this);
  }
  for (const ExprPtr& arg : v->args()) {
    arg->accept(this);
  }
}

void IRVisitor::visit(const ExternalCallWithAllocPtr& v) {
  for (const auto& buf_out_arg : v->buf_out_args()) {
    buf_out_arg->accept(this);
  }
  for (const auto& buf_arg : v->buf_args()) {
    buf_arg->accept(this);
  }
  for (const auto& arg : v->args()) {
    arg->accept(this);
  }
}

void IRVisitor::visit(const FreeExtPtr& v) {
  for (const auto& buf : v->bufs()) {
    buf->accept(this);
  }
}

void IRVisitor::visit(const BlockPtr& v) {
  for (const StmtPtr& s : *v) {
    s->accept(this);
  }
}

void IRVisitor::visit(const ForPtr& v) {
  v->var()->accept(this);
  v->start()->accept(this);
  v->stop()->accept(this);
  if (v->body()) {
    v->body()->accept(this);
  }
}

void IRVisitor::visit(const BroadcastPtr& v) {
  v->value()->accept(this);
}

void IRVisitor::visit(const IfThenElsePtr& v) {
  v->condition()->accept(this);
  v->true_value()->accept(this);
  v->false_value()->accept(this);
}

void IRVisitor::visit(const IntrinsicsPtr& v) {
  for (const auto i : c10::irange(v->nparams())) {
    v->param(i)->accept(this);
  }
}

void IRVisitor::visit(const AllocatePtr& v) {
  v->buffer_var()->accept(this);
  std::vector<ExprPtr> dims = v->dims();
  for (const ExprPtr& dim : dims) {
    dim->accept(this);
  }
}

void IRVisitor::visit(const FreePtr& v) {
  v->buffer_var()->accept(this);
}

void IRVisitor::visit(const PlacementAllocatePtr& v) {
  v->buf()->accept(this);
  v->buf_to_reuse()->accept(this);
}

void IRVisitor::visit(const LetPtr& v) {
  v->var()->accept(this);
  v->value()->accept(this);
}

void IRVisitor::visit(const CondPtr& v) {
  ExprPtr condition = v->condition();
  StmtPtr true_stmt = v->true_stmt();
  StmtPtr false_stmt = v->false_stmt();
  condition->accept(this);
  if (true_stmt) {
    true_stmt->accept(this);
  }
  if (false_stmt) {
    false_stmt->accept(this);
  }
}

void IRVisitor::visit(const TermPtr& v) {
  v->scalar()->accept(this);
  for (const auto& t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(const PolynomialPtr& v) {
  v->scalar()->accept(this);
  for (const auto& t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(const RoundOffPtr& v) {
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}

void IRVisitor::visit(const MaxTermPtr& v) {
  if (v->scalar()) {
    v->scalar()->accept(this);
  }
  for (const auto& t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(const MinTermPtr& v) {
  if (v->scalar()) {
    v->scalar()->accept(this);
  }
  for (const auto& t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(const ReduceOpPtr& v) {
  v->body()->accept(this);

  for (const auto& r : v->reduce_args()) {
    r->accept(this);
  }
}

} // namespace torch::jit::tensorexpr
