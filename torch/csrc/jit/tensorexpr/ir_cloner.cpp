#include <torch/csrc/jit/tensorexpr/ir_cloner.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <
    typename Op,
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
static ExprPtr mutate_binary_op(
    NodePtr<Op> v,
    IRCloner* cloner,
    bool option = false) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(cloner);
  ExprPtr rhs_new = v->rhs()->accept_mutator(cloner);
  IRNodeType expr_type = v->expr_type();
  switch (expr_type) {
    case IRNodeType::kAdd:
      return alloc<Add>(lhs_new, rhs_new);
    case IRNodeType::kSub:
      return alloc<Sub>(lhs_new, rhs_new);
    case IRNodeType::kMul:
      return alloc<Mul>(lhs_new, rhs_new);
    case IRNodeType::kDiv:
      return alloc<Div>(lhs_new, rhs_new);
    case IRNodeType::kMod:
      return alloc<Mod>(lhs_new, rhs_new);
    case IRNodeType::kMax:
      return alloc<Max>(lhs_new, rhs_new, option);
    case IRNodeType::kMin:
      return alloc<Min>(lhs_new, rhs_new, option);
    case IRNodeType::kAnd:
      return alloc<And>(lhs_new, rhs_new);
    case IRNodeType::kOr:
      return alloc<Or>(lhs_new, rhs_new);
    case IRNodeType::kXor:
      return alloc<Xor>(lhs_new, rhs_new);
    case IRNodeType::kLshift:
      return alloc<Lshift>(lhs_new, rhs_new);
    case IRNodeType::kRshift:
      return alloc<Rshift>(lhs_new, rhs_new);
    default:
      throw unimplemented_lowering(v);
  }
}

ExprPtr IRCloner::mutate(AddPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(SubPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(MulPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(DivPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(ModPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(AndPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(OrPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(XorPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(LshiftPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(RshiftPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRCloner::mutate(MaxPtr v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

ExprPtr IRCloner::mutate(MinPtr v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

ExprPtr IRCloner::mutate(CompareSelectPtr v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);
  ExprPtr retval1_new = v->ret_val1()->accept_mutator(this);
  ExprPtr retval2_new = v->ret_val2()->accept_mutator(this);
  return alloc<CompareSelect>(
      lhs_new,
      rhs_new,
      retval1_new,
      retval2_new,
      v->compare_select_op(),
      v->bias());
}

// NOLINTNEXTLINE
#define IMM_MUTATE_DEFINE(_1, Name)          \
  ExprPtr IRCloner::mutate(Name##ImmPtr v) { \
    return v;                                \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE

ExprPtr IRCloner::mutate(CastPtr v) {
  ExprPtr src_value_new = v->src_value()->accept_mutator(this);
  return alloc<Cast>(v->dtype(), src_value_new);
}

ExprPtr IRCloner::mutate(BitCastPtr v) {
  ExprPtr src_value_new = v->src_value()->accept_mutator(this);
  return alloc<BitCast>(v->dtype(), src_value_new);
}

ExprPtr IRCloner::mutate(RampPtr v) {
  ExprPtr base_new = v->base()->accept_mutator(this);
  ExprPtr stride_new = v->stride()->accept_mutator(this);
  return alloc<Ramp>(base_new, stride_new, v->lanes());
}

ExprPtr IRCloner::mutate(LoadPtr v) {
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (ExprPtr ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return alloc<Load>(v->dtype(), buf_new, indices_new);
}

// We do not clone Vars since the original IR and cloned IR are expected to
// share the underlying variables.
ExprPtr IRCloner::mutate(VarPtr v) {
  return v;
}

// We do not clone Bufs since the original IR and cloned IR are expected to
// share the underlying Bufs. In spite of Bufs having expressions as dims and
// initializers, this is the expected usage of clone at this point.
//
// TODO: Revisit this if Bufs need to be cloned as well.
ExprPtr IRCloner::mutate(BufPtr v) {
  return v;
}

ExprPtr IRCloner::mutate(BroadcastPtr v) {
  int lanes = v->lanes();
  ExprPtr value_new = v->value()->accept_mutator(this);
  return alloc<Broadcast>(value_new, lanes);
}

ExprPtr IRCloner::mutate(IfThenElsePtr v) {
  ExprPtr condition_new = v->condition()->accept_mutator(this);
  ExprPtr true_value_new = v->true_value()->accept_mutator(this);
  ExprPtr false_value_new = v->false_value()->accept_mutator(this);

  return alloc<IfThenElse>(condition_new, true_value_new, false_value_new);
}

ExprPtr IRCloner::mutate(IntrinsicsPtr v) {
  std::vector<ExprPtr> params_new;
  params_new.reserve(v->nparams());
  for (auto param : v->params()) {
    params_new.push_back(param->accept_mutator(this));
  }
  return alloc<Intrinsics>(v->op_type(), v->dtype(), params_new);
}

ExprPtr IRCloner::mutate(TermPtr v) {
  ExprPtr scalar_new = v->scalar()->accept_mutator(this);

  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return alloc<Term>(v->hasher(), scalar_new, variables_new);
}

ExprPtr IRCloner::mutate(PolynomialPtr v) {
  ExprPtr scalar_new = v->scalar()->accept_mutator(this);

  std::vector<TermPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto t : v->variables()) {
    variables_new.push_back(static_to<Term>(t->accept_mutator(this)));
  }
  return alloc<Polynomial>(v->hasher(), scalar_new, variables_new);
}

ExprPtr IRCloner::mutate(RoundOffPtr v) {
  return alloc<RoundOff>(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this));
}

ExprPtr IRCloner::mutate(MaxTermPtr v) {
  ExprPtr scalar_new =
      v->scalar() ? v->scalar()->accept_mutator(this) : nullptr;

  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return alloc<MaxTerm>(
      v->hasher(), scalar_new, v->propagate_nans(), variables_new);
}

ExprPtr IRCloner::mutate(MinTermPtr v) {
  ExprPtr scalar_new =
      v->scalar() ? v->scalar()->accept_mutator(this) : nullptr;

  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return alloc<MinTerm>(
      v->hasher(), scalar_new, v->propagate_nans(), variables_new);
}

ExprPtr IRCloner::mutate(ReduceOpPtr v) {
  ExprPtr body_new = v->body()->accept_mutator(this);

  std::vector<VarPtr> reduce_args_new;
  reduce_args_new.reserve(v->reduce_args().size());
  for (auto r : v->reduce_args()) {
    reduce_args_new.push_back(static_to<Var>(r->accept_mutator(this)));
  }

  return alloc<ReduceOp>(body_new, reduce_args_new, v->reducer());
}

StmtPtr IRCloner::mutate(ForPtr v) {
  auto start_new = v->start()->accept_mutator(this);
  auto stop_new = v->stop()->accept_mutator(this);
  auto body_new = v->body()->accept_mutator(this);

  return alloc<For>(v->var(), start_new, stop_new, body_new, v->loop_options());
}

StmtPtr IRCloner::mutate(BlockPtr v) {
  std::vector<StmtPtr> stmts_new;
  stmts_new.reserve(v->nstmts());
  for (StmtPtr stmt : *v) {
    stmts_new.push_back(stmt->accept_mutator(this));
  }
  return alloc<Block>(stmts_new);
}

StmtPtr IRCloner::mutate(StorePtr v) {
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (auto ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  auto value_new = v->value()->accept_mutator(this);
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return alloc<Store>(buf_new, indices_new, value_new);
}

StmtPtr IRCloner::mutate(AtomicAddPtr v) {
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (auto ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  auto value_new = v->value()->accept_mutator(this);
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return alloc<AtomicAdd>(buf_new, indices_new, value_new);
}

StmtPtr IRCloner::mutate(AllocatePtr v) {
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return alloc<Allocate>(buf_new);
}

StmtPtr IRCloner::mutate(FreePtr v) {
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return alloc<Free>(buf_new);
}

StmtPtr IRCloner::mutate(SyncThreadsPtr v) {
  return alloc<SyncThreads>();
}

StmtPtr IRCloner::mutate(ExternalCallPtr v) {
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));

  std::vector<BufPtr> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (BufPtr buf_arg : v->buf_args()) {
    buf_args_new.push_back(to<Buf>(buf_arg->accept_mutator(this)));
  }
  std::vector<ExprPtr> args_new;
  args_new.reserve(v->args().size());
  for (ExprPtr arg : v->args()) {
    args_new.push_back(arg->accept_mutator(this));
  }

  return alloc<ExternalCall>(buf_new, v->func_name(), buf_args_new, args_new);
}

StmtPtr IRCloner::mutate(ExternalCall2Ptr v) {
  std::vector<BufPtr> buf_out_args_new;
  buf_out_args_new.reserve(v->buf_out_args().size());
  for (BufPtr buf_out_arg : v->buf_out_args()) {
    buf_out_args_new.push_back(to<Buf>(buf_out_arg->accept_mutator(this)));
  }

  std::vector<BufPtr> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (BufPtr buf_arg : v->buf_args()) {
    buf_args_new.push_back(to<Buf>(buf_arg->accept_mutator(this)));
  }
  std::vector<ExprPtr> args_new;
  args_new.reserve(v->args().size());
  for (ExprPtr arg : v->args()) {
    args_new.push_back(arg->accept_mutator(this));
  }

  return alloc<ExternalCall2>(
      v->func_name(), buf_out_args_new, buf_args_new, args_new);
}

StmtPtr IRCloner::mutate(LetPtr v) {
  auto value_new = v->value()->accept_mutator(this);
  return alloc<Let>(v->var(), value_new);
}

StmtPtr IRCloner::mutate(CondPtr v) {
  auto condition_new = v->condition()->accept_mutator(this);
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;
  return alloc<Cond>(condition_new, true_new, false_new);
}

StmtPtr Stmt::clone(StmtPtr s) {
  IRCloner cloner;
  StmtPtr cloned = s->accept_mutator(&cloner);
  set_parent(cloned, nullptr);
  return cloned;
}

ExprPtr Expr::clone(ExprPtr e) {
  IRCloner cloner;
  return e->accept_mutator(&cloner);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
