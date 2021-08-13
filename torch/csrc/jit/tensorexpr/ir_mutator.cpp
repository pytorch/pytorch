#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

#include <torch/csrc/jit/tensorexpr/eval.h>
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
    IRMutator* mutator,
    bool option = false) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  ExprPtr lhs_new = lhs->accept_mutator(mutator);
  ExprPtr rhs_new = rhs->accept_mutator(mutator);
  if (lhs == lhs_new && rhs == rhs_new) {
    return v;
  }
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
      throw unsupported_dtype();
  }
}

ExprPtr IRMutator::mutate(AddPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(SubPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(MulPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(DivPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(ModPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(AndPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(OrPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(XorPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(LshiftPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(RshiftPtr v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(MaxPtr v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

ExprPtr IRMutator::mutate(MinPtr v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

ExprPtr IRMutator::mutate(CompareSelectPtr v) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  ExprPtr retval1 = v->ret_val1();
  ExprPtr retval2 = v->ret_val2();
  ExprPtr lhs_new = lhs->accept_mutator(this);
  ExprPtr rhs_new = rhs->accept_mutator(this);
  ExprPtr retval1_new = retval1->accept_mutator(this);
  ExprPtr retval2_new = retval2->accept_mutator(this);
  if (lhs == lhs_new && rhs == rhs_new && retval1 == retval1_new &&
      retval2 == retval2_new) {
    return v;
  }
  return CompareSelect::make(
             ExprHandle(lhs_new),
             ExprHandle(rhs_new),
             ExprHandle(retval1_new),
             ExprHandle(retval2_new),
             v->compare_select_op(),
             v->bias())
      .node();
}

// NOLINTNEXTLINE
#define IMM_MUTATE_DEFINE(_1, Name)           \
  ExprPtr IRMutator::mutate(Name##ImmPtr v) { \
    return v;                                 \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE

ExprPtr IRMutator::mutate(CastPtr v) {
  ExprPtr src_value = v->src_value();
  ExprPtr src_value_new = src_value->accept_mutator(this);
  if (src_value_new == v->src_value()) {
    return v;
  }
  return alloc<Cast>(v->dtype(), src_value_new);
}

ExprPtr IRMutator::mutate(BitCastPtr v) {
  ExprPtr src_value = v->src_value();
  ExprPtr src_value_new = src_value->accept_mutator(this);
  if (src_value_new == v->src_value()) {
    return v;
  }
  return alloc<BitCast>(v->dtype(), src_value_new);
}

ExprPtr IRMutator::mutate(VarPtr v) {
  return v;
}

ExprPtr IRMutator::mutate(RampPtr v) {
  ExprPtr base = v->base();
  ExprPtr stride = v->stride();
  ExprPtr base_new = base->accept_mutator(this);
  ExprPtr stride_new = stride->accept_mutator(this);
  if (base == base_new && stride == stride_new) {
    return v;
  }
  return alloc<Ramp>(base_new, stride_new, v->lanes());
}

ExprPtr IRMutator::mutate(LoadPtr v) {
  Dtype dtype = v->dtype();
  BufPtr buf = v->buf();

  bool any_index_changed = false;
  std::vector<ExprPtr> indices_new;
  for (ExprPtr ind : v->indices()) {
    ExprPtr new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  if (buf == buf_new && !any_index_changed) {
    return v;
  }
  return alloc<Load>(dtype, buf_new, indices_new);
}

ExprPtr IRMutator::mutate(BufPtr v) {
  VarPtr var = v->base_handle();
  VarPtr var_new = to<Var>(var->accept_mutator(this));
  if (!var_new) {
    return nullptr;
  }
  bool any_change = var_new != var;

  std::vector<ExprPtr> dims_old = v->dims();
  std::vector<ExprPtr> dims_new(dims_old.size());
  for (const auto i : c10::irange(dims_old.size())) {
    dims_new[i] = dims_old[i]->accept_mutator(this);
    any_change |= (dims_new[i] != dims_old[i]);
  }

  if (!any_change) {
    return (ExprPtr)v;
  }

  v->set_base_handle(var_new);
  v->set_dims(dims_new);
  return v;
}

ExprPtr IRMutator::mutate(BroadcastPtr v) {
  ExprPtr value = v->value();
  int lanes = v->lanes();
  ExprPtr value_new = value->accept_mutator(this);
  if (value == value_new) {
    return v;
  }
  return alloc<Broadcast>(value_new, lanes);
}

ExprPtr IRMutator::mutate(IfThenElsePtr v) {
  ExprPtr condition = v->condition();
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();
  ExprPtr condition_new = condition->accept_mutator(this);
  ExprPtr true_value_new = true_value->accept_mutator(this);
  ExprPtr false_value_new = false_value->accept_mutator(this);

  if (condition == condition_new && true_value == true_value_new &&
      false_value == false_value_new) {
    return v;
  }

  return alloc<IfThenElse>(condition_new, true_value_new, false_value_new);
}

ExprPtr IRMutator::mutate(IntrinsicsPtr v) {
  std::vector<ExprPtr> params(v->nparams());
  bool any_change = false;
  for (int i = 0; i < v->nparams(); i++) {
    ExprPtr value = v->param(i);
    ExprPtr value_new = value->accept_mutator(this);
    if (value != value_new) {
      any_change = true;
    }
    params[i] = value_new;
  }
  if (!any_change) {
    return v;
  }
  return alloc<Intrinsics>(v->op_type(), params);
}

ExprPtr IRMutator::mutate(TermPtr v) {
  ExprPtr newScalar = v->scalar()->accept_mutator(this);

  std::vector<ExprPtr> variables;
  for (auto t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return alloc<Term>(v->hasher(), newScalar, variables);
}

ExprPtr IRMutator::mutate(PolynomialPtr v) {
  ExprPtr newScalar = v->scalar()->accept_mutator(this);

  std::vector<TermPtr> variables;
  for (auto t : v->variables()) {
    variables.push_back(static_to<Term>(t->accept_mutator(this)));
  }
  return alloc<Polynomial>(v->hasher(), newScalar, variables);
}

ExprPtr IRMutator::mutate(RoundOffPtr v) {
  return alloc<RoundOff>(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this));
}

ExprPtr IRMutator::mutate(MaxTermPtr v) {
  ExprPtr newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  std::vector<ExprPtr> variables;
  for (auto t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return alloc<MaxTerm>(v->hasher(), newScalar, v->propagate_nans(), variables);
}

ExprPtr IRMutator::mutate(MinTermPtr v) {
  ExprPtr newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  std::vector<ExprPtr> variables;
  for (auto t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return alloc<MinTerm>(v->hasher(), newScalar, v->propagate_nans(), variables);
}

ExprPtr IRMutator::mutate(ReduceOpPtr v) {
  ExprPtr body_new = v->body()->accept_mutator(this);

  std::vector<VarPtr> new_reduce_args;
  for (auto r : v->reduce_args()) {
    new_reduce_args.push_back(static_to<Var>(r->accept_mutator(this)));
  }

  return alloc<ReduceOp>(body_new, new_reduce_args, v->reducer());
}

StmtPtr IRMutator::mutate(ForPtr v) {
  ExprPtr var = v->var();
  ExprPtr start = v->start();
  ExprPtr stop = v->stop();
  StmtPtr body = v->body();
  LoopOptions loop_options = v->loop_options();
  ExprPtr var_new_expr = var->accept_mutator(this);
  VarPtr var_new = to<Var>(var_new_expr);
  ExprPtr start_new = start->accept_mutator(this);
  ExprPtr stop_new = stop->accept_mutator(this);
  StmtPtr body_new = body->accept_mutator(this);
  if (!body_new) {
    return nullptr;
  }
  if (var == var_new && start == start_new && stop == stop_new &&
      body == body_new) {
    return (StmtPtr)v;
  }
  if (body_new == body) {
    body_new = Stmt::clone(body);
  }
  return For::make(var_new, start_new, stop_new, body_new, loop_options);
}

StmtPtr IRMutator::mutate(BlockPtr v) {
  bool any_change = false;

  std::vector<StmtPtr> stmts;
  for (StmtPtr stmt : *v) {
    StmtPtr stmt_new = stmt->accept_mutator(this);
    if (stmt != stmt_new) {
      any_change = true;
    } else {
      stmt_new = Stmt::clone(stmt);
    }
    if (stmt_new) {
      stmts.push_back(stmt_new);
    }
  }
  if (!any_change) {
    return (StmtPtr)v;
  }
  return Block::make(stmts);
}

StmtPtr IRMutator::mutate(StorePtr v) {
  BufPtr buf = v->buf();

  bool any_index_changed = false;
  std::vector<ExprPtr> indices_new;
  for (ExprPtr ind : v->indices()) {
    ExprPtr new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  ExprPtr value = v->value();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  ExprPtr value_new = value->accept_mutator(this);
  if (buf == buf_new && !any_index_changed && value == value_new) {
    return (StmtPtr)v;
  }
  return alloc<Store>(buf_new, indices_new, value_new);
}

StmtPtr IRMutator::mutate(AtomicAddPtr v) {
  BufPtr buf = v->buf();

  bool any_index_changed = false;
  std::vector<ExprPtr> indices_new;
  for (ExprPtr ind : v->indices()) {
    ExprPtr new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  ExprPtr value = v->value();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  ExprPtr value_new = value->accept_mutator(this);
  if (buf == buf_new && !any_index_changed && value == value_new) {
    return (StmtPtr)v;
  }
  return alloc<AtomicAdd>(buf_new, indices_new, value_new);
}

StmtPtr IRMutator::mutate(SyncThreadsPtr v) {
  return alloc<SyncThreads>();
}

StmtPtr IRMutator::mutate(ExternalCallPtr v) {
  bool changed = false;
  BufPtr new_buf = to<Buf>(v->buf()->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(new_buf);
  changed |= new_buf != v->buf();

  std::vector<BufPtr> new_buf_args;
  for (BufPtr buf_arg : v->buf_args()) {
    BufPtr new_buf_arg = to<Buf>(buf_arg->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(new_buf_arg);
    new_buf_args.push_back(new_buf_arg);
    changed |= new_buf_arg != buf_arg;
  }
  std::vector<ExprPtr> new_args;
  for (ExprPtr arg : v->args()) {
    ExprPtr new_arg = arg->accept_mutator(this);
    new_args.push_back(new_arg);
    changed |= new_arg != arg;
  }
  return changed
      ? alloc<ExternalCall>(new_buf, v->func_name(), new_buf_args, new_args)
      : (StmtPtr)v;
}

StmtPtr IRMutator::mutate(AllocatePtr v) {
  BufPtr buf = v->buf();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  if (buf_new == buf) {
    return (StmtPtr)v;
  }
  return alloc<Allocate>(buf_new);
}

StmtPtr IRMutator::mutate(FreePtr v) {
  BufPtr buf = v->buf();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  if (buf_new == buf) {
    return (StmtPtr)v;
  }

  return alloc<Free>(buf_new);
}

StmtPtr IRMutator::mutate(LetPtr v) {
  VarPtr var_old = v->var();
  VarPtr var_new = to<Var>(var_old->accept_mutator(this));

  ExprPtr val_old = v->value();
  ExprPtr val_new = val_old->accept_mutator(this);

  if (var_new == var_old && val_old == val_new) {
    return (StmtPtr)v;
  }

  return alloc<Let>(var_new, val_new);
}

StmtPtr IRMutator::mutate(CondPtr v) {
  ExprPtr cond_old = v->condition();
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();

  ExprPtr cond_new = cond_old->accept_mutator(this);
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;

  if (cond_old == cond_new && true_old == true_new && false_old == false_new) {
    return (StmtPtr)v;
  }

  if (true_old && true_new == true_old) {
    true_new = Stmt::clone(true_old);
  }
  if (false_old && false_new == false_old) {
    false_new = Stmt::clone(false_old);
  }

  return Cond::make(cond_new, true_new, false_new);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
