#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

#include <c10/util/irange.h>

namespace torch::jit::tensorexpr {

template <
    typename Op,
    std::enable_if_t<std::is_same_v<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>>* = nullptr>
static ExprPtr mutate_binary_op(
    NodePtr<Op> v,
    IRMutator* mutator,
    bool option = false) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  ExprPtr lhs_new = lhs->accept_mutator(mutator);
  ExprPtr rhs_new = rhs->accept_mutator(mutator);
  if (lhs != lhs_new) {
    v->set_lhs(lhs_new);
  }
  if (rhs != rhs_new) {
    v->set_rhs(rhs_new);
  }
  Dtype dtype_new =
      BinaryOpDtype(lhs_new->dtype(), rhs_new->dtype(), ScalarType::Undefined);
  if (dtype_new != v->dtype()) {
    v->set_dtype(dtype_new);
  }
  return v;
}

ExprPtr IRMutator::mutate(const AddPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const SubPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const MulPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const DivPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const ModPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const AndPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const OrPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const XorPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const LshiftPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const RshiftPtr& v) {
  return mutate_binary_op(v, this);
}

ExprPtr IRMutator::mutate(const MaxPtr& v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

ExprPtr IRMutator::mutate(const MinPtr& v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

ExprPtr IRMutator::mutate(const CompareSelectPtr& v) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  ExprPtr ret_val1 = v->ret_val1();
  ExprPtr ret_val2 = v->ret_val2();
  ExprPtr lhs_new = lhs->accept_mutator(this);
  ExprPtr rhs_new = rhs->accept_mutator(this);
  ExprPtr ret_val1_new = ret_val1->accept_mutator(this);
  ExprPtr ret_val2_new = ret_val2->accept_mutator(this);
  if (lhs != lhs_new) {
    v->set_lhs(lhs_new);
  }
  if (rhs != rhs_new) {
    v->set_rhs(rhs_new);
  }
  if (ret_val1 != ret_val1_new) {
    v->set_ret_val1(ret_val1_new);
  }
  if (ret_val2 != ret_val2_new) {
    v->set_ret_val2(ret_val2_new);
  }
  return v;
}

#define IMM_MUTATE_DEFINE(_1, Name)                  \
  ExprPtr IRMutator::mutate(const Name##ImmPtr& v) { \
    return v;                                        \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DEFINE)
#undef IMM_MUTATE_DEFINE

ExprPtr IRMutator::mutate(const CastPtr& v) {
  ExprPtr src_value = v->src_value();
  ExprPtr src_value_new = src_value->accept_mutator(this);
  if (src_value != src_value_new) {
    v->set_src_value(src_value_new);
  }
  return v;
}

ExprPtr IRMutator::mutate(const BitCastPtr& v) {
  ExprPtr src_value = v->src_value();
  ExprPtr src_value_new = src_value->accept_mutator(this);
  if (src_value != src_value_new) {
    v->set_src_value(src_value_new);
  }
  return v;
}

ExprPtr IRMutator::mutate(const VarPtr& v) {
  return v;
}

ExprPtr IRMutator::mutate(const RampPtr& v) {
  ExprPtr base = v->base();
  ExprPtr stride = v->stride();
  ExprPtr base_new = base->accept_mutator(this);
  ExprPtr stride_new = stride->accept_mutator(this);
  if (base != base_new) {
    v->set_base(base_new);
  }
  if (stride != stride_new) {
    v->set_stride(stride_new);
  }
  return v;
}

ExprPtr IRMutator::mutate(const LoadPtr& v) {
  BufPtr buf = v->buf();

  bool any_index_changed = false;
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (const ExprPtr& ind : v->indices()) {
    ExprPtr new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));

  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  if (any_index_changed) {
    v->set_indices(indices_new);
  }
  return v;
}

ExprPtr IRMutator::mutate(const BufPtr& v) {
  const VarPtr& var = v->base_handle();
  const VarPtr& var_new = to<Var>(var->accept_mutator(this));
  if (!var_new) {
    return nullptr;
  }

  bool dims_changed = false;
  std::vector<ExprPtr> dims_old = v->dims();
  std::vector<ExprPtr> dims_new(dims_old.size());
  for (const auto i : c10::irange(dims_old.size())) {
    dims_new[i] = dims_old[i]->accept_mutator(this);
    dims_changed |= (dims_new[i] != dims_old[i]);
  }

  if (var != var_new) {
    v->set_base_handle(var_new);
  }
  if (dims_changed) {
    v->set_dims(dims_new);
  }

  ExprPtr qscale = v->qscale();
  if (qscale) {
    ExprPtr qscale_new = qscale->accept_mutator(this);
    if (qscale != qscale_new) {
      v->set_qscale(qscale_new);
    }
  }

  ExprPtr qzero = v->qzero();
  if (qzero) {
    ExprPtr qzero_new = qzero->accept_mutator(this);
    if (qzero != qzero_new) {
      v->set_qzero(qzero_new);
    }
  }

  return v;
}

ExprPtr IRMutator::mutate(const BroadcastPtr& v) {
  const ExprPtr& value = v->value();
  const ExprPtr& value_new = value->accept_mutator(this);
  if (value != value_new) {
    v->set_value(value_new);
  }
  return v;
}

ExprPtr IRMutator::mutate(const IfThenElsePtr& v) {
  ExprPtr condition = v->condition();
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();
  ExprPtr condition_new = condition->accept_mutator(this);
  ExprPtr true_value_new = true_value->accept_mutator(this);
  ExprPtr false_value_new = false_value->accept_mutator(this);

  if (condition != condition_new) {
    v->set_condition(condition_new);
  }
  if (true_value != true_value_new) {
    v->set_true_value(true_value_new);
  }
  if (false_value != false_value_new) {
    v->set_false_value(false_value_new);
  }
  return v;
}

ExprPtr IRMutator::mutate(const IntrinsicsPtr& v) {
  std::vector<ExprPtr> params(v->nparams());
  bool any_change = false;
  for (size_t i = 0; i < v->nparams(); i++) {
    const ExprPtr& value = v->param(i);
    const ExprPtr& value_new = value->accept_mutator(this);
    if (value != value_new) {
      any_change = true;
    }
    params[i] = value_new;
  }
  if (any_change) {
    v->set_params(params);
  }
  return v;
}

ExprPtr IRMutator::mutate(const TermPtr& v) {
  ExprPtr newScalar = v->scalar()->accept_mutator(this);

  std::vector<ExprPtr> variables;
  for (const auto& t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return alloc<Term>(v->hasher(), newScalar, variables);
}

ExprPtr IRMutator::mutate(const PolynomialPtr& v) {
  ExprPtr newScalar = v->scalar()->accept_mutator(this);

  std::vector<TermPtr> variables;
  for (const auto& t : v->variables()) {
    variables.push_back(static_to<Term>(t->accept_mutator(this)));
  }
  return alloc<Polynomial>(v->hasher(), newScalar, variables);
}

ExprPtr IRMutator::mutate(const RoundOffPtr& v) {
  return alloc<RoundOff>(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this));
}

ExprPtr IRMutator::mutate(const MaxTermPtr& v) {
  ExprPtr newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  std::vector<ExprPtr> variables;
  for (const auto& t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return alloc<MaxTerm>(v->hasher(), newScalar, v->propagate_nans(), variables);
}

ExprPtr IRMutator::mutate(const MinTermPtr& v) {
  ExprPtr newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  std::vector<ExprPtr> variables;
  for (const auto& t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return alloc<MinTerm>(v->hasher(), newScalar, v->propagate_nans(), variables);
}

ExprPtr IRMutator::mutate(const ReduceOpPtr& v) {
  ExprPtr body_new = v->body()->accept_mutator(this);

  std::vector<VarPtr> new_reduce_args;
  for (const auto& r : v->reduce_args()) {
    new_reduce_args.push_back(static_to<Var>(r->accept_mutator(this)));
  }

  return alloc<ReduceOp>(body_new, new_reduce_args, v->reducer());
}

StmtPtr IRMutator::mutate(const ForPtr& v) {
  const ExprPtr& var = v->var();
  ExprPtr start = v->start();
  ExprPtr stop = v->stop();
  StmtPtr body = v->body();
  LoopOptions loop_options = v->loop_options();
  const ExprPtr& var_new_expr = var->accept_mutator(this);
  const VarPtr& var_new = to<Var>(var_new_expr);
  ExprPtr start_new = start->accept_mutator(this);
  ExprPtr stop_new = stop->accept_mutator(this);
  StmtPtr body_new = body->accept_mutator(this);
  if (!body_new) {
    return nullptr;
  }
  if (body != body_new) {
    v->set_body(body_new);
  }
  if (var != var_new) {
    v->set_var(var_new);
  }
  if (start != start_new) {
    v->set_start(start_new);
  }
  if (stop != stop_new) {
    v->set_stop(stop_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const BlockPtr& v) {
  bool any_change = false;

  std::vector<StmtPtr> stmts;
  for (const StmtPtr& stmt : *v) {
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
  if (any_change) {
    v->set_stmts(stmts);
  }
  return v;
}

StmtPtr IRMutator::mutate(const StorePtr& v) {
  BufPtr buf = v->buf();

  bool any_index_changed = false;
  std::vector<ExprPtr> indices_new;
  for (const ExprPtr& ind : v->indices()) {
    ExprPtr new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  const ExprPtr& value = v->value();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  const ExprPtr& value_new = value->accept_mutator(this);

  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  if (any_index_changed) {
    v->set_indices(indices_new);
  }
  if (value != value_new) {
    v->set_value(value_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const AtomicAddPtr& v) {
  BufPtr buf = v->buf();

  bool any_index_changed = false;
  std::vector<ExprPtr> indices_new;
  for (const ExprPtr& ind : v->indices()) {
    ExprPtr new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  const ExprPtr& value = v->value();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  const ExprPtr& value_new = value->accept_mutator(this);

  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  if (any_index_changed) {
    v->set_indices(indices_new);
  }
  if (value != value_new) {
    v->set_value(value_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const SyncThreadsPtr& v) {
  return alloc<SyncThreads>();
}

StmtPtr IRMutator::mutate(const ExternalCallPtr& v) {
  BufPtr buf = v->buf();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(
      buf_new, buildErrorMessage("IRMutator produced null for Buf."));

  bool buf_args_changed = false;
  std::vector<BufPtr> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (const BufPtr& buf_arg : v->buf_args()) {
    BufPtr buf_arg_new = to<Buf>(buf_arg->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(
        buf_arg_new, buildErrorMessage("IRMutator produced null for Buf."));
    buf_args_new.push_back(buf_arg_new);
    buf_args_changed |= buf_arg_new != buf_arg;
  }

  bool args_changed = false;
  std::vector<ExprPtr> args_new;
  args_new.reserve(v->args().size());
  for (const ExprPtr& arg : v->args()) {
    ExprPtr arg_new = arg->accept_mutator(this);
    args_new.push_back(arg_new);
    args_changed |= arg_new != arg;
  }

  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  if (buf_args_changed) {
    v->set_buf_args(buf_args_new);
  }
  if (args_changed) {
    v->set_args(args_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const ExternalCallWithAllocPtr& v) {
  bool buf_out_args_changed = false;
  std::vector<BufPtr> buf_out_args_new;
  buf_out_args_new.reserve(v->buf_out_args().size());
  for (const auto& buf_out_arg : v->buf_out_args()) {
    BufPtr buf_out_arg_new = to<Buf>(buf_out_arg->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(
        buf_out_arg_new, buildErrorMessage("IRMutator produced null for Buf."));
    buf_out_args_new.push_back(buf_out_arg_new);
    buf_out_args_changed |= buf_out_arg_new != buf_out_arg;
  }

  bool buf_args_changed = false;
  std::vector<BufPtr> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (const auto& buf_arg : v->buf_args()) {
    BufPtr buf_arg_new = to<Buf>(buf_arg->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(
        buf_arg_new, buildErrorMessage("IRMutator produced null for Buf."));
    buf_args_new.push_back(buf_arg_new);
    buf_args_changed |= buf_arg_new != buf_arg;
  }

  bool args_changed = false;
  std::vector<ExprPtr> args_new;
  args_new.reserve(v->args().size());
  for (const auto& arg : v->args()) {
    ExprPtr arg_new = arg->accept_mutator(this);
    args_new.push_back(arg_new);
    args_changed |= arg_new != arg;
  }

  if (buf_out_args_changed) {
    v->set_buf_out_args(buf_out_args_new);
  }
  if (buf_args_changed) {
    v->set_buf_args(buf_args_new);
  }
  if (args_changed) {
    v->set_args(args_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const AllocatePtr& v) {
  BufPtr buf = v->buf();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(
      buf_new, buildErrorMessage("IRMutator produced null for Buf."));
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const FreePtr& v) {
  BufPtr buf = v->buf();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(
      buf_new, buildErrorMessage("IRMutator produced null for Buf."));
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const FreeExtPtr& v) {
  bool bufs_changed = false;
  std::vector<BufPtr> bufs_new;
  bufs_new.reserve(v->bufs().size());
  for (const auto& buf : v->bufs()) {
    BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(
        buf_new, buildErrorMessage("IRMutator produced null for Buf."));
    bufs_new.push_back(buf_new);
    bufs_changed |= buf_new != buf;
  }

  if (bufs_changed) {
    v->set_bufs(bufs_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const PlacementAllocatePtr& v) {
  BufPtr buf = v->buf();
  BufPtr buf_new = to<Buf>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(
      buf_new, buildErrorMessage("IRMutator produced null for Buf."));
  v->set_buf(buf_new);

  BufPtr buf_to_reuse = v->buf_to_reuse();
  BufPtr buf_to_reuse_new = to<Buf>(buf_to_reuse->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(
      buf_to_reuse_new, buildErrorMessage("IRMutator produced null for Buf."));
  v->set_buf_to_reuse(buf_to_reuse_new);

  return v;
}

StmtPtr IRMutator::mutate(const LetPtr& v) {
  const VarPtr& var_old = v->var();
  const VarPtr& var_new = to<Var>(var_old->accept_mutator(this));

  const ExprPtr& val_old = v->value();
  const ExprPtr& val_new = val_old->accept_mutator(this);

  if (var_old != var_new) {
    v->set_var(var_new);
  }
  if (val_old != val_new) {
    v->set_val(val_new);
  }
  return v;
}

StmtPtr IRMutator::mutate(const CondPtr& v) {
  ExprPtr cond_old = v->condition();
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();

  ExprPtr cond_new = cond_old->accept_mutator(this);
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;

  if (cond_old != cond_new) {
    v->set_condition(cond_new);
  }

  if (true_old != true_new) {
    v->set_true_stmt(true_new);
  }

  if (false_old != false_new) {
    v->set_false_stmt(false_new);
  }

  return v;
}

} // namespace torch::jit::tensorexpr
