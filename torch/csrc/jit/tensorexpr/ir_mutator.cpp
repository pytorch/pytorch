#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Op>
static Expr* mutate_binary_op(
    BinaryOpNode<Op>* v,
    IRMutator* mutator,
    bool option = false) {
  Expr* lhs = v->lhs();
  Expr* rhs = v->rhs();
  Expr* lhs_new = lhs->accept_mutator(mutator);
  Expr* rhs_new = rhs->accept_mutator(mutator);
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

Expr* IRMutator::mutate(Add* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Sub* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Mul* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Div* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Mod* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(And* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Or* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Xor* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Lshift* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Rshift* v) {
  return mutate_binary_op(v, this);
}

Expr* IRMutator::mutate(Max* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

Expr* IRMutator::mutate(Min* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

Expr* IRMutator::mutate(CompareSelect* v) {
  Expr* lhs = v->lhs();
  Expr* rhs = v->rhs();
  Expr* ret_val1 = v->ret_val1();
  Expr* ret_val2 = v->ret_val2();
  Expr* lhs_new = lhs->accept_mutator(this);
  Expr* rhs_new = rhs->accept_mutator(this);
  Expr* ret_val1_new = ret_val1->accept_mutator(this);
  Expr* ret_val2_new = ret_val2->accept_mutator(this);
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

// NOLINTNEXTLINE
#define IMM_MUTATE_DEFINE(_1, Name)       \
  Expr* IRMutator::mutate(Name##Imm* v) { \
    return v;                             \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE

Expr* IRMutator::mutate(Cast* v) {
  Expr* src_value = v->src_value();
  Expr* src_value_new = src_value->accept_mutator(this);
  if (src_value != src_value_new) {
    v->set_src_value(src_value_new);
  }
  return v;
}

Expr* IRMutator::mutate(BitCast* v) {
  Expr* src_value = v->src_value();
  Expr* src_value_new = src_value->accept_mutator(this);
  if (src_value != src_value_new) {
    v->set_src_value(src_value_new);
  }
  return v;
}

Expr* IRMutator::mutate(Var* v) {
  return v;
}

Expr* IRMutator::mutate(Ramp* v) {
  Expr* base = v->base();
  Expr* stride = v->stride();
  Expr* base_new = base->accept_mutator(this);
  Expr* stride_new = stride->accept_mutator(this);
  if (base != base_new) {
    v->set_base(base_new);
  }
  if (stride != stride_new) {
    v->set_stride(stride_new);
  }
  return v;
}

Expr* IRMutator::mutate(Load* v) {
  Buf* buf = v->buf();

  bool any_index_changed = false;
  std::vector<Expr*> indices_new;
  indices_new.reserve(v->indices().size());
  for (Expr* ind : v->indices()) {
    Expr* new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));

  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  if (any_index_changed) {
    v->set_indices(indices_new);
  }
  return v;
}

Expr* IRMutator::mutate(Buf* v) {
  Var* var = v->base_handle();
  Var* var_new =
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      dynamic_cast<Var*>(const_cast<Expr*>(var->accept_mutator(this)));
  if (!var_new) {
    return nullptr;
  }

  bool dims_changed = false;
  std::vector<Expr*> dims_old = v->dims();
  std::vector<Expr*> dims_new(dims_old.size());
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

  return v;
}

Expr* IRMutator::mutate(Broadcast* v) {
  Expr* value = v->value();
  Expr* value_new = value->accept_mutator(this);
  if (value != value_new) {
    v->set_value(value_new);
  }
  return v;
}

Expr* IRMutator::mutate(IfThenElse* v) {
  Expr* condition = v->condition();
  Expr* true_value = v->true_value();
  Expr* false_value = v->false_value();
  Expr* condition_new = condition->accept_mutator(this);
  Expr* true_value_new = true_value->accept_mutator(this);
  Expr* false_value_new = false_value->accept_mutator(this);

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

Expr* IRMutator::mutate(Intrinsics* v) {
  std::vector<Expr*> params(v->nparams());
  bool any_change = false;
  for (int i = 0; i < v->nparams(); i++) {
    Expr* value = v->param(i);
    Expr* value_new = value->accept_mutator(this);
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

Expr* IRMutator::mutate(Term* v) {
  Expr* newScalar = v->scalar()->accept_mutator(this);

  std::vector<Expr*> variables;
  for (auto* t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return new Term(v->hasher(), newScalar, variables);
}

Expr* IRMutator::mutate(Polynomial* v) {
  Expr* newScalar = v->scalar()->accept_mutator(this);

  std::vector<Term*> variables;
  for (auto* t : v->variables()) {
    variables.push_back(static_cast<Term*>(t->accept_mutator(this)));
  }
  return new Polynomial(v->hasher(), newScalar, variables);
}

Expr* IRMutator::mutate(RoundOff* v) {
  return new RoundOff(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this));
}

Expr* IRMutator::mutate(MaxTerm* v) {
  Expr* newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  std::vector<Expr*> variables;
  for (auto* t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return new MaxTerm(v->hasher(), newScalar, v->propagate_nans(), variables);
}

Expr* IRMutator::mutate(MinTerm* v) {
  Expr* newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  std::vector<Expr*> variables;
  for (auto* t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return new MinTerm(v->hasher(), newScalar, v->propagate_nans(), variables);
}

Expr* IRMutator::mutate(ReduceOp* v) {
  Expr* body_new = v->body()->accept_mutator(this);

  std::vector<Var*> new_reduce_args;
  for (auto* r : v->reduce_args()) {
    new_reduce_args.push_back(static_cast<Var*>(r->accept_mutator(this)));
  }

  return new ReduceOp(body_new, new_reduce_args, v->reducer());
}

Stmt* IRMutator::mutate(For* v) {
  Expr* var = v->var();
  Expr* start = v->start();
  Expr* stop = v->stop();
  Stmt* body = v->body();
  LoopOptions loop_options = v->loop_options();
  Expr* var_new_expr = var->accept_mutator(this);
  Var* var_new = dynamic_cast<Var*>(var_new_expr);
  Expr* start_new = start->accept_mutator(this);
  Expr* stop_new = stop->accept_mutator(this);
  Stmt* body_new = body->accept_mutator(this);
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

Stmt* IRMutator::mutate(Block* v) {
  bool any_change = false;

  std::vector<Stmt*> stmts;
  for (Stmt* stmt : *v) {
    Stmt* stmt_new = stmt->accept_mutator(this);
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

Stmt* IRMutator::mutate(Store* v) {
  Buf* buf = v->buf();

  bool any_index_changed = false;
  std::vector<Expr*> indices_new;
  for (Expr* ind : v->indices()) {
    Expr* new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  Expr* value = v->value();
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));
  Expr* value_new = value->accept_mutator(this);

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

Stmt* IRMutator::mutate(AtomicAdd* v) {
  Buf* buf = v->buf();

  bool any_index_changed = false;
  std::vector<Expr*> indices_new;
  for (Expr* ind : v->indices()) {
    Expr* new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  Expr* value = v->value();
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));
  Expr* value_new = value->accept_mutator(this);

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

Stmt* IRMutator::mutate(SyncThreads* v) {
  return new SyncThreads();
}

Stmt* IRMutator::mutate(ExternalCall* v) {
  Buf* buf = v->buf();
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);

  bool buf_args_changed = false;
  std::vector<Buf*> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (Buf* buf_arg : v->buf_args()) {
    Buf* buf_arg_new = dynamic_cast<Buf*>(buf_arg->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(buf_arg_new);
    buf_args_new.push_back(buf_arg_new);
    buf_args_changed |= buf_arg_new != buf_arg;
  }

  bool args_changed = false;
  std::vector<Expr*> args_new;
  args_new.reserve(v->args().size());
  for (Expr* arg : v->args()) {
    Expr* arg_new = arg->accept_mutator(this);
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

Stmt* IRMutator::mutate(Allocate* v) {
  Buf* buf = v->buf();
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  return v;
}

Stmt* IRMutator::mutate(Free* v) {
  Buf* buf = v->buf();
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  if (buf != buf_new) {
    v->set_buf(buf_new);
  }
  return v;
}

Stmt* IRMutator::mutate(Let* v) {
  Var* var_old = v->var();
  Var* var_new = dynamic_cast<Var*>(var_old->accept_mutator(this));

  Expr* val_old = v->value();
  Expr* val_new = val_old->accept_mutator(this);

  if (var_old != var_new) {
    v->set_var(var_new);
  }
  if (val_old != val_new) {
    v->set_val(val_new);
  }
  return v;
}

Stmt* IRMutator::mutate(Cond* v) {
  Expr* cond_old = v->condition();
  Stmt* true_old = v->true_stmt();
  Stmt* false_old = v->false_stmt();

  Expr* cond_new = cond_old->accept_mutator(this);
  Stmt* true_new = true_old ? true_old->accept_mutator(this) : true_old;
  Stmt* false_new = false_old ? false_old->accept_mutator(this) : false_old;

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

} // namespace tensorexpr
} // namespace jit
} // namespace torch
