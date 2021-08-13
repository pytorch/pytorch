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
  if (lhs == lhs_new && rhs == rhs_new) {
    return v;
  }
  IRNodeType expr_type = v->expr_type();
  switch (expr_type) {
    case IRNodeType::kAdd:
      return new Add(lhs_new, rhs_new);
    case IRNodeType::kSub:
      return new Sub(lhs_new, rhs_new);
    case IRNodeType::kMul:
      return new Mul(lhs_new, rhs_new);
    case IRNodeType::kDiv:
      return new Div(lhs_new, rhs_new);
    case IRNodeType::kMod:
      return new Mod(lhs_new, rhs_new);
    case IRNodeType::kMax:
      return new Max(lhs_new, rhs_new, option);
    case IRNodeType::kMin:
      return new Min(lhs_new, rhs_new, option);
    case IRNodeType::kAnd:
      return new And(lhs_new, rhs_new);
    case IRNodeType::kOr:
      return new Or(lhs_new, rhs_new);
    case IRNodeType::kXor:
      return new Xor(lhs_new, rhs_new);
    case IRNodeType::kLshift:
      return new Lshift(lhs_new, rhs_new);
    case IRNodeType::kRshift:
      return new Rshift(lhs_new, rhs_new);
    default:
      throw unsupported_dtype();
  }
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
  Expr* retval1 = v->ret_val1();
  Expr* retval2 = v->ret_val2();
  Expr* lhs_new = lhs->accept_mutator(this);
  Expr* rhs_new = rhs->accept_mutator(this);
  Expr* retval1_new = retval1->accept_mutator(this);
  Expr* retval2_new = retval2->accept_mutator(this);
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
#define IMM_MUTATE_DEFINE(_1, Name)       \
  Expr* IRMutator::mutate(Name##Imm* v) { \
    return v;                             \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE

Expr* IRMutator::mutate(Cast* v) {
  Expr* src_value = v->src_value();
  Expr* src_value_new = src_value->accept_mutator(this);
  if (src_value_new == v->src_value()) {
    return v;
  }
  return new Cast(v->dtype(), src_value_new);
}

Expr* IRMutator::mutate(BitCast* v) {
  Expr* src_value = v->src_value();
  Expr* src_value_new = src_value->accept_mutator(this);
  if (src_value_new == v->src_value()) {
    return v;
  }
  return new BitCast(v->dtype(), src_value_new);
}

Expr* IRMutator::mutate(Var* v) {
  return v;
}

Expr* IRMutator::mutate(Ramp* v) {
  Expr* base = v->base();
  Expr* stride = v->stride();
  Expr* base_new = base->accept_mutator(this);
  Expr* stride_new = stride->accept_mutator(this);
  if (base == base_new && stride == stride_new) {
    return v;
  }
  return new Ramp(base_new, stride_new, v->lanes());
}

Expr* IRMutator::mutate(Load* v) {
  Dtype dtype = v->dtype();
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
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));
  if (buf == buf_new && !any_index_changed) {
    return v;
  }
  return new Load(dtype, buf_new, indices_new);
}

Expr* IRMutator::mutate(Buf* v) {
  Var* var = v->base_handle();
  Var* var_new =
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      dynamic_cast<Var*>(const_cast<Expr*>(var->accept_mutator(this)));
  if (!var_new) {
    return nullptr;
  }
  bool any_change = var_new != var;

  std::vector<Expr*> dims_old = v->dims();
  std::vector<Expr*> dims_new(dims_old.size());
  for (auto i : c10::irange(dims_old.size())) {
    dims_new[i] = dims_old[i]->accept_mutator(this);
    any_change |= (dims_new[i] != dims_old[i]);
  }

  if (!any_change) {
    return (Expr*)v;
  }

  v->set_base_handle(var_new);
  v->set_dims(dims_new);
  return v;
}

Expr* IRMutator::mutate(Broadcast* v) {
  Expr* value = v->value();
  int lanes = v->lanes();
  Expr* value_new = value->accept_mutator(this);
  if (value == value_new) {
    return v;
  }
  return new Broadcast(value_new, lanes);
}

Expr* IRMutator::mutate(IfThenElse* v) {
  Expr* condition = v->condition();
  Expr* true_value = v->true_value();
  Expr* false_value = v->false_value();
  Expr* condition_new = condition->accept_mutator(this);
  Expr* true_value_new = true_value->accept_mutator(this);
  Expr* false_value_new = false_value->accept_mutator(this);

  if (condition == condition_new && true_value == true_value_new &&
      false_value == false_value_new) {
    return v;
  }

  return new IfThenElse(condition_new, true_value_new, false_value_new);
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
  if (!any_change) {
    return v;
  }
  return new Intrinsics(v->op_type(), params);
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
  if (var == var_new && start == start_new && stop == stop_new &&
      body == body_new) {
    return (Stmt*)v;
  }
  if (body_new == body) {
    body_new = Stmt::clone(body);
  }
  return new For(var_new, start_new, stop_new, body_new, loop_options);
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
  if (!any_change) {
    return (Stmt*)v;
  }
  return Block::make(stmts);
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
  if (buf == buf_new && !any_index_changed && value == value_new) {
    return (Stmt*)v;
  }
  return new Store(buf_new, indices_new, value_new);
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
  if (buf == buf_new && !any_index_changed && value == value_new) {
    return (Stmt*)v;
  }
  return new AtomicAdd(buf_new, indices_new, value_new);
}

Stmt* IRMutator::mutate(SyncThreads* v) {
  return new SyncThreads();
}

Stmt* IRMutator::mutate(ExternalCall* v) {
  bool changed = false;
  Buf* new_buf = dynamic_cast<Buf*>(v->buf()->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(new_buf);
  changed |= new_buf != v->buf();

  std::vector<Buf*> new_buf_args;
  for (Buf* buf_arg : v->buf_args()) {
    Buf* new_buf_arg = dynamic_cast<Buf*>(buf_arg->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(new_buf_arg);
    new_buf_args.push_back(new_buf_arg);
    changed |= new_buf_arg != buf_arg;
  }
  std::vector<Expr*> new_args;
  for (Expr* arg : v->args()) {
    Expr* new_arg = arg->accept_mutator(this);
    new_args.push_back(new_arg);
    changed |= new_arg != arg;
  }
  return changed
      ? new ExternalCall(new_buf, v->func_name(), new_buf_args, new_args)
      : (Stmt*)v;
}

Stmt* IRMutator::mutate(Allocate* v) {
  Buf* buf = v->buf();
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  if (buf_new == buf) {
    return (Stmt*)v;
  }
  return new Allocate(buf_new);
}

Stmt* IRMutator::mutate(Free* v) {
  Buf* buf = v->buf();
  Buf* buf_new = dynamic_cast<Buf*>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  if (buf_new == buf) {
    return (Stmt*)v;
  }

  return new Free(buf_new);
}

Stmt* IRMutator::mutate(Let* v) {
  Var* var_old = v->var();
  Var* var_new = dynamic_cast<Var*>(var_old->accept_mutator(this));

  Expr* val_old = v->value();
  Expr* val_new = val_old->accept_mutator(this);

  if (var_new == var_old && val_old == val_new) {
    return (Stmt*)v;
  }

  return new Let(var_new, val_new);
}

Stmt* IRMutator::mutate(Cond* v) {
  Expr* cond_old = v->condition();
  Stmt* true_old = v->true_stmt();
  Stmt* false_old = v->false_stmt();

  Expr* cond_new = cond_old->accept_mutator(this);
  Stmt* true_new = true_old ? true_old->accept_mutator(this) : true_old;
  Stmt* false_new = false_old ? false_old->accept_mutator(this) : false_old;

  if (cond_old == cond_new && true_old == true_new && false_old == false_new) {
    return (Stmt*)v;
  }

  if (true_old && true_new == true_old) {
    true_new = Stmt::clone(true_old);
  }
  if (false_old && false_new == false_old) {
    false_new = Stmt::clone(false_old);
  }

  return new Cond(cond_new, true_new, false_new);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
