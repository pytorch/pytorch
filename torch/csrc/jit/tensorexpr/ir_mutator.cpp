#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Op>
static const Expr* mutate_binary_op(
    const BinaryOpNode<Op>* v,
    IRMutator* mutator,
    bool option = false) {
  const Expr* lhs = v->lhs();
  const Expr* rhs = v->rhs();
  const Expr* lhs_new = lhs->accept_mutator(mutator);
  const Expr* rhs_new = rhs->accept_mutator(mutator);
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

const Expr* IRMutator::mutate(const Add* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Sub* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Mul* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Div* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Mod* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const And* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Or* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Xor* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Lshift* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Rshift* v) {
  return mutate_binary_op(v, this);
}

const Expr* IRMutator::mutate(const Max* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

const Expr* IRMutator::mutate(const Min* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

const Expr* IRMutator::mutate(const CompareSelect* v) {
  const Expr* lhs = v->lhs();
  const Expr* rhs = v->rhs();
  const Expr* retval1 = v->ret_val1();
  const Expr* retval2 = v->ret_val2();
  const Expr* lhs_new = lhs->accept_mutator(this);
  const Expr* rhs_new = rhs->accept_mutator(this);
  const Expr* retval1_new = retval1->accept_mutator(this);
  const Expr* retval2_new = retval2->accept_mutator(this);
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
#define IMM_MUTATE_DEFINE(_1, Name)                   \
  const Expr* IRMutator::mutate(const Name##Imm* v) { \
    return v;                                         \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE

const Expr* IRMutator::mutate(const Cast* v) {
  const Expr* src_value = v->src_value();
  const Expr* src_value_new = src_value->accept_mutator(this);
  if (src_value_new == v->src_value()) {
    return v;
  }
  return new Cast(v->dtype(), src_value_new);
}

const Expr* IRMutator::mutate(const BitCast* v) {
  const Expr* src_value = v->src_value();
  const Expr* src_value_new = src_value->accept_mutator(this);
  if (src_value_new == v->src_value()) {
    return v;
  }
  return new BitCast(v->dtype(), src_value_new);
}

const Expr* IRMutator::mutate(const Var* v) {
  return v;
}

const Expr* IRMutator::mutate(const Ramp* v) {
  const Expr* base = v->base();
  const Expr* stride = v->stride();
  const Expr* base_new = base->accept_mutator(this);
  const Expr* stride_new = stride->accept_mutator(this);
  if (base == base_new && stride == stride_new) {
    return v;
  }
  return new Ramp(base_new, stride_new, v->lanes());
}

const Expr* IRMutator::mutate(const Load* v) {
  Dtype dtype = v->dtype();
  const Buf* buf = v->buf();

  bool any_index_changed = false;
  std::vector<const Expr*> indices_new;
  for (const Expr* ind : v->indices()) {
    const Expr* new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  const Buf* buf_new = dynamic_cast<const Buf*>(buf->accept_mutator(this));
  if (buf == buf_new && !any_index_changed) {
    return v;
  }
  return new Load(dtype, buf_new, indices_new);
}

const Expr* IRMutator::mutate(Buf* v) {
  const Var* var = v->base_handle();
  Var* var_new =
      dynamic_cast<Var*>(const_cast<Expr*>(var->accept_mutator(this)));
  if (!var_new) {
    return nullptr;
  }
  bool any_change = var_new != var;

  std::vector<const Expr*> dims_old = v->dims();
  std::vector<const Expr*> dims_new(dims_old.size());
  for (size_t i = 0; i < dims_old.size(); i++) {
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

const Expr* IRMutator::mutate(const Broadcast* v) {
  const Expr* value = v->value();
  int lanes = v->lanes();
  const Expr* value_new = value->accept_mutator(this);
  if (value == value_new) {
    return v;
  }
  return new Broadcast(value_new, lanes);
}

const Expr* IRMutator::mutate(const IfThenElse* v) {
  const Expr* condition = v->condition();
  const Expr* true_value = v->true_value();
  const Expr* false_value = v->false_value();
  const Expr* condition_new = condition->accept_mutator(this);
  const Expr* true_value_new = true_value->accept_mutator(this);
  const Expr* false_value_new = false_value->accept_mutator(this);

  if (condition == condition_new && true_value == true_value_new &&
      false_value == false_value_new) {
    return v;
  }

  return new IfThenElse(condition_new, true_value_new, false_value_new);
}

const Expr* IRMutator::mutate(const Intrinsics* v) {
  std::vector<const Expr*> params(v->nparams());
  bool any_change = false;
  for (int i = 0; i < v->nparams(); i++) {
    const Expr* value = v->param(i);
    const Expr* value_new = value->accept_mutator(this);
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

const Expr* IRMutator::mutate(const Term* v) {
  const Expr* newScalar = v->scalar()->accept_mutator(this);

  std::vector<const Expr*> variables;
  for (const auto* t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return new Term(v->hasher(), newScalar, variables);
}

const Expr* IRMutator::mutate(const Polynomial* v) {
  const Expr* newScalar = v->scalar()->accept_mutator(this);

  std::vector<const Term*> variables;
  for (const auto* t : v->variables()) {
    variables.push_back(static_cast<const Term*>(t->accept_mutator(this)));
  }
  return new Polynomial(v->hasher(), newScalar, variables);
}

const Expr* IRMutator::mutate(const RoundOff* v) {
  return new RoundOff(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this));
}

const Expr* IRMutator::mutate(const MaxTerm* v) {
  const Expr* newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  std::vector<const Expr*> variables;
  for (const auto* t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return new MaxTerm(v->hasher(), newScalar, v->propagate_nans(), variables);
}

const Expr* IRMutator::mutate(const MinTerm* v) {
  const Expr* newScalar = nullptr;
  if (v->scalar()) {
    newScalar = v->scalar()->accept_mutator(this);
  }

  std::vector<const Expr*> variables;
  for (const auto* t : v->variables()) {
    variables.push_back(t->accept_mutator(this));
  }
  return new MinTerm(v->hasher(), newScalar, v->propagate_nans(), variables);
}

const Expr* IRMutator::mutate(const ReduceOp* v) {
  const Expr* body_new = v->body()->accept_mutator(this);

  std::vector<const Var*> new_reduce_args;
  for (auto* r : v->reduce_args()) {
    new_reduce_args.push_back(static_cast<const Var*>(r->accept_mutator(this)));
  }

  return new ReduceOp(body_new, new_reduce_args, v->reducer());
}

Stmt* IRMutator::mutate(const For* v) {
  const Expr* var = v->var();
  const Expr* start = v->start();
  const Expr* stop = v->stop();
  Stmt* body = v->body();
  LoopOptions loop_options = v->loop_options();
  const Expr* var_new_expr = var->accept_mutator(this);
  const Var* var_new = dynamic_cast<const Var*>(var_new_expr);
  const Expr* start_new = start->accept_mutator(this);
  const Expr* stop_new = stop->accept_mutator(this);
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

Stmt* IRMutator::mutate(const Block* v) {
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

Stmt* IRMutator::mutate(const Store* v) {
  const Buf* buf = v->buf();

  bool any_index_changed = false;
  std::vector<const Expr*> indices_new;
  for (const Expr* ind : v->indices()) {
    const Expr* new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  const Expr* value = v->value();
  const Buf* buf_new = dynamic_cast<const Buf*>(buf->accept_mutator(this));
  const Expr* value_new = value->accept_mutator(this);
  if (buf == buf_new && !any_index_changed && value == value_new) {
    return (Stmt*)v;
  }
  return new Store(buf_new, indices_new, value_new);
}

Stmt* IRMutator::mutate(const AtomicAdd* v) {
  const Buf* buf = v->buf();

  bool any_index_changed = false;
  std::vector<const Expr*> indices_new;
  for (const Expr* ind : v->indices()) {
    const Expr* new_ind = ind->accept_mutator(this);
    if (new_ind != ind) {
      any_index_changed = true;
    }
    indices_new.push_back(new_ind);
  }
  const Expr* value = v->value();
  const Buf* buf_new = dynamic_cast<const Buf*>(buf->accept_mutator(this));
  const Expr* value_new = value->accept_mutator(this);
  if (buf == buf_new && !any_index_changed && value == value_new) {
    return (Stmt*)v;
  }
  return new AtomicAdd(buf_new, indices_new, value_new);
}

Stmt* IRMutator::mutate(const SyncThreads* v) {
  return new SyncThreads();
}

Stmt* IRMutator::mutate(const ExternalCall* v) {
  bool changed = false;
  const Buf* new_buf = dynamic_cast<const Buf*>(v->buf()->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(new_buf);
  changed |= new_buf != v->buf();

  std::vector<const Buf*> new_buf_args;
  for (const Buf* buf_arg : v->buf_args()) {
    const Buf* new_buf_arg =
        dynamic_cast<const Buf*>(buf_arg->accept_mutator(this));
    TORCH_INTERNAL_ASSERT(new_buf_arg);
    new_buf_args.push_back(new_buf_arg);
    changed |= new_buf_arg != buf_arg;
  }
  std::vector<const Expr*> new_args;
  for (const Expr* arg : v->args()) {
    const Expr* new_arg = arg->accept_mutator(this);
    new_args.push_back(new_arg);
    changed |= new_arg != arg;
  }
  return changed
      ? new ExternalCall(new_buf, v->func_name(), new_buf_args, new_args)
      : (Stmt*)v;
}

Stmt* IRMutator::mutate(const Allocate* v) {
  const Buf* buf = v->buf();
  const Buf* buf_new = dynamic_cast<const Buf*>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  if (buf_new == buf) {
    return (Stmt*)v;
  }
  return new Allocate(buf_new);
}

Stmt* IRMutator::mutate(const Free* v) {
  const Buf* buf = v->buf();
  const Buf* buf_new = dynamic_cast<const Buf*>(buf->accept_mutator(this));
  TORCH_INTERNAL_ASSERT(buf_new);
  if (buf_new == buf) {
    return (Stmt*)v;
  }

  return new Free(buf_new);
}

Stmt* IRMutator::mutate(const Let* v) {
  const Var* var_old = v->var();
  const Var* var_new = dynamic_cast<const Var*>(var_old->accept_mutator(this));

  const Expr* val_old = v->value();
  const Expr* val_new = val_old->accept_mutator(this);

  if (var_new == var_old && val_old == val_new) {
    return (Stmt*)v;
  }

  return new Let(var_new, val_new);
}

Stmt* IRMutator::mutate(const Cond* v) {
  const Expr* cond_old = v->condition();
  Stmt* true_old = v->true_stmt();
  Stmt* false_old = v->false_stmt();

  const Expr* cond_new = cond_old->accept_mutator(this);
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

class StmtClone : public IRMutator {
 public:
  Stmt* mutate(const For* v) override;
  Stmt* mutate(const Block* v) override;
  Stmt* mutate(const Store* v) override;
  Stmt* mutate(const Allocate* v) override;
  Stmt* mutate(const Free* v) override;
  Stmt* mutate(const Let* v) override;
  Stmt* mutate(const Cond* v) override;
  Stmt* mutate(const AtomicAdd* v) override;
};

Stmt* StmtClone::mutate(const For* v) {
  // Only body needs to be cloned as only statements are mutable
  Stmt* body_new = v->body()->accept_mutator(this);

  return new For(v->var(), v->start(), v->stop(), body_new, v->loop_options());
}

Stmt* StmtClone::mutate(const Block* v) {
  std::vector<Stmt*> stmts;
  for (Stmt* stmt : *v) {
    stmts.push_back(stmt->accept_mutator(this));
  }
  return new Block(stmts);
}

Stmt* StmtClone::mutate(const Store* v) {
  return new Store(v->buf(), v->indices(), v->value());
}

Stmt* StmtClone::mutate(const AtomicAdd* v) {
  return new AtomicAdd(v->buf(), v->indices(), v->value());
}

Stmt* StmtClone::mutate(const Allocate* v) {
  return new Allocate(v->buf());
}

Stmt* StmtClone::mutate(const Free* v) {
  return new Free(v->buf());
}

Stmt* StmtClone::mutate(const Let* v) {
  return new Let(v->var(), v->value());
}

Stmt* StmtClone::mutate(const Cond* v) {
  Stmt* true_old = v->true_stmt();
  Stmt* false_old = v->false_stmt();

  Stmt* true_new = true_old ? true_old->accept_mutator(this) : true_old;
  Stmt* false_new = false_old ? false_old->accept_mutator(this) : false_old;

  return new Cond(v->condition(), true_new, false_new);
}

Stmt* Stmt::clone(Stmt* s) {
  StmtClone clone_mutator;
  Stmt* cloned = s->accept_mutator(&clone_mutator);
  set_parent(cloned, nullptr);
  return cloned;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
