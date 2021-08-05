#include <torch/csrc/jit/tensorexpr/ir_cloner.h>

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
    IRCloner* cloner,
    bool option = false) {
  Expr* lhs_new = v->lhs()->accept_mutator(cloner);
  Expr* rhs_new = v->rhs()->accept_mutator(cloner);
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
      throw unimplemented_lowering(v);
  }
}

Expr* IRCloner::mutate(Add* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Sub* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Mul* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Div* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Mod* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(And* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Or* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Xor* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Lshift* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Rshift* v) {
  return mutate_binary_op(v, this);
}

Expr* IRCloner::mutate(Max* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

Expr* IRCloner::mutate(Min* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

Expr* IRCloner::mutate(CompareSelect* v) {
  Expr* lhs_new = v->lhs()->accept_mutator(this);
  Expr* rhs_new = v->rhs()->accept_mutator(this);
  Expr* retval1_new = v->ret_val1()->accept_mutator(this);
  Expr* retval2_new = v->ret_val2()->accept_mutator(this);
  return new CompareSelect(
      lhs_new,
      rhs_new,
      retval1_new,
      retval2_new,
      v->compare_select_op(),
      v->bias());
}

// NOLINTNEXTLINE
#define IMM_MUTATE_DEFINE(_1, Name)      \
  Expr* IRCloner::mutate(Name##Imm* v) { \
    return v;                            \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE

Expr* IRCloner::mutate(Cast* v) {
  Expr* src_value_new = v->src_value()->accept_mutator(this);
  return new Cast(v->dtype(), src_value_new);
}

Expr* IRCloner::mutate(BitCast* v) {
  Expr* src_value_new = v->src_value()->accept_mutator(this);
  return new BitCast(v->dtype(), src_value_new);
}

Expr* IRCloner::mutate(Ramp* v) {
  Expr* base_new = v->base()->accept_mutator(this);
  Expr* stride_new = v->stride()->accept_mutator(this);
  return new Ramp(base_new, stride_new, v->lanes());
}

Expr* IRCloner::mutate(Load* v) {
  std::vector<Expr*> indices_new;
  indices_new.reserve(v->indices().size());
  for (Expr* ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  Buf* buf_new = dynamic_cast<Buf*>(v->buf()->accept_mutator(this));
  return new Load(v->dtype(), buf_new, indices_new);
}

// We do not clone Vars since the original IR and cloned IR are expected to
// share the underlying variables.
Expr* IRCloner::mutate(Var* v) {
  return v;
}

// We do not clone Bufs since the original IR and cloned IR are expected to
// share the underlying Bufs. In spite of Bufs having expressions as dims and
// initializers, this is the expected usage of clone at this point.
//
// TODO: Revisit this if Bufs need to be cloned as well.
Expr* IRCloner::mutate(Buf* v) {
  return v;
}

Expr* IRCloner::mutate(Broadcast* v) {
  int lanes = v->lanes();
  Expr* value_new = v->value()->accept_mutator(this);
  return new Broadcast(value_new, lanes);
}

Expr* IRCloner::mutate(IfThenElse* v) {
  Expr* condition_new = v->condition()->accept_mutator(this);
  Expr* true_value_new = v->true_value()->accept_mutator(this);
  Expr* false_value_new = v->false_value()->accept_mutator(this);

  return new IfThenElse(condition_new, true_value_new, false_value_new);
}

Expr* IRCloner::mutate(Intrinsics* v) {
  std::vector<Expr*> params_new;
  params_new.reserve(v->nparams());
  for (auto param : v->params()) {
    params_new.push_back(param->accept_mutator(this));
  }
  return new Intrinsics(v->op_type(), v->dtype(), params_new);
}

Expr* IRCloner::mutate(Term* v) {
  Expr* scalar_new = v->scalar()->accept_mutator(this);

  std::vector<Expr*> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto* t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return new Term(v->hasher(), scalar_new, variables_new);
}

Expr* IRCloner::mutate(Polynomial* v) {
  Expr* scalar_new = v->scalar()->accept_mutator(this);

  std::vector<Term*> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto* t : v->variables()) {
    variables_new.push_back(static_cast<Term*>(t->accept_mutator(this)));
  }
  return new Polynomial(v->hasher(), scalar_new, variables_new);
}

Expr* IRCloner::mutate(RoundOff* v) {
  return new RoundOff(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this));
}

Expr* IRCloner::mutate(MaxTerm* v) {
  Expr* scalar_new = v->scalar() ? v->scalar()->accept_mutator(this) : nullptr;

  std::vector<Expr*> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto* t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return new MaxTerm(
      v->hasher(), scalar_new, v->propagate_nans(), variables_new);
}

Expr* IRCloner::mutate(MinTerm* v) {
  Expr* scalar_new = v->scalar() ? v->scalar()->accept_mutator(this) : nullptr;

  std::vector<Expr*> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto* t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return new MinTerm(
      v->hasher(), scalar_new, v->propagate_nans(), variables_new);
}

Expr* IRCloner::mutate(ReduceOp* v) {
  Expr* body_new = v->body()->accept_mutator(this);

  std::vector<Var*> reduce_args_new;
  reduce_args_new.reserve(v->reduce_args().size());
  for (auto* r : v->reduce_args()) {
    reduce_args_new.push_back(static_cast<Var*>(r->accept_mutator(this)));
  }

  return new ReduceOp(body_new, reduce_args_new, v->reducer());
}

Stmt* IRCloner::mutate(For* v) {
  auto start_new = v->start()->accept_mutator(this);
  auto stop_new = v->stop()->accept_mutator(this);
  auto body_new = v->body()->accept_mutator(this);

  return new For(v->var(), start_new, stop_new, body_new, v->loop_options());
}

Stmt* IRCloner::mutate(Block* v) {
  std::vector<Stmt*> stmts_new;
  stmts_new.reserve(v->nstmts());
  for (Stmt* stmt : *v) {
    stmts_new.push_back(stmt->accept_mutator(this));
  }
  return new Block(stmts_new);
}

Stmt* IRCloner::mutate(Store* v) {
  std::vector<Expr*> indices_new;
  indices_new.reserve(v->indices().size());
  for (auto ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  auto value_new = v->value()->accept_mutator(this);
  Buf* buf_new = dynamic_cast<Buf*>(v->buf()->accept_mutator(this));
  return new Store(buf_new, indices_new, value_new);
}

Stmt* IRCloner::mutate(AtomicAdd* v) {
  std::vector<Expr*> indices_new;
  indices_new.reserve(v->indices().size());
  for (auto ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  auto value_new = v->value()->accept_mutator(this);
  Buf* buf_new = dynamic_cast<Buf*>(v->buf()->accept_mutator(this));
  return new AtomicAdd(buf_new, indices_new, value_new);
}

Stmt* IRCloner::mutate(Allocate* v) {
  Buf* buf_new = dynamic_cast<Buf*>(v->buf()->accept_mutator(this));
  return new Allocate(buf_new);
}

Stmt* IRCloner::mutate(Free* v) {
  Buf* buf_new = dynamic_cast<Buf*>(v->buf()->accept_mutator(this));
  return new Free(buf_new);
}

Stmt* IRCloner::mutate(SyncThreads* v) {
  return new SyncThreads();
}

Stmt* IRCloner::mutate(ExternalCall* v) {
  Buf* buf_new = dynamic_cast<Buf*>(v->buf()->accept_mutator(this));

  std::vector<Buf*> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (Buf* buf_arg : v->buf_args()) {
    buf_args_new.push_back(dynamic_cast<Buf*>(buf_arg->accept_mutator(this)));
  }
  std::vector<Expr*> args_new;
  args_new.reserve(v->args().size());
  for (Expr* arg : v->args()) {
    args_new.push_back(arg->accept_mutator(this));
  }

  return new ExternalCall(buf_new, v->func_name(), buf_args_new, args_new);
}

Stmt* IRCloner::mutate(Let* v) {
  auto value_new = v->value()->accept_mutator(this);
  return new Let(v->var(), value_new);
}

Stmt* IRCloner::mutate(Cond* v) {
  auto condition_new = v->condition()->accept_mutator(this);
  Stmt* true_old = v->true_stmt();
  Stmt* false_old = v->false_stmt();
  Stmt* true_new = true_old ? true_old->accept_mutator(this) : true_old;
  Stmt* false_new = false_old ? false_old->accept_mutator(this) : false_old;
  return new Cond(condition_new, true_new, false_new);
}

Stmt* Stmt::clone(Stmt* s) {
  IRCloner cloner;
  Stmt* cloned = s->accept_mutator(&cloner);
  set_parent(cloned, nullptr);
  return cloned;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
