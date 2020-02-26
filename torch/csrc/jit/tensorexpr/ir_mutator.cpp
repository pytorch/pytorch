#include "torch/csrc/jit/tensorexpr/ir_mutator.h"

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

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
    case IRNodeType::kXor:
      return new Xor(lhs_new, rhs_new);
    case IRNodeType::kLshift:
      return new Lshift(lhs_new, rhs_new);
    case IRNodeType::kRshift:
      return new Rshift(lhs_new, rhs_new);
    default:
      LOG(FATAL) << "unsupported expr_type: " << static_cast<int>(expr_type);
      return nullptr;
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
             v->compare_select_op())
      .node();
}

const Expr* IRMutator::mutate(const IntImm* v) {
  return v;
}

const Expr* IRMutator::mutate(const FloatImm* v) {
  return v;
}

const Expr* IRMutator::mutate(const Cast* v) {
  const Expr* src_value = v->src_value();
  const Expr* src_value_new = src_value->accept_mutator(this);
  if (src_value_new == v->src_value()) {
    return v;
  }
  return new Cast(v->dtype(), src_value_new);
}

const Expr* IRMutator::mutate(const Var* v) {
  return v;
}

const Expr* IRMutator::mutate(const Let* v) {
  const Expr* var = v->var();
  const Expr* value = v->value();
  const Expr* body = v->body();
  const Expr* var_new = var->accept_mutator(this);
  const Expr* value_new = value->accept_mutator(this);
  const Expr* body_new = body->accept_mutator(this);
  if ((var == var_new) && (value == value_new) &&
      (body == body_new)) {
    return v;
  }
  return new Let(var_new, value_new, body_new);
}

Stmt* IRMutator::mutate(const LetStmt* v) {
  const Var* var = v->var();
  const Expr* value = v->value();
  Stmt* body = v->body();
  const Var* var_new = dynamic_cast<const Var*>(var->accept_mutator(this));
  if (var_new == nullptr) {
    throw std::runtime_error("LetStmt var must be variable");
  }
  const Expr* value_new = value->accept_mutator(this);
  Stmt* body_new = body->accept_mutator(this);
  if ((var == var_new) && (value == value_new) &&
      (body == body_new)) {
    return (Stmt*)v;
  }
  return new LetStmt(var_new, value_new, body_new);
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
  const Var* base_handle = v->base_handle();
  const Expr* index = v->index();
  const Expr* mask = v->mask();
  const Expr* base_handle_expr = base_handle->accept_mutator(this);
  const Var* base_handle_new = dynamic_cast<const Var*>(base_handle_expr);
  const Expr* index_new = index->accept_mutator(this);
  const Expr* mask_new = mask->accept_mutator(this);
  if (base_handle == base_handle_new && index == index_new &&
      mask == mask_new) {
    return v;
  }
  return new Load(dtype, base_handle_new, index_new, mask_new);
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
  if (condition == condition_new &&
      true_value == true_value_new &&
      false_value == false_value_new) {
    return v;
  }

  return new IfThenElse(condition_new, true_value_new, false_value_new);
}

const Expr* IRMutator::mutate(const Intrinsics* v) {
  const BaseCallNode* base = v;
  return this->mutate(base);
}

const Expr* IRMutator::mutate(const FunctionCall* v) {
  const BaseCallNode* base = v;
  return this->mutate(base);
}

const Expr* IRMutator::mutate(const BaseCallNode* v) {
  std::vector<const Expr*> params(v->nparams());
  bool any_change = false;
  for (int i = 0; i < v->nparams(); i++) {
    const Expr* value = v->param(i);
    const Expr* value_new = value->accept_mutator(this);
    if (value != value_new) {
      any_change = true;
    }
    params[i] = std::move(value_new);
  }
  if (!any_change) {
    return v;
  }
  return v->DefaultMutator(params);
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
  if (var == var_new && start == start_new &&
      stop == stop_new && body == body_new) {
    return (Stmt*)v;
  }
  return new For(var_new, start_new, stop_new, body_new, loop_options);
}

Stmt* IRMutator::mutate(const Block* v) {
  bool any_change = false;
  std::vector<Stmt*> stmts;
  for (int i = 0; i < v->nstmts(); i++) {
    Stmt* stmt = v->stmt(i);
    Stmt* stmt_new = stmt->accept_mutator(this);
    if (stmt != stmt_new) {
      any_change = true;
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
  const Var* base_handle = v->base_handle();
  const Expr* index = v->index();
  const Expr* value = v->value();
  const Expr* mask = v->mask();
  const Expr* base_handle_expr = base_handle->accept_mutator(this);
  const Var* base_handle_new = dynamic_cast<const Var*>(base_handle_expr);
  const Expr* index_new = index->accept_mutator(this);
  const Expr* value_new = value->accept_mutator(this);
  const Expr* mask_new = mask->accept_mutator(this);
  if (base_handle == base_handle_new && index == index_new &&
      value == value_new && mask == mask_new) {
    return (Stmt*)v;
  }
  return new Store(base_handle_new, index_new, value_new, mask_new);
}

Stmt* IRMutator::mutate(const Allocate* v) {
  const Var* buffer_var_old = v->buffer_var();
  const Var* buffer_var_new =
      dynamic_cast<const Var*>(buffer_var_old->accept_mutator(this));
  bool any_change = buffer_var_new == buffer_var_old;

  std::vector<const Expr*> dims_old = v->dims();
  std::vector<const Expr*> dims_new(dims_old.size());
  for (size_t i = 0; i < dims_old.size(); i++) {
    dims_new[i] = dims_old[i]->accept_mutator(this);
    any_change |= (dims_new[i] == dims_old[i]);
  }

  if (!any_change) {
    return (Stmt*)v;
  }

  return new Allocate(buffer_var_new, v->dtype(), dims_new);
}

Stmt* IRMutator::mutate(const Free* v) {
  const Expr* buffer_var_old = v->buffer_var();
  const Var* buffer_var_new = dynamic_cast<const Var*>(buffer_var_old->accept_mutator(this));
  if (buffer_var_new == buffer_var_old) {
    return (Stmt*)v;
  }

  return new Free(buffer_var_new);
}

Stmt* IRMutator::mutate(const Cond* v) {
  const Expr* cond_old = v->condition();
  Stmt* true_old = v->true_stmt();
  Stmt* false_old = v->false_stmt();

  const Expr* cond_new = cond_old->accept_mutator(this);
  Stmt* true_new = true_old ? true_old->accept_mutator(this) : true_old;
  Stmt* false_new = false_old ? false_old->accept_mutator(this) : false_old;

  if (cond_old == cond_new && true_old == true_new &&
      false_old == false_new) {
    return (Stmt*)v;
  }
  return new Cond(cond_new, true_new, false_new);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
