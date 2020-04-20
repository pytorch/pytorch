#include <torch/csrc/jit/tensorexpr/cost_estimator.h>

#include <set>

namespace torch {
namespace jit {
namespace tensorexpr {

const Stmt* CostEstimator::getSharedParent(const Stmt* a, const Stmt* b) {
  std::set<const Stmt*> ancestors;
  const Stmt* p = a;
  while (p) {
    ancestors.insert(p);
    p = p->get_parent();
  }

  p = b;
  bool found = false;
  while (p) {
    if (ancestors.count(p) > 0) {
      found = true;
      break;
    }
    p = p->get_parent();
  }

  return p;
}

void CostEstimator::visit(const Add* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Sub* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Mul* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Div* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Mod* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Max* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Min* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const And* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Or* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Xor* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Lshift* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const Rshift* v) {
  scanBinaryOp(v);
}

void CostEstimator::visit(const CompareSelect* v) {
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  v->ret_val1()->accept(this);
  v->ret_val2()->accept(this);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  auto lhsInfo = exprInfo_[hasher_.hash(v->lhs())];
  auto rhsInfo = exprInfo_[hasher_.hash(v->rhs())];
  auto ret1Info = exprInfo_[hasher_.hash(v->ret_val1())];
  auto ret2Info = exprInfo_[hasher_.hash(v->ret_val2())];

  const Expr* cost = new Add(
      getImmediateByType(kInt, dict_.COMPARE_OP_COST),
      new Add(lhsInfo.cost, rhsInfo.cost));
  cost = new Add(cost, new Max(ret1Info.cost, ret2Info.cost, true));

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Cast* v) {
  v->src_value()->accept(this);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  auto srcInfo = exprInfo_[hasher_.hash(v->src_value())];
  const Expr* cost =
      new Add(getImmediateByType(kInt, dict_.CAST_OP_COST), srcInfo.cost);

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Var* v) {
  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  exprInfo_.emplace(
      hash,
      SubExprInfo(1, getImmediateByType(kInt, dict_.VAR_REF_COST), lastBlock_));
}

void CostEstimator::visit(const Ramp* v) {
  v->base()->accept(this);
  v->stride()->accept(this);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  auto baseInfo = exprInfo_[hasher_.hash(v->base())];
  auto strideInfo = exprInfo_[hasher_.hash(v->stride())];
  const Expr* cost = new Add(
      baseInfo.cost,
      new Mul(strideInfo.cost, getImmediateByType(kInt, v->lanes())));

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Load* v) {
  v->base_handle()->accept(this);
  const Expr* cost = getImmediateByType(kInt, dict_.LOAD_OP_COST);

  for (const Expr* ind : v->indices()) {
    ind->accept(this);
    cost = new Add(cost, exprInfo_[hasher_.hash(ind)].cost);
  }
  v->mask()->accept(this);
  cost = new Add(cost, exprInfo_[hasher_.hash(v->mask())].cost);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Store* v) {
  v->base_handle()->accept(this);
  const Expr* cost = getImmediateByType(kInt, dict_.STORE_OP_COST);

  for (const Expr* ind : v->indices()) {
    ind->accept(this);
    cost = new Add(cost, exprInfo_[hasher_.hash(ind)].cost);
  }

  v->value()->accept(this);
  cost = new Add(cost, exprInfo_[hasher_.hash(v->value())].cost);
  v->mask()->accept(this);

  // TODO: does the mark affect the cost of the store (ie. mask of 0 is free?).
  cost = new Add(cost, exprInfo_[hasher_.hash(v->mask())].cost);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Block* v) {
  // push this block onto block stack.
  const Stmt* prev = lastBlock_;
  lastBlock_ = v;

  const Expr* cost = getImmediateByType(kInt, 0);
  for (Stmt* s : *v) {
    s->accept(this);
    cost = new Add(cost, exprInfo_[hasher_.hash(s)].cost);
  }

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));

  // pop this block.
  lastBlock_ = prev;
}

void CostEstimator::visit(const For* v) {
  v->var()->accept(this);
  v->start()->accept(this);
  v->stop()->accept(this);

  const Expr* var_cost = exprInfo_[hasher_.hash(v->var())].cost;
  auto start_cost = exprInfo_[hasher_.hash(v->start())].cost;
  auto stop_cost = exprInfo_[hasher_.hash(v->stop())].cost;

  // cost of For
  //  = start_cost + (loops+1) * (compare_cost) + loops * (body_cost +
  //  increment_cost).
  const Expr* loops = new Sub(v->stop(), v->start());

  const Expr* compare_cost = new Add(
      getImmediateByType(kInt, dict_.COMPARE_OP_COST),
      new Add(var_cost, stop_cost));
  const Expr* increment_cost = getImmediateByType(kInt, dict_.BINARY_OP_COST);

  const Expr* cost = new Add(
      start_cost,
      new Mul(new Add(getImmediateByType(kInt, 1), loops), compare_cost));

  cost = new Add(cost, new Mul(loops, increment_cost));

  if (v->body()) {
    v->body()->accept(this);
    auto bodyInfo = exprInfo_[hasher_.hash(v->body())];
    cost = new Add(cost, new Mul(loops, bodyInfo.cost));
  }

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Broadcast* v) {
  v->value()->accept(this);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  const Expr* cost = new Mul(
      exprInfo_[hasher_.hash(v->value())].cost,
      getImmediateByType(kInt, v->lanes()));

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const IfThenElse* v) {
  v->condition()->accept(this);
  v->true_value()->accept(this);
  v->false_value()->accept(this);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  auto conditionInfo = exprInfo_[hasher_.hash(v->condition())];
  auto trueInfo = exprInfo_[hasher_.hash(v->true_value())];
  auto falseInfo = exprInfo_[hasher_.hash(v->false_value())];

  const Expr* cost =
      new Add(conditionInfo.cost, new Max(trueInfo.cost, falseInfo.cost, true));

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const BaseCallNode* v) {
  const Expr* cost = getImmediateByType(kInt, dict_.CALL_OP_COST);
  for (int i = 0; i < v->nparams(); i++) {
    v->param(i)->accept(this);
    cost = new Add(cost, exprInfo_[hasher_.hash(v->param(i))].cost);
  }

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Allocate* v) {
  v->buffer_var()->accept(this);
  const Expr* cost = exprInfo_[hasher_.hash(v->buffer_var())].cost;

  const Expr* allocWeight = getImmediateByType(kFloat, dict_.ALLOC_COST);
  std::vector<const Expr*> dims = v->dims();
  for (const Expr* dim : dims) {
    dim->accept(this);
    allocWeight = new Mul(dim, allocWeight);
    cost = new Add(cost, exprInfo_[hasher_.hash(dim)].cost);
  }

  cost = new Add(cost, allocWeight);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Free* v) {
  v->buffer_var()->accept(this);

  const Expr* cost = new Add(
      exprInfo_[hasher_.hash(v->buffer_var())].cost,
      getImmediateByType(kInt, dict_.FREE_OP_COST));

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

void CostEstimator::visit(const Cond* v) {
  v->condition()->accept(this);
  v->true_stmt()->accept(this);
  v->false_stmt()->accept(this);

  auto hash = hasher_.hash(v);
  if (canUpdateExisting(v, hash)) {
    return;
  }

  auto condInfo = exprInfo_[hasher_.hash(v->condition())];
  auto trueInfo = exprInfo_[hasher_.hash(v->true_stmt())];
  auto falseInfo = exprInfo_[hasher_.hash(v->false_stmt())];

  const Expr* cost =
      new Add(condInfo.cost, new Max(trueInfo.cost, falseInfo.cost, true));

  exprInfo_.emplace(
      hash, SubExprInfo(1, IRSimplifier::simplify(cost), lastBlock_));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
