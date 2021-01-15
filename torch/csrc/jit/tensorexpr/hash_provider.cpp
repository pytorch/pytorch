#include <torch/csrc/jit/tensorexpr/hash_provider.h>

#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch {
namespace jit {
namespace tensorexpr {

bool SimplifierHashType::operator==(const SimplifierHashType& other) const {
  return _h == other._h;
}

bool SimplifierHashType::operator!=(const SimplifierHashType& other) const {
  return _h != other._h;
}

bool SimplifierHashType::operator<(const SimplifierHashType& other) const {
  return _h < other._h;
}

bool SimplifierHashType::operator==(const size_t other) const {
  return _h == other;
}

bool SimplifierHashType::operator!=(const size_t other) const {
  return _h != other;
}

void HashProvider::visit(const Add* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "+", hashOf(v->rhs())));
}

void HashProvider::visit(const Sub* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "-", hashOf(v->rhs())));
}

void HashProvider::visit(const Mul* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "*", hashOf(v->rhs())));
}

void HashProvider::visit(const Div* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "/", hashOf(v->rhs())));
}

void HashProvider::visit(const Mod* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "%", hashOf(v->rhs())));
}

void HashProvider::visit(const Max* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "Mx", hashOf(v->rhs())));
}

void HashProvider::visit(const Min* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "Mn", hashOf(v->rhs())));
}

void HashProvider::visit(const And* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "&", hashOf(v->rhs())));
}

void HashProvider::visit(const Or* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "|", hashOf(v->rhs())));
}

void HashProvider::visit(const Xor* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "^", hashOf(v->rhs())));
}

void HashProvider::visit(const Lshift* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "<<", hashOf(v->rhs())));
}

void HashProvider::visit(const Rshift* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), ">>", hashOf(v->rhs())));
}

void HashProvider::visit(const CompareSelect* v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  v->ret_val1()->accept(this);
  v->ret_val2()->accept(this);
  putHash(
      v,
      hash_combine(
          hashOf(v->lhs()),
          (int)v->compare_select_op(),
          hashOf(v->rhs()),
          hashOf(v->ret_val1()),
          hashOf(v->ret_val2())));
}

void HashProvider::visit(const Cast* v) {
  CACHE_GUARD();
  v->src_value()->accept(this);
  putHash(v, hash_combine("cast", v->dtype(), hashOf(v->src_value())));
}

void HashProvider::visit(const Var* v) {
  CACHE_GUARD();
  putHash(v, hash_combine("var", name_manager_.get_unique_name(v)));
}

void HashProvider::visit(const Ramp* v) {
  CACHE_GUARD();
  v->base()->accept(this);
  v->stride()->accept(this);
  putHash(
      v,
      hash_combine("ramp", hashOf(v->base()), hashOf(v->stride()), v->lanes()));
}

void HashProvider::visit(const Load* v) {
  CACHE_GUARD();
  v->base_handle()->accept(this);
  SimplifierHashType indices_hash;
  for (const Expr* ind : v->indices()) {
    ind->accept(this);
    indices_hash = hash_combine(indices_hash, hashOf(ind));
  }
  v->mask()->accept(this);
  putHash(
      v,
      hash_combine(
          "load", hashOf(v->base_handle()), indices_hash, hashOf(v->mask())));
}

void HashProvider::visit(const Store* v) {
  CACHE_GUARD();
  v->base_handle()->accept(this);
  SimplifierHashType indices_hash;
  for (const Expr* ind : v->indices()) {
    ind->accept(this);
    indices_hash = hash_combine(indices_hash, hashOf(ind));
  }
  v->value()->accept(this);
  v->mask()->accept(this);
  putHash(
      v,
      hash_combine(
          "store",
          hashOf(v->base_handle()),
          indices_hash,
          hashOf(v->value()),
          hashOf(v->mask())));
}

void HashProvider::visit(const Block* v) {
  CACHE_GUARD();
  SimplifierHashType hash;

  for (Stmt* s : *v) {
    s->accept(this);
    hash = hash_combine(hash, hashOf(s));
  }
  putHash(v, hash);
}

void HashProvider::visit(const For* v) {
  CACHE_GUARD();
  v->var()->accept(this);
  v->start()->accept(this);
  v->stop()->accept(this);

  SimplifierHashType hash = hash_combine(
      "for", hashOf(v->var()), hashOf(v->start()), hashOf(v->stop()));
  hash = hash_combine(hash, v->loop_options().ToString());
  if (v->body()) {
    v->body()->accept(this);
    hash = hash_combine(hash, hashOf(v->body()));
  }

  putHash(v, hash);
}

void HashProvider::visit(const Broadcast* v) {
  CACHE_GUARD();
  v->value()->accept(this);
  putHash(v, hash_combine("broadcast", hashOf(v->value()), v->lanes()));
}

void HashProvider::visit(const IfThenElse* v) {
  CACHE_GUARD();
  v->condition()->accept(this);
  v->true_value()->accept(this);
  v->false_value()->accept(this);

  putHash(
      v,
      hash_combine(
          "ifthenelse",
          hashOf(v->condition()),
          hashOf(v->true_value()),
          hashOf(v->false_value())));
}

void HashProvider::visit(const BaseCallNode* v) {
  CACHE_GUARD();
  SimplifierHashType hash(te_hash(v->func_name()));
  for (int i = 0; i < v->nparams(); i++) {
    v->param(i)->accept(this);
    hash = hash_combine(hash, hashOf(v->param(i)));
  }

  putHash(v, hash);
}

void HashProvider::visit(const Intrinsics* v) {
  CACHE_GUARD();
  // calls to rand are not symbolic and have a different value each time, they
  // should not hash to anything and this is the best we can do.
  if (v->op_type() == kRand) {
    putHash(v, (SimplifierHashType)rand());
    return;
  }

  SimplifierHashType hash(te_hash(v->func_name()));
  for (int i = 0; i < v->nparams(); i++) {
    v->param(i)->accept(this);
    hash = hash_combine(hash, hashOf(v->param(i)));
  }

  putHash(v, hash);
}

void HashProvider::visit(const Allocate* v) {
  CACHE_GUARD();
  const Var* buffer_var = v->buffer_var();
  buffer_var->accept(this);

  SimplifierHashType hash =
      hash_combine("allocate", hashOf(buffer_var), v->dtype());

  std::vector<const Expr*> dims = v->dims();
  for (const Expr* dim : dims) {
    dim->accept(this);
    hash = hash_combine(hash, hashOf(dim));
  }
  putHash(v, hash);
}

void HashProvider::visit(const Free* v) {
  CACHE_GUARD();
  const Var* buffer_var = v->buffer_var();
  buffer_var->accept(this);

  putHash(v, hash_combine("free", hashOf(buffer_var)));
}

void HashProvider::visit(const Cond* v) {
  CACHE_GUARD();
  const Expr* condition = v->condition();
  Stmt* true_stmt = v->true_stmt();
  Stmt* false_stmt = v->false_stmt();
  condition->accept(this);

  SimplifierHashType hash = hash_combine("cond", hashOf(condition));
  if (true_stmt) {
    true_stmt->accept(this);
    hash = hash_combine(hash, hashOf(true_stmt));
  }
  if (false_stmt) {
    false_stmt->accept(this);
    hash = hash_combine(hash, hashOf(false_stmt));
  }

  putHash(v, hash);
}

void HashProvider::visit(const Term* v) {
  CACHE_GUARD();
  v->scalar()->accept(this);

  SimplifierHashType hash = hash_combine("term", hashOf(v->scalar()));
  for (auto* c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  putHash(v, hash);
}

void HashProvider::visit(const Polynomial* v) {
  CACHE_GUARD();
  v->scalar()->accept(this);

  SimplifierHashType hash = hash_combine("term", hashOf(v->scalar()));
  for (auto* c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  putHash(v, hash);
}

void HashProvider::visit(const MaxTerm* v) {
  CACHE_GUARD();
  SimplifierHashType hash = hash_combine("maxterm");
  if (v->scalar()) {
    v->scalar()->accept(this);
    hash = hash_combine(hash, hashOf(v->scalar()));
  }

  for (auto* c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  putHash(v, hash);
}

void HashProvider::visit(const MinTerm* v) {
  CACHE_GUARD();
  SimplifierHashType hash = hash_combine("minterm");
  if (v->scalar()) {
    v->scalar()->accept(this);
    hash = hash_combine(hash, hashOf(v->scalar()));
  }

  for (auto* c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  putHash(v, hash);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
