#include <torch/csrc/jit/tensorexpr/hash_provider.h>

#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

#include <c10/util/irange.h>

namespace torch::jit::tensorexpr {

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

void HashProvider::visit(AddPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "+", hashOf(v->rhs())));
}

void HashProvider::visit(SubPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "-", hashOf(v->rhs())));
}

void HashProvider::visit(MulPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "*", hashOf(v->rhs())));
}

void HashProvider::visit(DivPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "/", hashOf(v->rhs())));
}

void HashProvider::visit(ModPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "%", hashOf(v->rhs())));
}

void HashProvider::visit(RoundOffPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "rof", hashOf(v->rhs())));
}

void HashProvider::visit(MaxPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "Mx", hashOf(v->rhs())));
}

void HashProvider::visit(MinPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "Mn", hashOf(v->rhs())));
}

void HashProvider::visit(AndPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "&", hashOf(v->rhs())));
}

void HashProvider::visit(OrPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "|", hashOf(v->rhs())));
}

void HashProvider::visit(XorPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "^", hashOf(v->rhs())));
}

void HashProvider::visit(LshiftPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), "<<", hashOf(v->rhs())));
}

void HashProvider::visit(RshiftPtr v) {
  CACHE_GUARD();
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  putHash(v, hash_combine(hashOf(v->lhs()), ">>", hashOf(v->rhs())));
}

void HashProvider::visit(CompareSelectPtr v) {
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

void HashProvider::visit(CastPtr v) {
  CACHE_GUARD();
  v->src_value()->accept(this);
  putHash(v, hash_combine("cast", v->dtype(), hashOf(v->src_value())));
}

void HashProvider::visit(VarPtr v) {
  CACHE_GUARD();
  putHash(v, hash_combine("var", name_manager_.get_unique_name(v)));
}

void HashProvider::visit(RampPtr v) {
  CACHE_GUARD();
  v->base()->accept(this);
  v->stride()->accept(this);
  putHash(
      v,
      hash_combine("ramp", hashOf(v->base()), hashOf(v->stride()), v->lanes()));
}

void HashProvider::visit(LoadPtr v) {
  CACHE_GUARD();
  v->base_handle()->accept(this);
  SimplifierHashType indices_hash;
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
    indices_hash = hash_combine(indices_hash, hashOf(ind));
  }
  putHash(v, hash_combine("load", hashOf(v->base_handle()), indices_hash));
}

void HashProvider::visit(StorePtr v) {
  CACHE_GUARD();
  v->base_handle()->accept(this);
  SimplifierHashType indices_hash;
  for (const ExprPtr& ind : v->indices()) {
    ind->accept(this);
    indices_hash = hash_combine(indices_hash, hashOf(ind));
  }
  v->value()->accept(this);
  putHash(
      v,
      hash_combine(
          "store", hashOf(v->base_handle()), indices_hash, hashOf(v->value())));
}

void HashProvider::visit(BlockPtr v) {
  CACHE_GUARD();
  SimplifierHashType hash;

  for (const StmtPtr& s : *v) {
    s->accept(this);
    hash = hash_combine(hash, hashOf(s));
  }
  putHash(v, hash);
}

void HashProvider::visit(ForPtr v) {
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

void HashProvider::visit(BroadcastPtr v) {
  CACHE_GUARD();
  v->value()->accept(this);
  putHash(v, hash_combine("broadcast", hashOf(v->value()), v->lanes()));
}

void HashProvider::visit(IfThenElsePtr v) {
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

void HashProvider::visit(IntrinsicsPtr v) {
  CACHE_GUARD();
  // calls to rand are not symbolic and have a different value each time, they
  // should not hash to anything and this is the best we can do.
  if (v->op_type() == kRand) {
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
    putHash(v, (SimplifierHashType)rand());
    return;
  }

  SimplifierHashType hash(te_hash(v->func_name()));
  for (const auto i : c10::irange(v->nparams())) {
    v->param(i)->accept(this);
    hash = hash_combine(hash, hashOf(v->param(i)));
  }

  putHash(v, hash);
}

void HashProvider::visit(AllocatePtr v) {
  CACHE_GUARD();
  VarPtr buffer_var = v->buffer_var();
  buffer_var->accept(this);

  SimplifierHashType hash =
      hash_combine("allocate", hashOf(buffer_var), v->dtype());

  std::vector<ExprPtr> dims = v->dims();
  for (const ExprPtr& dim : dims) {
    dim->accept(this);
    hash = hash_combine(hash, hashOf(dim));
  }
  putHash(v, hash);
}

void HashProvider::visit(FreePtr v) {
  CACHE_GUARD();
  VarPtr buffer_var = v->buffer_var();
  buffer_var->accept(this);

  putHash(v, hash_combine("free", hashOf(buffer_var)));
}

void HashProvider::visit(CondPtr v) {
  CACHE_GUARD();
  ExprPtr condition = v->condition();
  StmtPtr true_stmt = v->true_stmt();
  StmtPtr false_stmt = v->false_stmt();
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

void HashProvider::visit(TermPtr v) {
  CACHE_GUARD();
  v->scalar()->accept(this);

  SimplifierHashType hash = hash_combine("term", hashOf(v->scalar()));
  for (const auto& c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  putHash(v, hash);
}

void HashProvider::visit(PolynomialPtr v) {
  CACHE_GUARD();
  v->scalar()->accept(this);

  SimplifierHashType hash = hash_combine("term", hashOf(v->scalar()));
  for (const auto& c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  putHash(v, hash);
}

void HashProvider::visit(MaxTermPtr v) {
  CACHE_GUARD();
  SimplifierHashType hash = hash_combine("maxterm");
  if (v->scalar()) {
    v->scalar()->accept(this);
    hash = hash_combine(hash, hashOf(v->scalar()));
  }

  for (const auto& c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  putHash(v, hash);
}

void HashProvider::visit(MinTermPtr v) {
  CACHE_GUARD();
  SimplifierHashType hash = hash_combine("minterm");
  if (v->scalar()) {
    v->scalar()->accept(this);
    hash = hash_combine(hash, hashOf(v->scalar()));
  }

  for (const auto& c : v->variables()) {
    c->accept(this);
    hash = hash_combine(hash, hashOf(c));
  }

  putHash(v, hash);
}

} // namespace torch::jit::tensorexpr
