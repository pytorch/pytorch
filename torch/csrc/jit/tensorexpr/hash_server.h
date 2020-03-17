#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

#define CACHE_GUARD()  \
  if (cachedHash(v)) { \
    return;            \
  }

using SimplifierHashType = size_t;

/* Expression hasher providing comparable values representing sub-exprs.
 * Uses memoization to avoid excessive recursion. */
class HashProvider : public IRVisitor {
 public:
  template <class T>
  SimplifierHashType hash(const T* e) {
    e->accept(this);
    return hashOf(e);
  }

  bool cachedHash(const KernelScopedObject* e) {
    return exprToHash_.find(e) != exprToHash_.end();
  }

  void visit(const Add* v) override {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "+", hashOf(v->rhs())));
  }

  void visit(const Sub* v) override {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "-", hashOf(v->rhs())));
  }

  void visit(const Mul* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "*", hashOf(v->rhs())));
  }

  void visit(const Div* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "/", hashOf(v->rhs())));
  }

  void visit(const Mod* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "%", hashOf(v->rhs())));
  }

  void visit(const Max* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "Mx", hashOf(v->rhs())));
  }

  void visit(const Min* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "Mn", hashOf(v->rhs())));
  }

  void visit(const And* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "&", hashOf(v->rhs())));
  }

  void visit(const Or* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "|", hashOf(v->rhs())));
  }

  void visit(const Xor* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "^", hashOf(v->rhs())));
  }

  void visit(const Lshift* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), "<<", hashOf(v->rhs())));
  }

  void visit(const Rshift* v) {
    CACHE_GUARD();
    v->lhs()->accept(this);
    v->rhs()->accept(this);
    putHash(v, hash_combine(hashOf(v->lhs()), ">>", hashOf(v->rhs())));
  }

  void visit(const CompareSelect* v) {
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

// NOLINTNEXTLINE
#define IMM_VISIT(Type, Name)                    \
  void visit(const Name##Imm* v) {               \
    CACHE_GUARD();                               \
    putHash(v, hash_combine(#Name, v->value())); \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_VISIT);
#undef IMM_VISIT

  void visit(const Cast* v) {
    CACHE_GUARD();
    v->src_value()->accept(this);
    putHash(v, hash_combine("cast", v->dtype(), hashOf(v->src_value())));
  }

  void visit(const Var* v) {
    CACHE_GUARD();
    putHash(v, hash_combine("var", name_manager_.get_unique_name(v)));
  }

  void visit(const Let* v) {
    CACHE_GUARD();
    v->var()->accept(this);
    v->value()->accept(this);
    v->body()->accept(this);

    putHash(
        v,
        hash_combine(
            "let", hashOf(v->var()), hashOf(v->value()), hashOf(v->body())));
  }

  void visit(const LetStmt* v) {
    CACHE_GUARD();
    v->var()->accept(this);
    v->value()->accept(this);
    v->body()->accept(this);
    putHash(
        v,
        hash_combine(
            "letstmt",
            hashOf(v->var()),
            hashOf(v->value()),
            hashOf(v->body())));
  }

  void visit(const Ramp* v) {
    CACHE_GUARD();
    v->base()->accept(this);
    v->stride()->accept(this);
    putHash(
        v,
        hash_combine(
            "ramp", hashOf(v->base()), hashOf(v->stride()), v->lanes()));
  }

  void visit(const Load* v) {
    CACHE_GUARD();
    v->base_handle()->accept(this);
    v->index()->accept(this);
    v->mask()->accept(this);
    putHash(
        v,
        hash_combine(
            "load",
            hashOf(v->base_handle()),
            hashOf(v->index()),
            hashOf(v->mask())));
  }

  void visit(const Store* v) {
    CACHE_GUARD();
    v->base_handle()->accept(this);
    v->index()->accept(this);
    v->value()->accept(this);
    v->mask()->accept(this);
    putHash(
        v,
        hash_combine(
            "store",
            hashOf(v->base_handle()),
            hashOf(v->index()),
            hashOf(v->value()),
            hashOf(v->mask())));
  }

  void visit(const Block* v) {
    CACHE_GUARD();
    SimplifierHashType hash;
    for (Stmt* s : v->stmts()) {
      s->accept(this);
      hash = hash_combine(hash, hashOf(s));
    }
    putHash(v, hash);
  }

  void visit(const For* v) {
    CACHE_GUARD();
    v->var()->accept(this);
    v->start()->accept(this);
    v->stop()->accept(this);

    SimplifierHashType hash = hash_combine(
        "for", hashOf(v->var()), hashOf(v->start()), hashOf(v->stop()));
    if (v->body()) {
      v->body()->accept(this);
      hash = hash_combine(hash, hashOf(v->body()));
    }

    putHash(v, hash);
  }

  void visit(const Broadcast* v) {
    CACHE_GUARD();
    v->value()->accept(this);
    putHash(v, hash_combine("broadcast", hashOf(v->value()), v->lanes()));
  }

  void visit(const IfThenElse* v) {
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

  void visit(const BaseCallNode* v) {
    CACHE_GUARD();
    SimplifierHashType hash = std::hash<std::string>()(v->func_name());
    for (int i = 0; i < v->nparams(); i++) {
      v->param(i)->accept(this);
      hash = hash_combine(hash, hashOf(v->param(i)));
    }

    putHash(v, hash);
  }

  void visit(const Allocate* v) {
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

  void visit(const Free* v) {
    CACHE_GUARD();
    const Var* buffer_var = v->buffer_var();
    buffer_var->accept(this);

    putHash(v, hash_combine("free", hashOf(buffer_var)));
  }

  void visit(const Cond* v) {
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

 private:
  SimplifierHashType hashOf(const Expr* e) {
    auto it = exprToHash_.find(e);
    if (it != exprToHash_.end()) {
      return it->second;
    }

    // As a failsafe fall back to IRPrinter.
    std::stringstream ss;
    IRPrinter printer(ss);
    e->accept(&printer);
    SimplifierHashType hash = std::hash<std::string>()(ss.str());
    putHash(e, hash);

    return hash;
  }

  SimplifierHashType hashOf(const Stmt* s) {
    auto it = exprToHash_.find(s);
    if (it != exprToHash_.end()) {
      return it->second;
    }

    // As a failsafe fall back to IRPrinter.
    std::stringstream ss;
    IRPrinter printer(ss);
    s->accept(&printer);
    SimplifierHashType hash = std::hash<std::string>()(ss.str());
    putHash(s, hash);

    return hash;
  }

  // Hash funcs for various types, numbers are random.
  template <typename T>
  void _hash_combine(SimplifierHashType& seed, const T& val) {
    seed ^= std::hash<T>()(val) + 0x1f752c19 + (seed << 7) + (seed >> 4);
  }

  void _hash_combine(SimplifierHashType& seed, const char* val) {
    seed ^=
        std::hash<std::string>()(val) + 0x1f752c19 + (seed << 7) + (seed >> 4);
  }

  // at:::Half doesn't have a std::hash, so cast to short.
  void _hash_combine(SimplifierHashType& seed, const at::Half& val) {
    seed ^= std::hash<uint16_t>()((uint16_t)val) + 0x1f752c19 + (seed << 7) +
        (seed >> 4);
  }

  void _hash_combine(SimplifierHashType& seed, const Dtype& val) {
    seed ^= std::hash<std::string>()(val.ToCppString()) + 0x1f752c19 +
        (seed << 7) + (seed >> 4);
  }

  template <typename T, typename... Types>
  void _hash_combine(
      SimplifierHashType& seed,
      const T& val,
      const Types&... args) {
    _hash_combine(seed, val);
    _hash_combine(seed, args...);
  }

  template <typename... Types>
  SimplifierHashType hash_combine(const Types&... args) {
    SimplifierHashType seed = 0;
    _hash_combine(seed, args...);
    return seed;
  }

  void putHash(const KernelScopedObject* e, SimplifierHashType h) {
    auto res = exprToHash_.emplace(e, h);
    if (res.second == false) {
      // This is always a logic bug since we should check the cache first.
      throw std::runtime_error("hash collision");
    }
  }

  std::unordered_map<const KernelScopedObject*, SimplifierHashType> exprToHash_;
  UniqueNameManager name_manager_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
