#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
namespace torch {
namespace jit {
namespace fuser {

namespace scope_utils {

// START SCOPE HELPER SYSTEMS
namespace {

struct Loops : private OptInDispatch {
 private:
  std::deque<ForLoop*> loops;
  void handle(ForLoop* fl) final {
    loops.insert(loops.begin(), fl);
  }

  void handle(IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static std::vector<ForLoop*> getLoops(Expr* scope) {
    Loops loops;
    Expr* it = scope;
    while (it != nullptr) {
      loops.handle(it);
      it = scope_utils::getParent(it);
    }
    return std::vector<ForLoop*>(loops.loops.begin(), loops.loops.end());
  }
};

struct forLoopCount : private OptInDispatch {
 private:
  unsigned int count_ = 0;

  void handle(ForLoop* fl) final {
    count_++;
  }

  void handle(IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static unsigned int get(Expr* scope) {
    forLoopCount flc;
    Expr* it = scope;
    while (it != nullptr) {
      flc.handle(it);
      it = scope_utils::getParent(it);
    }
    return flc.count_;
  }
};

struct scopePushBack : private OptInDispatch {
 private:
  Expr* expr_;
  void handle(ForLoop* fl) final {
    fl->body().push_back(expr_);
  }

  void handle(IfThenElse* ite) final {
    ite->body().push_back(expr_);
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  scopePushBack(Expr* expr) : expr_(expr) {}

 public:
  static void push(Expr* scope, Expr* expr) {
    scopePushBack pb(expr);
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    pb.handle(scope);
  }
};

struct scopeInsertBefore : private OptInDispatch {
 private:
  Expr* ref_;
  Expr* expr_;
  void handle(ForLoop* fl) final {
    fl->body().insert_before(ref_, expr_);
  }

  void handle(IfThenElse* ite) final {
    ite->body().insert_before(ref_, expr_);
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  scopeInsertBefore(Expr* ref, Expr* expr) : ref_(ref), expr_(expr) {}

 public:
  static void insert(Expr* scope, Expr* ref, Expr* expr) {
    scopeInsertBefore scb(ref, expr);
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    scb.handle(scope);
  }
};

struct parentScope : private OptInDispatch {
 private:
  Expr* parent_ = nullptr;

  void handle(ForLoop* fl) final {
    parent_ = fl->parentScope();
  }

  void handle(IfThenElse* ite) final {
    parent_ = ite->parentScope();
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static Expr* get(Expr* scope) {
    parentScope sp;
    sp.handle(scope);
    return sp.parent_;
  }
};

struct scopeClearExprs : private OptInDispatch {
 private:
  void handle(ForLoop* fl) final {
    fl->body().clear();
  }

  void handle(IfThenElse* ite) final {
    ite->body().clear();
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static void clear(Expr* scope) {
    scopeClearExprs sce;
    TORCH_INTERNAL_ASSERT(
        scope != nullptr, "Cannot clear scope, scope is a nullptr.");
    sce.handle(scope);
  }
};

void assertScope(Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      expr->getExprType() == ExprType::ForLoop ||
          expr->getExprType() == ExprType::IfThenElse,
      "Assert Scope failed when calling a scope_util function.");
}

struct CloneLoopNest : public OptOutMutator {
 private:
  Expr* parent_scope_ = nullptr;
  Expr* to_clone_ = nullptr;

  Statement* mutate(ForLoop* fl) final {
    std::vector<Expr*> mutated_exprs;
    for (Expr* expr : fl->body().exprs()) {
      mutated_exprs.push_back(ir_utils::asExpr(OptOutMutator::mutate(expr)));
    }
    if (fl == to_clone_)
      return new ForLoop(
          fl->index(), fl->iter_domain(), mutated_exprs, parent_scope_);
    return new ForLoop(
        fl->index(), fl->iter_domain(), mutated_exprs, fl->parentScope());
  }

  CloneLoopNest(Expr* _to_clone, Expr* _parent_scope)
      : parent_scope_(_parent_scope), to_clone_(_to_clone) {}

 public:
  static ForLoop* getClone(ForLoop* _to_clone, Expr* _parent_scope) {
    TORCH_INTERNAL_ASSERT(
        _to_clone != nullptr,
        "Tried to clone a scope, but received a nullptr.");
    CloneLoopNest cln(_to_clone, _parent_scope);
    return ir_utils::asForLoop(ir_utils::asExpr(cln.mutate(_to_clone)));
  }
};

struct ReplaceExprsInScope : public OptOutDispatch {
 private:
  std::unordered_map<Expr*, Expr*> replacement_map_;

  void handle(Expr* expr) final {
    OptOutDispatch::handle(expr);
  }

  void handle(ForLoop* fl) final {
    for (Expr* expr : fl->body().exprs()) {
      auto it = replacement_map_.find(expr);
      if (it == replacement_map_.end()) {
        handle(expr);
        continue;
      }
      fl->body().insert_before(expr, replacement_map_[expr]);
      fl->body().erase(expr);
    }
  }

  void handle(IfThenElse* ite) final {
    for (Expr* expr : ite->body().exprs()) {
      auto it = replacement_map_.find(expr);
      if (it == replacement_map_.end()) {
        handle(expr);
        continue;
      }
      ite->body().insert_before(expr, replacement_map_[expr]);
      ite->body().erase(expr);
    }
    for (Expr* expr : ite->elseBody().exprs()) {
      auto it = replacement_map_.find(expr);
      if (it == replacement_map_.end()) {
        handle(expr);
        continue;
      }
      ite->elseBody().insert_before(expr, replacement_map_[expr]);
      ite->elseBody().erase(expr);
    }
  }

  ReplaceExprsInScope(std::unordered_map<Expr*, Expr*> _replacement_map)
      : replacement_map_(std::move(_replacement_map)) {}

 public:
  static void replace(
      Expr* scope,
      std::unordered_map<Expr*, Expr*> replacement_map) {
    ReplaceExprsInScope reis(std::move(replacement_map));
    reis.handle(scope);
  }
};

struct FirstInnerMostScope : private OptInDispatch {
 private:
  Expr* active_scope = nullptr;

  void handle(ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    active_scope = nullptr;
  }

  void handle(IfThenElse* ite) final {
    for (auto expr : ite->body().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    for (auto expr : ite->elseBody().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    active_scope = nullptr;
  }

  Expr* getInner(Expr* expr) {
    OptInDispatch::handle(expr);
    return active_scope;
  }

 public:
  static Expr* get(Expr* scope) {
    TORCH_INTERNAL_ASSERT(
        scope != nullptr,
        "Tried to get inner most scope, but was provided nullptr.");

    FirstInnerMostScope fims;
    Expr* inner = fims.getInner(scope);
    while (fims.getInner(inner) != nullptr)
      inner = fims.getInner(inner);
    return inner;
  }
};

// END SCOPE HELPER SYSTEMS
} // namespace

// Grab the ForLoop starting from scope working out
std::vector<ForLoop*> getLoops(Expr* scope) {
  if (scope == nullptr)
    return std::vector<ForLoop*>();
  assertScope(scope);
  return Loops::getLoops(scope);
}

// Track how far our for loop scope is
unsigned int computeForDepth(Expr* scope) {
  if (scope == nullptr)
    return 0;
  assertScope(scope);
  return forLoopCount::get(scope);
}

// Push back an expr to scope
void pushBack(Expr* scope, Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr, "Scope is a nullptr, cannot push an expr to it.");
  assertScope(scope);
  scopePushBack::push(scope, expr);
}

// Insert expr in scope before ref
void insertBefore(Expr* scope, Expr* ref, Expr* expr) {
  scopeInsertBefore::insert(scope, ref, expr);
}

// Return the parent of the active scope
Expr* getParent(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr,
      "Tried to close the active scope, but there isn't one set.");
  assertScope(scope);
  return parentScope::get(scope);
}

// Open a new inner most for loop
ForLoop* openFor(Expr* scope, IterDomain* id) {
  ForLoop* new_scope = nullptr;
  if (id->isThread()) {
    std::stringstream ss;
    ss << id->parallel_method();
    new_scope =
        new ForLoop(new NamedScalar(ss.str(), DataType::Int), id, {}, scope);
  } else {
    new_scope = new ForLoop(new Int(), id, {}, scope);
  }
  if (scope != nullptr)
    pushBack(scope, new_scope);
  return new_scope;
}

// Close the inner most for loop
Expr* closeScope(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr, "Tried to close a scope but got a nullptr.");
  return getParent(scope);
}

// Clear all expressions from the scope
Expr* clearScope(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr, "Tried to clear a scope but got a nullptr.");
  assertScope(scope);
  scopeClearExprs::clear(scope);
  return scope;
}

ForLoop* cloneLoopNest(ForLoop* to_clone, Expr* parent_scope) {
  return CloneLoopNest::getClone(to_clone, parent_scope);
}

void replaceExprsInScope(
    Expr* scope,
    std::unordered_map<Expr*, Expr*> replacement_map) {
  TORCH_INTERNAL_ASSERT(
      replacement_map.find(scope) == replacement_map.end(),
      "Error trying to replace expressions in a scope, scope wants to be replaced entirely.");
  ReplaceExprsInScope::replace(scope, std::move(replacement_map));
}

Expr* firstInnerMostScope(Expr* scope) {
  return FirstInnerMostScope::get(scope);
}

} // namespace scope_utils

namespace ir_utils {

std::vector<Val*> indices(std::vector<ForLoop*> loops) {
  std::vector<Val*> inds(loops.size());
  std::transform(loops.begin(), loops.end(), inds.begin(), [](ForLoop* fl) {
    return fl->index();
  });
  return inds;
}

std::vector<IterDomain*> iterDomains(std::vector<ForLoop*> loops) {
  std::vector<IterDomain*> ids(loops.size());
  std::transform(loops.begin(), loops.end(), ids.begin(), [](ForLoop* fl) {
    return fl->iter_domain();
  });
  return ids;
}

bool isTV(const Val* const val) {
  return val->getValType().value() == ValType::TensorView;
}

// Check if we're a TensorView op that we can generate code for.
bool isTVOp(const Expr* expr) {
  if (expr->nOutputs() == 1 && isTV(expr->output(0)) &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp ||
       expr->getExprType().value() == ExprType::TernaryOp ||
       expr->getExprType().value() == ExprType::ReductionOp ||
       expr->getExprType().value() == ExprType::BroadcastOp))
    return true;
  return false;
}

bool isScalarOp(const Expr* expr) {
  for (auto out : expr->outputs())
    if (!out->isScalar())
      return false;
  return true;
}

void ASSERT_EXPR(Statement* stmt) {
  TORCH_INTERNAL_ASSERT(
      stmt->isExpr(),
      "Tried to generate a kernel but hit a non expression during lowering: ",
      stmt);
}

Expr* asExpr(Statement* stmt) {
  ASSERT_EXPR(stmt);
  return static_cast<Expr*>(stmt);
}

TensorView* asTV(Val* val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return static_cast<TensorView*>(val);
}

bool isScope(const Expr* expr) {
  return expr->getExprType() == ExprType::ForLoop ||
      expr->getExprType() == ExprType::IfThenElse;
}

ForLoop* asForLoop(Statement* stmt) {
  Expr* expr = asExpr(stmt);
  TORCH_INTERNAL_ASSERT(expr->getExprType() == ExprType::ForLoop);
  return static_cast<ForLoop*>(expr);
}

const TensorView* asConstTV(const Val* const val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return static_cast<const TensorView*>(val);
}

bool isUnrolledFor(const Expr* expr) {
  if (expr->getExprType() != ExprType::ForLoop) {
    return false;
  }
  return static_cast<const ForLoop*>(expr)->iter_domain()->parallel_method() ==
      ParallelType::Unroll;
}

} // namespace ir_utils

} // namespace fuser
} // namespace jit
} // namespace torch