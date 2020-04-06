#pragma once

#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {

namespace scope_utils {

namespace {

struct forLoopIndices : private OptInDispatch {
 private:
  std::vector<Val*> inds_;
  void handle(ForLoop* fl) final {
    inds_.insert(inds_.begin(), fl->index());
  }

  void handle(IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static std::vector<Val*> get(Expr* scope) {
    forLoopIndices fli;
    Expr* it = scope;
    while (it != nullptr) {
      fli.handle(it);
      it = getParent(it);
    }
    return fli.inds_;
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
      it = getParent(it);
    }
    return flc.count_;
  }
};

struct forLoopIDs : private OptInDispatch {
 private:
  std::vector<IterDomain*> IDs_;
  void handle(ForLoop* fl) final {
    IDs_.insert(IDs_.begin(), fl->iter_domain());
  }

  void handle(IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static std::vector<IterDomain*> get(Expr* scope) {
    forLoopIDs fli;
    Expr* it = scope;
    while (it != nullptr) {
      fli.handle(it);
      it = getParent(it);
    }
    return fli.IDs_;
  }
};

struct scopePushBack : private OptInDispatch {
 private:
  Expr* _expr = nullptr;
  void handle(ForLoop* fl) final {
    fl->body().push_back(_expr);
  }

  void handle(IfThenElse* ite) final {
    ite->body().push_back(_expr);
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static void push(Expr* scope, Expr* expr) {
    scopePushBack pb;
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    pb._expr = expr;
    pb.handle(scope);
  }
};

struct scopeClearExprs : private OptInDispatch {
 private:
  Expr* _expr = nullptr;
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

} // namespace

// Grab the index variables of the active loop nest
std::vector<Val*> getLoopIndices(Expr* scope) {
  if (scope == nullptr)
    return std::vector<Val*>();
  assertScope(scope);
  return forLoopIndices::get(scope);
}

// Grab the iterDomains of the active loops
std::vector<IterDomain*> getLoopIterDomains(Expr* scope) {
  if (scope == nullptr)
    return std::vector<IterDomain*>();
  assertScope(scope);
  return forLoopIDs::get(scope);
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

// Return the parent of the active scope
Expr* getParent(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr,
      "Tried to close the active scope, but there isn't one set.");
  assertScope(scope);
  return parentScope::get(scope);
}

// Open a new inner most for loop
Expr* openFor(Expr* scope, IterDomain* id) {
  ForLoop* new_scope = nullptr;
  if (id->isThread()) {
    new_scope = new ForLoop(
        new NamedScalar(stringify(id->parallel_method()), DataType::Int),
        id,
        {},
        scope);
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

} // namespace scope_utils

namespace ir_utils {

bool isTV(const Val* const val) {
  return val->getValType().value() == ValType::TensorView;
}

// Check if we're a TensorView op that we can generate code for.
bool isTVOp(const Expr* expr) {
  if (expr->nOutputs() == 1 && isTV(expr->output(0)) &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp))
    return true;
  return false;
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