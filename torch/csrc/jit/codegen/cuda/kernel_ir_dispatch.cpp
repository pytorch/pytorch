#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {
std::vector<Expr*> IrVisitor::handle(const std::vector<Expr*>& exprs) {
  exprs_ = std::vector<Expr*>(exprs);
  for (auto expr : exprs) {
    handle(expr);
  }
  return exprs_;
}

void IrVisitor::handle(ForLoop* fl) {
  for_loops_.push_back(fl);
  scope_.push_back(&fl->body());
  scope_exprs_.push_back(fl);
  auto body_exprs = std::vector<Expr*>(fl->body().exprs());
  for (auto expr : body_exprs) {
    handle(expr);
  }
  scope_exprs_.pop_back();
  scope_.pop_back();
  for_loops_.pop_back();
}

void IrVisitor::handle(IfThenElse* ite) {
  scope_exprs_.push_back(ite);
  scope_.push_back(&ite->thenBody());
  auto then_exprs = std::vector<Expr*>(ite->thenBody().exprs());
  for (auto expr : then_exprs) {
    handle(expr);
  }
  scope_.pop_back();

  scope_.push_back(&ite->elseBody());
  auto else_exprs = std::vector<Expr*>(ite->elseBody().exprs());
  for (auto expr : else_exprs) {
    handle(expr);
  }
  scope_.pop_back();
  scope_exprs_.pop_back();
}

std::vector<const Expr*> ConstIrVisitor::handle(
    const std::vector<const Expr*>& exprs) {
  exprs_ = exprs;
  for (auto expr : exprs) {
    handle(expr);
  }
  return exprs_;
}

void ConstIrVisitor::handle(const ForLoop* fl) {
  for_loops_.push_back(fl);
  scope_.push_back(&fl->body());
  scope_exprs_.push_back(fl);
  auto body_exprs = fl->body().exprs();
  for (auto expr : body_exprs) {
    handle(expr);
  }
  scope_exprs_.pop_back();
  scope_.pop_back();
  for_loops_.pop_back();
}

void ConstIrVisitor::handle(const IfThenElse* ite) {
  scope_exprs_.push_back(ite);
  scope_.push_back(&ite->thenBody());
  auto then_exprs = ite->thenBody().exprs();
  for (auto expr : then_exprs) {
    handle(expr);
  }
  scope_.pop_back();

  scope_.push_back(&ite->elseBody());
  auto else_exprs = ite->elseBody().exprs();
  for (auto expr : else_exprs) {
    handle(expr);
  }
  scope_.pop_back();
  scope_exprs_.pop_back();
}

std::vector<Expr*> ExprMutator::mutate(bool reverse_order) {
  if (insertions_.empty() && replacements_.empty() && removal_.empty()) {
    return exprs_;
  }

  auto run_insertion = [&](MutationInformation info) {
    if (info.scope == nullptr) {
      // If reference is nullptr and there are no expressions, simply insert the
      // expr
      if (exprs_.empty() && info.reference == nullptr) {
        exprs_.push_back(info.new_expr);
        return;
      }
      auto pos_it = std::find(exprs_.begin(), exprs_.end(), info.reference);
      TORCH_INTERNAL_ASSERT(
          pos_it != exprs_.end(),
          "Issue finding reference expression for insertion.");
      if (info.mode == MutationMode::BEFORE) {
        exprs_.insert(pos_it, info.new_expr);
      } else {
        exprs_.insert(pos_it + 1, info.new_expr);
      }
    } else {
      // If reference is nullptr and there are no expressions, simply insert the
      // expr
      if (info.scope->exprs().empty() && info.reference == nullptr) {
        info.scope->push_back(info.new_expr);
        return;
      }
      if (info.mode == MutationMode::BEFORE) {
        info.scope->insert_before(info.reference, info.new_expr);
      } else {
        info.scope->insert_after(info.reference, info.new_expr);
      }
    }
  };

  if (reverse_order) {
    for (auto it = insertions_.rbegin(); it != insertions_.rend(); ++it) {
      run_insertion(*it);
    }
  } else {
    for (auto insertion_info : insertions_) {
      run_insertion(insertion_info);
    }
  }

  for (auto replacement_info : replacements_) {
    if (replacement_info.scope == nullptr) {
      auto pos_it =
          std::find(exprs_.begin(), exprs_.end(), replacement_info.reference);
      TORCH_INTERNAL_ASSERT(
          pos_it != exprs_.end(),
          "Issue finding reference expression for replacement.");
      exprs_.insert(pos_it, replacement_info.new_expr);
      // iterator can be invalidated from insertion
      pos_it =
          std::find(exprs_.begin(), exprs_.end(), replacement_info.reference);
      exprs_.erase(pos_it);
    } else {
      replacement_info.scope->insert_before(
          replacement_info.reference, replacement_info.new_expr);
      replacement_info.scope->erase(replacement_info.reference);
    }
  }

  for (auto removal_info : removal_) {
    if (removal_info.scope == nullptr) {
      auto pos_it =
          std::find(exprs_.begin(), exprs_.end(), removal_info.reference);
      TORCH_INTERNAL_ASSERT(
          pos_it != exprs_.end(), "Issue finding expression to remove.");
      exprs_.erase(pos_it);
    } else {
      TORCH_INTERNAL_ASSERT(
          removal_info.scope->contains(removal_info.reference),
          "Expression to remove is not found in the given scope: ",
          removal_info.reference->toString());
      removal_info.scope->erase(removal_info.reference);
    }
  }

  insertions_.clear();
  replacements_.clear();

  return exprs_;
}

std::vector<Expr*> ExprMutator::traverseAndInsert(
    const std::vector<Expr*>& exprs,
    bool reverse_order) {
  IrVisitor::handle(exprs);
  return mutate(reverse_order);
}

void ExprMutator::registerMutation(
    Expr* reference,
    Expr* new_expr,
    Scope* scope,
    MutationMode mode) {
  MutationInformation mutation;
  mutation.reference = reference;
  mutation.new_expr = new_expr;
  mutation.scope = scope;
  mutation.mode = mode;
  if (mode == MutationMode::BEFORE || mode == MutationMode::AFTER) {
    insertions_.push_back(mutation);
  } else if (mode == MutationMode::REPLACE) {
    replacements_.push_back(mutation);
  } else if (mode == MutationMode::REMOVE) {
    removal_.push_back(mutation);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Invalid mutation type");
  }
}

void ExprMutator::registerInsertBefore(
    Expr* reference,
    Expr* new_expr,
    Scope* scope) {
  registerMutation(reference, new_expr, scope, MutationMode::BEFORE);
}

void ExprMutator::registerInsertAfter(
    Expr* reference,
    Expr* new_expr,
    Scope* scope) {
  registerMutation(reference, new_expr, scope, MutationMode::AFTER);
}

void ExprMutator::registerReplace(
    Expr* reference,
    Expr* new_expr,
    Scope* scope) {
  registerMutation(reference, new_expr, scope, MutationMode::REPLACE);
}

void ExprMutator::registerRemove(Expr* expr_to_remove, Scope* scope) {
  registerMutation(expr_to_remove, nullptr, scope, MutationMode::REMOVE);
}

void ExprMutator::registerInsertBefore(Expr* reference, Expr* new_expr) {
  Scope* scope = scope_.empty() ? nullptr : scope_.back();
  registerInsertBefore(reference, new_expr, scope);
}

void ExprMutator::registerInsertAfter(Expr* reference, Expr* new_expr) {
  Scope* scope = scope_.empty() ? nullptr : scope_.back();
  registerInsertAfter(reference, new_expr, scope);
}

void ExprMutator::registerReplace(Expr* reference, Expr* new_expr) {
  Scope* scope = scope_.empty() ? nullptr : scope_.back();
  registerReplace(reference, new_expr, scope);
}

void ExprMutator::registerRemove(Expr* expr_to_remove) {
  Scope* scope = scope_.empty() ? nullptr : scope_.back();
  registerRemove(expr_to_remove, scope);
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
