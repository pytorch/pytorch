#pragma once

#include <dispatch.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class Expr;

namespace kir {
class Predicate;
class TensorIndex;
class ForLoop;
class IfThenElse;
class Scope;

// Base visitor class that visits all nodes in provided vector<Expr*>.
//
// Includes visiting through scopes like IfThenElse and ForLoop, and tracks
// them in scopes_ and for_loops_.
//
// Makes a copy of exprs at exprs_ which could be used to modify and return.
//
// When traversing through ITE/FLs it will use a copy
// of the provided expressions to make it safe to insert/delete nodes.
//
// Provides a simple base class to inherit from for typical lowering passes on
// Expr list
class TORCH_CUDA_CU_API IrVisitor : public OptOutDispatch {
 public:
  std::vector<Expr*> handle(const std::vector<Expr*>& expr);

 protected:
  using OptOutDispatch::handle;

  virtual void handle(ForLoop*) override;
  virtual void handle(IfThenElse*) override;

 protected:
  std::vector<ForLoop*> for_loops_;
  std::vector<Scope*> scope_;
  std::vector<Expr*> scope_exprs_;
  std::vector<Expr*> exprs_;
};

// Const version of IrVisitor
class TORCH_CUDA_CU_API ConstIrVisitor : public OptOutConstDispatch {
 public:
  std::vector<const Expr*> handle(const std::vector<const Expr*>& expr);

 protected:
  using OptOutConstDispatch::handle;

  virtual void handle(const ForLoop*) override;
  virtual void handle(const IfThenElse*) override;

 protected:
  std::vector<const ForLoop*> for_loops_;
  std::vector<const Scope*> scope_;
  std::vector<const Expr*> scope_exprs_;
  std::vector<const Expr*> exprs_;
};

// Base Expr Mutator class that visits all nodes with IrVisitor, and then
// inserts new expressions, replaces expressions based on insertion/replace
// maps provided or removes existing expressions. These replacement
// maps are expected to accumulate during an initial traversal, then
// runs an insertion based on them after the overloaded traversal.
//
// Order of mutations may be important, mutations are ordered according to the
// following rules:
//   Before/After insertions are ordered as registered when reverse_order ==
//   false,
//
//   Before/After insertions are in reverse order as registered when
//   reverse_order == true,
//
//   Before/After insertions are done before Expr replacements, so reference for
//   insertions must be on pre-replaced Exprs
//
//   Removal of expressions is done after replacements.
//
// To place in a scope that is empty, simply provide a nullptr reference
// Since insertions are done in order, it's possible to insert an expression in
// an empty scope, and then use that inserted scope as a reference for
// subsequent mutations.
class ExprMutator : public IrVisitor {
 protected:
  std::vector<Expr*> traverseAndInsert(
      const std::vector<Expr*>& expr,
      bool reverse_order = false);

  std::vector<Expr*> mutate(bool reverse_order = false);

  using IrVisitor::handle;
  // Registration function which *don't* need to be called "in place" during
  // visiting.
  void registerInsertBefore(Expr* reference, Expr* new_expr, Scope* scope);
  void registerInsertAfter(Expr* reference, Expr* new_expr, Scope* scope);
  void registerReplace(Expr* reference, Expr* new_expr, Scope* scope);
  void registerRemove(Expr* expr_to_remove, Scope* scope);

  // Registration function which need to be called "in place" during visiting.
  // I.E.
  // if you want to insert before/after or replace an Expr, you must register
  // when in handle(Expr*) of that expr.
  void registerInsertBefore(Expr* reference, Expr* new_expr);
  void registerInsertAfter(Expr* reference, Expr* new_expr);
  void registerReplace(Expr* reference, Expr* new_expr);
  void registerRemove(Expr* expr_to_remove);

 private:
  enum class MutationMode { BEFORE, AFTER, REPLACE, REMOVE };

  void registerMutation(
      Expr* ref,
      Expr* new_expr,
      Scope* scope,
      MutationMode mode);

  struct MutationInformation {
    Expr* reference = nullptr;
    Expr* new_expr = nullptr;
    Scope* scope = nullptr;
    MutationMode mode = MutationMode::BEFORE;
  };

  // Track insertions as they're registered
  std::vector<MutationInformation> insertions_;

  // Track replacements as they're registered
  std::vector<MutationInformation> replacements_;

  // Track removal as they're registered
  std::vector<MutationInformation> removal_;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
