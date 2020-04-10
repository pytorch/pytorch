#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

// Provides utilities for dealing with nested ForLoop and IfThenElse scopes

namespace torch {
namespace jit {
namespace fuser {

namespace scope_utils {

// Grab the index variables of the active loop nest
std::vector<Val*> getLoopIndices(Expr* scope);

// Grab the iterDomains of the active loops
std::vector<IterDomain*> getLoopIterDomains(Expr* scope);

// Track how far our for loop scope is
unsigned int computeForDepth(Expr* scope);

// Push back an expr to scope
void pushBack(Expr* scope, Expr* expr);

// Insert expr in scope before ref
void insertBefore(Expr* scope, Expr* ref, Expr* expr);

// Return the parent of the active scope
Expr* getParent(Expr* scope);

// Open a new inner most for loop
ForLoop* openFor(Expr* scope, IterDomain*);

// Close the inner most for loop
Expr* closeScope(Expr* scope);

// Clear all expressions from the scope
Expr* clearScope(Expr* scope);

// Provide a new for loop matching the one provided
ForLoop* cloneLoopNest(ForLoop* to_clone, Expr* parent_scope);

// Run through a scope and replace expressions inside with replacement_map
void replaceExprsInScope(
    Expr* scope,
    std::unordered_map<Expr*, Expr*> replacement_map);

Expr* firstInnerMostScope(Expr* scope);

} // namespace scope_utils

namespace ir_utils {

bool isTV(const Val* const);

bool isTVOp(const Expr*);

void ASSERT_EXPR(Statement*);

bool isScope(const Expr*);

Expr* asExpr(Statement*);

TensorView* asTV(Val*);

ForLoop* asForLoop(Statement*);

const TensorView* asConstTV(const Val* const);

bool isUnrolledFor(const Expr*);

} // namespace ir_utils
} // namespace fuser
} // namespace jit
} // namespace torch
