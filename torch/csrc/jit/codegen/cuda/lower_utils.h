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

// Return the parent of the active scope
Expr* getParent(Expr* scope);

// Open a new inner most for loop
Expr* openFor(Expr* scope, IterDomain*);

// Close the inner most for loop
Expr* closeScope(Expr* scope);

// Clear all expressions from the scope
Expr* clearScope(Expr* scope);

// Track how far our for loop scope is
unsigned int computeForDepth(Expr* scope);

} // namespace scope_utils
} // namespace fuser
} // namespace jit
} // namespace torch