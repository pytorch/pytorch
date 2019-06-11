#pragma once
#include <ATen/core/interned_strings.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {

// Scope is a node of a trie that represents the tree of nested scopes.
// Individual scopes are pushed and popped from Graph, which holds a
// pointer to the current scope. Each Node in Graph holds a pointer
// to the scope that was current when the node was created.
// The trie never needs to shrink, it only grows until it is disposed
// of when Graph is deallocated. Hence, pointers to scopes held by nodes
// will always be valid as long as Graph is alive.
struct Scope;
using ScopePtr = c10::intrusive_ptr<Scope>;
using c10::Symbol;

struct TORCH_API Scope : public c10::intrusive_ptr_target {
 private:
  ScopePtr parent_;
  Symbol name_;
  ScopePtr intrusive_from_this();

 public:
  Scope();

  Scope(ScopePtr parent, Symbol name);

  ScopePtr push(Symbol name);

  ScopePtr parent();

  bool isRoot() const;

  bool isBlank() const;

  ScopePtr getRoot();

  size_t getDepth();

  Symbol name() const;

  std::string namesFromRoot(const std::string& separator = "/") const;
};

} // namespace jit
} // namespace torch
