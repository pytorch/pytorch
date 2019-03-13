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
  ScopePtr intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return c10::intrusive_ptr<Scope>::reclaim(this);
  }

 public:
  Scope() {
    name_ = Symbol::scope("");
  }
  Scope(ScopePtr parent, Symbol name) {
    name_ = name;
    parent_ = std::move(parent);
  }
  ScopePtr push(Symbol name);

  ScopePtr parent() {
    if (!parent_) {
      throw std::runtime_error("Cannot get parent from Scope with no parent");
    }
    return parent_;
  }
  bool isRoot() const {
    return !parent_;
  }
  bool isBlank() const {
    static const Symbol blank = Symbol::scope("");
    return isRoot() && name() == blank;
  }

  ScopePtr getRoot();

  size_t getDepth();

  Symbol name() const {
    return name_;
  }

  std::string namesFromRoot(const std::string& separator = "/") const;
};

} // namespace jit
} // namespace torch
