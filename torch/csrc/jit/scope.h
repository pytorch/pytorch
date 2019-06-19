#pragma once
#include <ATen/core/interned_strings.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <unordered_map>

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

/**
 * CallStack is a node of a trie that represents the callstack.
 * Initially callstacks are always only one level deep: when we create a graph
 * for a function FOO every node in this graph and the graph itself has a
 * callstack [FOO]. However, when we inline FOO to another function BAR, nodes
 * inserted to the caller (BAR) get callstacks of [BAR, FOO].
 */
struct Function;
struct CallStack;
using CallStackPtr = c10::intrusive_ptr<CallStack>;
struct TORCH_API CallStack : public c10::intrusive_ptr_target {
 private:
  c10::optional<CallStackPtr> caller_;
  std::unordered_map<Function*, CallStackPtr> callees_;
  Function* fn_;
  CallStackPtr intrusive_from_this();

 public:
  // Constructor for the root callstack node.
  CallStack(Function* fn);

  // Constructor for an inner callstack node.
  CallStack(CallStackPtr caller, Function* fn);

  // Return callstack for the caller.
  // Essentially, move one level up in the trie.
  c10::optional<CallStackPtr> caller() const;

  // Insert new callee to the callstack.
  // Essentially, find existing or insert new node into the trie.
  CallStackPtr insertCallee(Function* fn);

  // Return callstack as a vector.
  std::vector<Function*> asVector();
};

} // namespace jit
} // namespace torch
