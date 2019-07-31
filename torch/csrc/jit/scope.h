#pragma once
#include <ATen/core/interned_strings.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/source_range.h>
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

struct Function;
struct CallStack;

/**
 * CallStack is a node of a trie that represents the callstack.
 * Each node in the trie is a pair [Function, SourceRange].
 * Each IR Node has a callstack field, which is basically a pointer to a trie node.
 * Initially the field is null, but if the node is created during an inlining
 * from a different function it gets filled with the function and source range
 * info. As inlining continues, the trie can grow.
 */
using CallStackPtr = c10::intrusive_ptr<CallStack>;
using CallStackEntry = std::pair<Function*, SourceRange>;
struct CallStackHash {
  std::size_t operator()(const CallStackEntry& cs) const {
    return std::hash<void*>()(cs.first) ^
        std::hash<void*>()(&*cs.second.source()) ^
        std::hash<size_t>()(cs.second.start()) ^
        std::hash<size_t>()(cs.second.end());
  }
};

struct TORCH_API CallStack : public c10::intrusive_ptr_target {
 private:
  c10::optional<CallStackPtr> caller_;

  std::unordered_map<CallStackEntry, CallStackPtr, CallStackHash> callees_;
  Function* fn_;
  SourceRange source_range_;
  CallStackPtr intrusive_from_this();

 public:
  // Constructor for the root callstack node.
  CallStack(Function* fn, SourceRange source_range);

  // Constructor for an inner callstack node.
  CallStack(CallStackPtr caller, Function* fn, SourceRange source_range);

  // Return callstack for the caller.
  // Essentially, move one level up in the trie.
  c10::optional<CallStackPtr> caller() const;

  // Insert new callee to the callstack.
  // Essentially, find existing or insert new node into the trie.
  CallStackPtr insertCallStackEntry(Function* fn, SourceRange source_range);

  // Flatten callstack to a vector of [Function, SourceRange] pairs.
  std::vector<CallStackEntry> asVector();
};

} // namespace jit
} // namespace torch
