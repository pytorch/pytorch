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
struct InlinedCallStack;

/**
 * InlinedCallStack is a node of a trie that represents the callstack.
 * Each node in the trie is a pair [Function, SourceRange].
 * Each IR Node has a callstack field, which is basically a pointer to a trie
 * node. Initially the field is null, but if the node is created during an
 * inlining from a different function it gets filled with the function and
 * source range info. As inlining continues, the trie can grow.
 *
 * Examples to understand how callstack looks like:
 * - Callstack for a node in a function 'foo' that was never inlined is empty
 *   (c10::nullopt).
 * - Callstack for a node in a function 'foo' that was created when 'bar' was
 *   inlined to 'foo' contains only one entry: <'bar'>. Callstack for a node in
 *   'bar' is empty.
 * - When 'baz' is inlined into 'bar' and 'bar' is inlined to 'foo', then the
 *   node in 'foo' originated from 'baz' will have a callstack with two entries:
 *   <'bar', 'baz'>. The last element in this vector will correspond to the leaf
 *   function.
 */
using InlinedCallStackPtr = c10::intrusive_ptr<InlinedCallStack>;
using InlinedCallStackEntry = std::pair<Function*, SourceRange>;
struct InlinedCallStackHash {
  std::size_t operator()(const InlinedCallStackEntry& cs) const {
    return std::hash<void*>()(cs.first) ^
        std::hash<void*>()(&*cs.second.source()) ^
        std::hash<size_t>()(cs.second.start()) ^
        std::hash<size_t>()(cs.second.end());
  }
};

struct TORCH_API InlinedCallStack : public c10::intrusive_ptr_target {
 private:
  c10::optional<InlinedCallStackPtr> caller_;

  std::unordered_map<
      InlinedCallStackEntry,
      InlinedCallStackPtr,
      InlinedCallStackHash>
      callees_;
  Function* fn_;
  SourceRange source_range_;
  InlinedCallStackPtr intrusive_from_this();

 public:
  // Constructor for the root callstack node.
  InlinedCallStack(Function* fn, SourceRange source_range);

  // Constructor for an inner callstack node.
  InlinedCallStack(
      InlinedCallStackPtr caller,
      Function* fn,
      SourceRange source_range);

  // Return callstack for the caller.
  // Essentially, move one level up in the trie.
  c10::optional<InlinedCallStackPtr> caller() const;

  // Insert new callee to the callstack.
  // Essentially, find existing or insert new node into the trie.
  InlinedCallStackPtr insertCallStackEntry(
      Function* fn,
      const SourceRange& source_range);

  // Flatten callstack to a vector of [Function, SourceRange] pairs.
  std::vector<InlinedCallStackEntry> vec();
};

} // namespace jit
} // namespace torch
