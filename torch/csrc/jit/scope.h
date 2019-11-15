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
 * InlinedCallStack is an element in a list representing callstack of functions
 * that have been inlined.
 *
 * Each such element holds info about the current callsite (Function and
 * SourceRange) and a pointer to the next element in the list. The last element
 * in the list represents the innermost function that was inlined.
 *
 * For instance, if a node has a callstack
 *    [foo, source_range1] -> [bar, source_range2]
 * it means that this node was originally from function 'bar' that was called
 * at 'source_range2' in function 'foo' that was called in the current function
 * at 'source_range1'.
 *
 * If a node did not come from any inlined function, its callstack will be
 * empty.
 *
 * The callstack lists only grow, we never remove elements from them, which
 * allows us to reuse same elements in different lists. For instance, if we
 * inline function 'bar' to 'foo' and then inline 'foo' to two functions 'ham'
 * and 'baz', the callstacks would look like:
 *
 *  [baz, source_range3]  --
 *                           \
 *                             --> [foo, source_range1] -> [bar, source_range2]
 *                           /
 *  [ham, source_range4]  --
 */
using InlinedCallStackPtr = c10::intrusive_ptr<InlinedCallStack>;
using InlinedCallStackEntry = std::pair<Function*, SourceRange>;

struct TORCH_API InlinedCallStack : public c10::intrusive_ptr_target {
 private:
  c10::optional<InlinedCallStackPtr> callee_;
  Function* fn_;
  SourceRange source_range_;
  InlinedCallStackPtr intrusive_from_this();

 public:
  // Constructor for a leaf callstack node.
  InlinedCallStack(Function* fn, SourceRange source_range);

  // Constructor for an inner callstack node.
  InlinedCallStack(
      InlinedCallStackPtr callee,
      Function* fn,
      SourceRange source_range);

  // Return next element in the callstack list.
  c10::optional<InlinedCallStackPtr> callee() const;

  // Return callstack as a vector of [Function, SourceRange] pairs.
  std::vector<InlinedCallStackEntry> vec();
};

} // namespace jit
} // namespace torch
