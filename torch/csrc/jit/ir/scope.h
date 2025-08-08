#pragma once
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <optional>
#include <unordered_map>

namespace torch::jit {
struct ModuleInstanceInfo;
constexpr size_t kModuleInstanceInfo = 2;

namespace utils {
std::string get_module_info(const ModuleInstanceInfo& module_instance_info);
} // namespace utils

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
 * ModuleInstanceInfo is a structure to include the module type and instance
 * name. It also provide public methods to get the pointer to module type and
 * instance name.
 *
 * This structure is mainly used as a private member in InlinedCallStack, such
 * that one can follow the callstack to find the relevant module hierarchy.
 */
struct ModuleInstanceInfo {
 private:
  c10::ClassTypePtr module_type_{nullptr};
  std::string instance_name_;

 public:
  ModuleInstanceInfo() = default;
  ModuleInstanceInfo(c10::ClassTypePtr module_type, std::string instance_name);
  c10::ClassTypePtr class_type() {
    return module_type_;
  }
  c10::ClassTypePtr class_type() const {
    return module_type_;
  }
  std::string instance_name() const {
    return instance_name_;
  }

  bool operator==(const ModuleInstanceInfo& rhs) const {
    return (class_type() == rhs.class_type()) &&
        (instance_name() == rhs.instance_name());
  }
};

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
using InlinedCallStackEntry =
    std::tuple<Function*, SourceRange, std::optional<ModuleInstanceInfo>>;

struct TORCH_API InlinedCallStack : public c10::intrusive_ptr_target {
 private:
  std::optional<InlinedCallStackPtr> callee_;
  Function* fn_;
  // Reason for fn_name_ even though we have fn_
  // Serialized callstack is used in circustmances where InlinedCallstack
  // cannot be constructed during runtime, e.g. mobile runtime or
  // delegated backends.
  // Since in those cases we do not have Function* we store function name
  // fn_name does not give you access to the same information that Function*
  // does, however in mobile/delegated backend runtime we use InlindedCallStack
  // for exception stack and for that purpose fn_name_ suffices.
  const std::string fn_name_;
  SourceRange source_range_;
  InlinedCallStackPtr intrusive_from_this();
  std::optional<ModuleInstanceInfo> module_instance_info_;

 public:
  // Constructor for a leaf callstack node.
  InlinedCallStack(Function* fn, SourceRange source_range);

  // Constructor for a leaf callstack node.
  InlinedCallStack(
      Function* fn,
      SourceRange source_range,
      std::optional<ModuleInstanceInfo> module_instance_info);

  // Constructor for a leaf callstack node.
  InlinedCallStack(
      Function* fn,
      SourceRange source_range,
      std::optional<ModuleInstanceInfo> module_instance_info,
      std::string& function_name);

  // Constructor for an inner callstack node.
  InlinedCallStack(
      InlinedCallStackPtr callee,
      Function* fn,
      SourceRange source_range);

  InlinedCallStack(
      InlinedCallStackPtr callee,
      Function* fn,
      SourceRange source_range,
      std::optional<ModuleInstanceInfo> module_instance_info);

  InlinedCallStack(
      InlinedCallStackPtr callee,
      Function* fn,
      SourceRange source_range,
      std::optional<ModuleInstanceInfo> module_instance_info,
      std::string& function_name);

  // Return next element in the callstack list.
  std::optional<InlinedCallStackPtr> callee() const;

  // Return module instance associated with the current element.
  std::optional<ModuleInstanceInfo> module_instance() const;

  // Returns the source range of the node
  SourceRange source_range() const;

  Function* function() const;

  const std::string& function_name() const;

  // Return callstack as a vector of [Function, SourceRange] pairs.
  std::vector<InlinedCallStackEntry> vec();

  void setCallee(std::optional<InlinedCallStackPtr>);

  bool operator==(const InlinedCallStack& rhs) const {
    // No need to compare fn_, since source_range equivalence check
    // should suffice.
    return (module_instance().has_value() ==
            rhs.module_instance().has_value()) &&
        (module_instance().has_value() &&
         module_instance().value() == rhs.module_instance().value()) &&
        callee() == rhs.callee() && source_range() == rhs.source_range();
  }

  bool operator!=(const InlinedCallStack& rhs) const {
    return !(*this == rhs);
  }
};

// {source range, node name, InlinedCallStack}
// We store node name because same debug info will be used for
// profiling as well, so we need to know op names as well.
using DebugInfoTuple =
    std::tuple<SourceRange, std::string, InlinedCallStackPtr>;
constexpr size_t kDebugInfoTupleSourceRangeIndex{0};
constexpr size_t kDebugInfoTupleNodeNameIndex{1};
constexpr size_t kDebugInfoTupleInlinedCSIndex{2};
} // namespace torch::jit
