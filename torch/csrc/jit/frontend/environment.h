#pragma once

#include <torch/csrc/jit/frontend/ir_emitter_utils.h>

#include <ATen/core/jit_type.h>

#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/frontend/tree_views.h>

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using TypeTable = std::unordered_map<std::string, TypePtr>;

// Auxiliary data structure for desugaring variable binding into our always
// explicitly scoped language as we descend down nested control structures in
// the frontend (which themselves don't introduce scopes)
//
// The Environment keeps track of two tables, one for values which are not first
// class and a type table for values which are. When a first class value
// is set in the environment, we emit a prim::Store which sets the
// name of the variable to appropriate type, and when a first-class value is
// referenced we emit a prim::Load that generates a value of the appropriate
// type.
//
// a = 1
// print(a)
// becomes:
// = prim::Store[name="a"](%a.1)
// %a : int = prim::Load[name="a"]()
// prim::Print(%a)
struct Environment {
  Environment(
      Function& method,
      ResolverPtr resolver,
      Block* b,
      std::shared_ptr<Environment> next = nullptr);

  std::shared_ptr<Environment> getNext() {
    return next;
  }

  // set type error in the lowest environment. if the variable is used after an
  // error has been set, then we will use the more informative error message
  void setVariableTypeError(
      const std::string& name,
      std::function<std::string()> msg);

  // see if type error has been set for a variable
  c10::optional<std::string> findVariableTypeError(const std::string& name);

  SugaredValuePtr insertLoad(const std::string& name, const TypePtr& type);

  // note: type is not always the same as v->type(), e.g.
  // type: Optional[Tensor]
  // v->type(): Tensor
  void insertStore(
      const std::string& name,
      const SourceRange& loc,
      Value* v,
      TypePtr type);

  SugaredValuePtr findInThisFrame(const std::string& name);

  SugaredValuePtr findInParentFrame(const std::string& name);

  void setType(const std::string& name, TypePtr type);

  SugaredValuePtr findInAnyFrame(const std::string& name);

  Block* block();

  void setVar(const SourceRange& loc, const std::string& name, Value* value);

  void setSugaredVar(
      const SourceRange& loc,
      const std::string& name,
      SugaredValuePtr value,
      TypePtr annotated_type);

  SugaredValuePtr getSugaredVar(const Ident& ident, bool required = true);

  Value* getVar(const Ident& ident);

  void throwVarNotFoundError(
      const std::string& ident,
      const SourceRange& range);

  SugaredValuePtr getSugaredVar(
      const std::string& ident,
      const SourceRange& range,
      bool required = true);

  Value* getVar(const std::string& ident, const SourceRange& range);

  void removeVar(const Ident& ident, bool check_if_removed = false);

  std::vector<std::string> definedVariables();

 protected:
  Function& method;
  ResolverPtr resolver;
  Block* b;
  std::shared_ptr<Environment> next;
  std::unordered_map<std::string, std::function<std::string()>> error_messages;
  friend struct to_ir;

 private:
  TypeTable type_table;
  ValueTable value_table;
};

} // namespace jit
} // namespace torch
