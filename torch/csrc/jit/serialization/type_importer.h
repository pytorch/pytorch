#pragma once

#include "ATen/core/jit_type.h"
#include "torch/csrc/jit/api/compilation_unit.h"

namespace torch {
namespace jit {

  /**
   * [remapping followups]
   * Our types have different ways of specifying equivalence:
   * - Tuples and NamedTuples are structurally typed
   * - Functions types are the same if they store the same function ptr
   * - Primitive types have global singletons representing them.
   * - ClassTypes are equal only to themselves
   * - etc.
   *
   * Our current equality functions aren't quite right, so we should fix them
   * and just use TypeEquals here.
   *
   * We need to fix this for this pass to be totally correct. The known places
   * that should be fixed up are annotated inline.
   */
class TypeImporter {
 public:
  explicit TypeImporter(std::shared_ptr<CompilationUnit> cu)
      : cu_(std::move(cu)) {}

  /**
   * Given `origType`, clone the type into the CompilationUnit `cu_` and return
   * the the new type. Note that this function is idempotent because we cache
   * the mapping of old type -> imported type.
   *
   * The cloning only happens for types that will be serialized (e.g.
   * NamedTypes). Any types that `origType` references will be recursively
   * cloned to `cu_`. Any naming collisions will be resolved by mangling.
   *
   * At the end, we expect to have all types in the hierarchy live in `cu_`.
   */
  TypePtr import(const TypePtr& origType);

 private:
  std::shared_ptr<CompilationUnit> cu_;
  std::unordered_map<TypePtr, TypePtr> remappedTypes_;

  TypePtr remap(const InterfaceTypePtr& origType);
  TypePtr remap(const FunctionTypePtr& origType);
  TypePtr remap(const ClassTypePtr& origType);
};
} // namespace jit
} // namespace torch
