#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/function.h>

namespace torch {
namespace jit {

/**
 * class CompilationArena
 *
 * Every Arena is owned by a CompilationUnit, which is the publically-visible
 * way to access classes and functions.
 *
 * So this class is pretty dumb; it's just supposed to keep unique pointers to
 * the classes/functions so that they don't get freed. We should never be
 * actually asking an Arena what it owns.
 */
class CompilationArena {
 public:
  CompilationArena() : blob_(std::make_shared<OwnerBlob>()){};

  /**
   * Union this Arena with another one. After this now both Arenas own the same
   * stuff, and as long as one of the arenas is alive, that stuff will stay
   * alive.
   */
  void unionWith(CompilationArena* other);

  const std::vector<c10::ClassTypePtr>& getClasses() const;
  const std::vector<std::unique_ptr<Function>>& getFunctions() const;
  std::vector<c10::ClassTypePtr>& getClasses();
  std::vector<std::unique_ptr<Function>>& getFunctions();

 private:
  // The actual owner of the classes/functions. We have a layer of indirection
  // here so that we can union Arenas transparently.
  struct OwnerBlob {
    // TODO: these are shared ptrs today, but eventually arenas will have unique
    // ptrs when they become the sole owners of classes/functions
    std::vector<c10::ClassTypePtr> classes_;
    std::vector<std::unique_ptr<Function>> functions_;
    size_t size() const {
      return classes_.size() + functions_.size();
    }
  };
  std::shared_ptr<OwnerBlob> blob_;
  std::weak_ptr<script::CompilationUnit> cu_;
};
} // namespace jit
} // namespace torch