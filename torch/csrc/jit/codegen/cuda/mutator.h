#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

/*
 * Mutators are the mechanism used to modify IR nodes. Since most nodes are
 * immutable or at least partially immutable changeing them can require creating
 * a new node. Base mutator at the moment is a dumb sample mutator that takes
 * any float of value 1.0 and converts it to 0.0; It is currently used as a
 * dummy example, however, we should make it a simple instantiation of all the
 * mutate functions on all node types so that people can inhereit it, and only
 * specialize those nodes which they want to have a particular transformation.
 */


// Search through "within" and replace all instances of "instance" with the
// value "with".
struct TORCH_API ReplaceAll : public OptOutMutator {
 private:
  Statement* mutate(Val*);
  Statement* mutate(Expr* expr) {
    return OptOutMutator::mutate(expr);
  }

  ReplaceAll(Val* _instance, Val* _with) : instance_(_instance), with_(_with) {}

  Val* instance_;
  Val* with_;

 public:
  // Traverses Statement, and replaces all instances of _instance with _with.
  TORCH_API static void instancesWithin(
      Val* _instance,
      Val* _with,
      Expr* _within) {
    if (_within == nullptr)
      return;
    FusionGuard fg(_within->fusion());
    ReplaceAll ra(_instance, _with);
    ra.mutate(_within);
  }

  TORCH_API static void instancesOf(Val* instance, Val* with);
};

} // namespace fuser
} // namespace jit
} // namespace torch
