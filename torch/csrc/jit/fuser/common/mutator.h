#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

/*
 * Mutators are the mechanism used to modify IR nodes. Since all nodes are
 * immutable the only way to change them is to create new ones. Base mutator at
 * the moment is a dumb sample mutator that takes any float of value 1.0 and
 * converts it to 0.0; It is currently used as a dummy example, however, we
 * should make it a simple instantiation of all the mutate functions on all node
 * types so that people can inhereit it, and only specialize those nodes which
 * they want to have a particular transformation.
 */

struct TORCH_API BaseMutator {
  void mutate(Fusion* fusion);
  virtual Statement* mutate(Statement*);

  virtual Statement* mutate(Val*);
  virtual Statement* mutate(Expr*);

  virtual Statement* mutate(UnaryOp*);
  virtual Statement* mutate(BinaryOp*);

  virtual Statement* mutate(Split*);
  virtual Statement* mutate(Merge*);
  virtual Statement* mutate(Reorder*);

  virtual Statement* mutate(TensorDomain*);
  virtual Statement* mutate(TensorView*);
  virtual Statement* mutate(IterDomain*);
  // Tensor is reference to JIT tensor, should never be mutated.
  virtual Statement* mutate(Tensor*) final;

  virtual Statement* mutate(Float*);
  virtual Statement* mutate(Int*);
};

struct TORCH_API ReplaceAll : public BaseMutator {
 private:
  Statement* mutate(Val*);
  Statement* mutate(Expr* expr) {
    return BaseMutator::mutate(expr);
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
    ReplaceAll ra(_instance, _with);
    ra.mutate(_within);
  }

  TORCH_API static void instancesOf(Val* instance, Val* with);
};

} // namespace fuser
} // namespace jit
} // namespace torch
