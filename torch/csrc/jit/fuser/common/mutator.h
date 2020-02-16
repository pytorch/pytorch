#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

/*
 * Mutators are the mechanism used to modify IR nodes. Since all nodes are immutable the only way to
 * change them is to create new ones. Base mutator at the moment is a dumb sample mutator that takes
 * any float of value 1.0 and converts it to 0.0; It is currently used as a dummy example, however,
 * we should make it a simple instantiation of all the mutate functions on all node types so that
 * people can inhereit it, and only specialize those nodes which they want to have a particular
 * transformation.
 */

struct TORCH_API BaseMutator {

  void mutate(Fusion* fusion);
  virtual const Statement* mutate(const Statement* const);

  virtual const Statement* mutate(const Val* const);
  virtual const Statement* mutate(const Expr* const);

  virtual const Statement* mutate(const UnaryOp* const);
  virtual const Statement* mutate(const BinaryOp* const);

  virtual const Statement* mutate(const Split* const);
  virtual const Statement* mutate(const Merge* const);
  virtual const Statement* mutate(const Reorder* const);

  virtual const Statement* mutate(const TensorDomain* const);
  virtual const Statement* mutate(const TensorView* const);
  virtual const Statement* mutate(const IterDomain* const);
  virtual const Statement* mutate(const Tensor* const) final; //I believe tensor should never be mutated.

  virtual const Statement* mutate(const Float* const);
  virtual const Statement* mutate(const Int* const);

};

struct TORCH_API ReplaceAll : public BaseMutator{

  const Statement* mutate(const Val* const);

  TORCH_API static void instancesOf(const Val* const instance, const Val* const with);

private:
  ReplaceAll(const Val* _instance, const Val* const _with):instance_(_instance), with_(_with){}

  const Val* instance_;
  const Val* with_;

};

}}} // torch::jit::fuser
