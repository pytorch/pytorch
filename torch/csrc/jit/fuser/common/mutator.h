#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>
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
  const Statement* mutate(const Statement* const statement);
  const Statement* mutate(const Float* const f);
  const Statement* mutate(const Int* const i);
  const Statement* mutate(const UnaryOp* const uop);
  const Statement* mutate(const BinaryOp* const bop);

};

}}} // torch::jit::fuser
