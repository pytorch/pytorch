#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

namespace torch {
namespace jit {
namespace fuser {

/*
 * TransformIter iterates on the split/merge/reorder graph of TensorDomain
 *
 * Running backward will execute these Exprs in reverse order. If you record
 * these events (generate_record=true) you can then replay them on another
 * tensor domain.
 */
struct TORCH_CUDA_API TransformIter : public IterVisitor {
 protected:
  virtual void replayBackward(Split* expr);
  virtual void replayBackward(Merge* expr);
  virtual void replayBackward(Reorder* expr);

  // dispatch
  void replayBackward(Expr* expr);

  // Iterates td's history starting with td, then origin(td), origin(origin(td))
  // etc. Returns root TensorDomain once it iterates through history. If
  // generate_record=true It will record the history of td in record. Record is
  // order operations root->td.
  TensorDomain* runBackward(TensorDomain* td, bool generate_record);

  virtual TensorDomain* replay(Split* expr, TensorDomain* tv);
  virtual TensorDomain* replay(Merge* expr, TensorDomain* tv);
  virtual TensorDomain* replay(Reorder* expr, TensorDomain* tv);

  // dispatch
  TensorDomain* replay(Expr* expr, TensorDomain* tv);

  // Runs through operations recorded in record from root-> present
  TensorDomain* runReplay(TensorDomain* tv);

  // Forward record from root, to replay_ref/ref_root
  std::vector<Expr*> record;

 public:
  static TensorDomain* getRoot(TensorDomain* td) {
    TransformIter ti;
    return ti.runBackward(td, false);
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
