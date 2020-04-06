#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>

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

  virtual TensorView* replay(Split* expr, TensorView* tv);
  virtual TensorView* replay(Merge* expr, TensorView* tv);
  virtual TensorView* replay(Reorder* expr, TensorView* tv);

  // dispatch
  TensorView* replay(Expr* expr, TensorView* tv);

  // Runs through operations recorded in record from root-> present
  TensorView* runReplay(TensorView* tv);

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
