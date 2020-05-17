#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <vector>

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
  virtual TensorDomain* replayBackward(Split*, TensorDomain*);
  virtual TensorDomain* replayBackward(Merge*, TensorDomain*);
  virtual TensorDomain* replayBackward(Reorder*, TensorDomain*);

  // dispatch
  TensorDomain* replayBackward(Expr*, TensorDomain*);

  // Iterates td's history starting with td, then origin(td), origin(origin(td))
  // etc. Returns root TensorDomain once it iterates through history. If
  // generate_record=true It will record the history of td in record. Record is
  // order operations root->td.
  virtual TensorDomain* runBackward(TensorDomain*);

  virtual TensorDomain* replay(Split*, TensorDomain*);
  virtual TensorDomain* replay(Merge*, TensorDomain*);
  virtual TensorDomain* replay(Reorder*, TensorDomain*);

  // dispatch
  virtual TensorDomain* replay(Expr*, TensorDomain*);

  // Runs through operations in history and applies them to TD, runs exprs from
  // begining to end
  virtual TensorDomain* runReplay(TensorDomain*, const std::vector<Expr*>&);

 public:
  // Returns transformation exprs in forward order
  static std::vector<Expr*> getHistory(TensorDomain*);

  // TODO: make td const
  static TensorDomain* getRoot(TensorDomain* td) {
    TransformIter ti;
    return ti.runBackward(td);
  }

  // Takes influence vector of bools, tracks them back to propagate true to root
  // axes that were modified into td axes matching marked influence vector
  static std::vector<bool> getRootInfluence(
      TensorDomain* td,
      const std::vector<bool>& influence);

  static std::vector<bool> replayBackwardInfluence(
      const std::vector<Expr*>& history,
      const std::vector<bool>& td_influence);

  // Runs through history, applying only on influence to track how modifications
  // would influence the original axes.
  static std::vector<bool> replayInfluence(
      const std::vector<Expr*>& history,
      const std::vector<bool>& td_influence);

  // Goes through history and applies it to td, with the axis_map provided.
  // Axis_map entries of -1 mean those axes won't be modified
  static TensorDomain* replay(
      TensorDomain* td,
      const std::vector<Expr*>& history,
      const std::vector<int>& axis_map);

  // Takes td, and replays history backwards on it to create a new root tensor
  // domain using axis_map. Entries in axis_map == -1 will not be modified
  static TensorDomain* replayBackward(
      TensorDomain* td,
      const std::vector<Expr*>& history,
      const std::vector<int>& axis_map);

  static TensorDomain* replaySelf(
      TensorDomain* td,
      const std::vector<Expr*>& history,
      const std::vector<int>& axis_map);

  // Replays backwards all non-rfactor axes
  static TensorDomain* getRFactorRoot(TensorDomain* td);
};

} // namespace fuser
} // namespace jit
} // namespace torch
