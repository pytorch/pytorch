#pragma once

#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <vector>

/*
 * Index compute takes in a list of indices typically generated from the
 * surrounding for loop nest. The number of indicies are intended to match the
 * number of dimensions of the incomming TensorView which may have less or more
 * dimensions than its root due to split/merge/reorder operations.
 * Split/merge/reorder operations are then replayed backwards produce resulting
 * indices (based on input indices) that match the root dimension.
 *
 * For example:
 * TV[I, J]
 * TV[Io, Ii{4}, J] = TV.split(I, factor=4)
 * indexCompute(TV, {i, j, k}) -> {i * 4 + j, k}
 *
 * These indices can then be flattened later based on strides.
 */

namespace torch {
namespace jit {
namespace fuser {

// Play split/merge/reorder operations backwards to compute indexing into
// original tensor.
struct IndexCompute : public TransformIter {
 protected:
  // Replay overrides which modify indices
  void replayBackward(Split* expr) override;
  void replayBackward(Merge* expr) override;
  void replayBackward(Reorder* expr) override;

  // Axis_map for
  std::vector<Int*> indices;

  IndexCompute(const TensorView* tv, std::vector<Int*> _indices);

 public:
  static std::vector<Int*> computeIndices(
      const TensorView* tv,
      std::vector<Int*> _indices);
};

} // namespace fuser
} // namespace jit
} // namespace torch
