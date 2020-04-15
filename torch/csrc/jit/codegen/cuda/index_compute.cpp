#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {

void IndexCompute::replayBackward(Split* expr) {
  int ax = expr->axis();
  TORCH_INTERNAL_ASSERT(
      ax >= 0 && ax + 1 < indices.size(),
      "Hit an invalid Split transformation during IndexCompute, axis is not within bounds.");
  indices[ax] = add(mul(indices[ax], expr->factor()), indices[ax + 1]);
  indices.erase(indices.begin() + ax + 1);
}

void IndexCompute::replayBackward(Merge* expr) {
  int ax = expr->axis();
  TORCH_INTERNAL_ASSERT(
      ax >= 0 && ax < indices.size(),
      "Hit an invalid MERGE transformation during IndexCompute, axis is not within bounds.");

  Val* I = expr->in()->axis(ax + 1)->size();
  Val* ind = indices[ax];
  indices[ax] = div(ind, I);
  indices.insert(indices.begin() + ax + 1, mod(ind, I));
}

void IndexCompute::replayBackward(Reorder* expr) {
  // pos2axis[new_pos] = old_pos Generate new axis2pos map
  const std::vector<int>& pos2axis = expr->pos2axis();

  std::vector<Val*> reordered_indices;

  // Reverse the map so we can simply push back into reordered_indices
  // axis2pos[old_pos] = new_pos
  std::vector<int> axis2pos(pos2axis.size(), -1);

  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    int new_pos = i;
    int old_pos = pos2axis[i];
    TORCH_INTERNAL_ASSERT(
        new_pos >= 0 && new_pos < indices.size() && old_pos >= 0 &&
            old_pos < indices.size(),
        "Hit an invalid reorder transformation during IndexCompute,"
        " at least one move position is not within bounds.");
    axis2pos[old_pos] = new_pos;
  }
  for (decltype(axis2pos.size()) i = 0; i < axis2pos.size(); i++) {
    int new_pos = axis2pos[i];
    int old_pos = i;
    // reordered_indices[old_pos] = indices[new_pos];
    reordered_indices.push_back(indices[new_pos]);
  }

  indices = reordered_indices;
}

IndexCompute::IndexCompute(const TensorView* tv, std::vector<Val*> _indices) {
  indices = std::move(_indices);

  TensorDomain* td = tv->domain();

  bool exclude_reduction = td->size() > indices.size();

  TORCH_CHECK(
      exclude_reduction || td->size() == indices.size(),
      "For IndexCompute the number of axis should match the number of dimensions"
      " in the TensorView.");

  // If we need to ignore the reduction dimensions because a tensor is
  // being consumed, not produced, then insert dummy dimensions in the
  // indices for bookkeeping while replaying split/merge/reorder operations.
  if (exclude_reduction)
    for (decltype(td->size()) i{0}; i < td->size(); i++)
      if (td->axis(i)->isReduction())
        indices.insert(indices.begin() + i, new Int(-1));

  // Run the split/merge/reorder operations backwards. This will
  // Modify std::vector<Int*> indices so it can be used to index
  // the root TensorDomain which should now match the physical axes.
  TensorDomain* root = TransformIter::runBackward(td, true);

  TORCH_INTERNAL_ASSERT(
      root->size() == indices.size(),
      "Error during IndexCompute. The number of indices generated"
      " after running the transformations backwards should match"
      " the number of dimensions of the root TensorView.");

  // Remove indices associated with reduction axes, we had them just for
  // bookkeeping.
  if (exclude_reduction) {
    for (auto i = root->size() - 1; i >= 0; i--)
      if (root->axis(i)->isReduction())
        indices.erase(indices.begin() + i);
  }
}

std::vector<Val*> IndexCompute::computeIndices(
    const TensorView* tv,
    std::vector<Val*> _indices) {
  IndexCompute ic(tv, std::move(_indices));
  return ic.indices;
}

} // namespace fuser
} // namespace jit
} // namespace torch
