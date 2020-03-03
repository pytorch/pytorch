#pragma once

#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/transform_iter.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

struct IndexCompute : public TransformIter {
 protected:
  void replayBackward(Split* expr) override {
    int ax = expr->axis();
    TORCH_CHECK(ax >= 0 && ax + 1 < indices.size());
    indices[ax] = static_cast<Int*>(mul(indices[ax], indices[ax + 1]));
    indices.erase(indices.begin() + ax + 1);
  }

  void replayBackward(Merge* expr) override {
    int ax = expr->axis();
    TORCH_CHECK(ax >= 0 && ax < indices.size());

    Int* O = expr->in()->axis(ax + 1)->size();
    Int* ind = indices[ax];
    indices[ax] = static_cast<Int*>(div(ind, O));
    indices.insert(indices.begin() + ax + 1, static_cast<Int*>(mod(ind, O)));
  }

  void replayBackward(Reorder* expr) override {
    // pos2axis[new_pos] = old_pos Generate new axis2pos map
    const std::vector<int>& pos2axis = expr->pos2axis();

    std::vector<Int*> reordered_indices;

    // Reverse the map so we can simply push back into reordered_indices
    // axis2pos[old_pos] = new_pos
    std::vector<int> axis2pos(pos2axis.size(), -1);

    for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
      int new_pos = i;
      int old_pos = pos2axis[i];
      TORCH_CHECK(
          new_pos >= 0 && new_pos < indices.size() && old_pos >= 0 &&
          old_pos < indices.size());
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

  std::vector<int> axis_map;
  std::vector<Int*> indices;

 public:
  IndexCompute(const TensorView* tv, std::vector<Int*> _indices) {
    indices = _indices;

    TensorDomain* td = tv->domain();

    bool exclude_reduction = td->size() > indices.size();
    TORCH_CHECK(td->size() >= indices.size());

    // Add fake indices on reduction axes if they aren't there
    // just for bookkeeping of split/merge/reorder.
    if (exclude_reduction)
      for (decltype(td->size()) i{0}; i < td->size(); i++)
        if (td->axis(i)->isReduction())
          indices.insert(indices.begin() + i, new Int(-1));
    TensorDomain* root = TransformIter::runBackward(td, true);
    TORCH_CHECK(root->size() == indices.size());
    // Remove indices associated with reduction axes
    if (exclude_reduction) {
      for (int i = root->size() - 1; i >= 0; i--)
        if (root->axis(i)->isReduction())
          indices.erase(indices.begin() + i);
    }
  }
  static std::vector<Int*> computeIndices(
      const TensorView* tv,
      std::vector<Int*> _indices) {
    IndexCompute ic(tv, _indices);
    return ic.indices;
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch