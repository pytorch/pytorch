#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

namespace torch {
namespace jit {
namespace fuser {

TensorDomain* TransformRFactor::runReplay(
    TensorDomain* orig_td,
    std::vector<int> axes) {
  int ndims = (int)orig_td->nDims();

  // Adjust and check provided axes
  std::transform(axes.begin(), axes.end(), axes.begin(), [ndims](int i) {
    TORCH_CHECK(
        i >= -ndims && i < ndims,
        "Rfactor replay recieved an axis outside the number of dims in the tensor, acceptable inclusive range is ",
        -ndims,
        " to ",
        ndims - 1);
    return i < 0 ? i + ndims : i;
  });

  // remove duplicates, and put into a set for searching
  std::set<int> axes_set(axes.begin(), axes.end());

  // Make a copy of orig_td as we're going to change its history:
  bool found_rfactor = false;
  std::vector<IterDomain*> domain_copy;
  for (int i{0}; i < ndims; i++) {
    IterDomain* orig_axis = orig_td->axis(i);
    if (axes_set.find(i) != axes_set.end())
      TORCH_CHECK(
          orig_axis->isReduction(),
          "Tried to rFactor an axis that is not a reduction.");

    if (orig_axis->isReduction()) {
      if (axes_set.find(i) == axes_set.end()) {
        domain_copy.push_back(new IterDomain(
            orig_axis->start(),
            orig_axis->extent(),
            orig_axis->parallel_method(),
            false,
            true));
        found_rfactor = true;
      } else {
        domain_copy.push_back(new IterDomain(
            orig_axis->start(),
            orig_axis->extent(),
            orig_axis->parallel_method(),
            true,
            true));
      }
    } else {
      domain_copy.push_back(orig_td->axis(i));
    }
  }
  TORCH_CHECK(found_rfactor, "Could not find axis to rfactor out.");

  // TD that we will actually modify,
  TensorDomain* out_td = new TensorDomain(domain_copy);

  // Axis map to create history for non-rfactor axes
  std::vector<int> axis_map(ndims, -1);
  std::vector<int> orig_rfactor_axis_map(ndims, -1);
  std::set<IterDomain*> rfactor_ids;
  for (decltype(out_td->nDims()) i{0}; i < out_td->nDims(); i++)
    if (!out_td->axis(i)->isRFactorProduct()) {
      axis_map[i] = i;
    } else {
      orig_rfactor_axis_map[i] = i;
    }

  // Replay non-rfactor axes
  auto running_td = TransformIter::replayBackward(
      out_td, TransformIter::getHistory(orig_td), axis_map);

  // running_td has iteration domains on the right, but to find a valid rfactor
  // root, we want those to be on the right. If we continued to replay backward
  // we likely won't have a valid rfactor root. Lets manually insert a reorder
  // so we have a valid rfactor root.

  std::vector<int> new2old(running_td->nDims());
  {
    int running_pos = 0;
    for (decltype(running_td->nDims()) i{0}; i < running_td->nDims(); i++)
      if (!running_td->axis(i)->isRFactorProduct())
        new2old[i] = running_pos++;

    for (decltype(running_td->nDims()) i{0}; i < running_td->nDims(); i++)
      if (running_td->axis(i)->isRFactorProduct())
        new2old[i] = running_pos++;
  }
  std::vector<int> reorder_axis_map(running_td->nDims());
  std::iota(reorder_axis_map.begin(), reorder_axis_map.end(), 0);

  running_td = TransformIter::replayBackward(
      running_td,
      // include dummy reorder
      {new Reorder(
          new TensorDomain(running_td->domain()), running_td, new2old)},
      reorder_axis_map);

  // how do we find axes
  // Need axis map from rfactor axes in running_td to corresponding axes in
  // orig_td orig_rfactor_axis_map goes from orig_td to out_td we want it to
  // go from orig_td to running_td

  // Go from IterDomain to its position in running_td
  std::unordered_map<IterDomain*, int> new_pos;
  for (decltype(running_td->nDims()) i{0}; i < running_td->nDims(); i++) {
    new_pos[running_td->axis(i)] = i;
  }

  for (decltype(out_td->nDims()) i{0}; i < out_td->nDims(); i++)
    if (orig_rfactor_axis_map[i] != -1) {
      // int orig_td_pos = i;
      int out_td_pos = orig_rfactor_axis_map[i];
      TORCH_INTERNAL_ASSERT(
          new_pos.find(out_td->axis(out_td_pos)) != new_pos.end(),
          "Error aligning axes in rfactor first TD replay.");
      int running_td_pos = new_pos[out_td->axis(out_td_pos)];
      orig_rfactor_axis_map[i] = running_td_pos;
    }

  TransformIter::replayBackward(
      running_td, TransformIter::getHistory(orig_td), orig_rfactor_axis_map);

  return out_td;
}

TensorDomain* TransformRFactor::runReplay2(
    TensorDomain* in_td,
    std::vector<int> axes) {
  int ndims = (int)in_td->nDims();

  // Adjust and check provided axes
  std::transform(axes.begin(), axes.end(), axes.begin(), [ndims](int i) {
    TORCH_CHECK(
        i >= -ndims && i < ndims,
        "Rfactor replay recieved an axis outside the number of dims in the tensor, acceptable inclusive range is ",
        -ndims,
        " to ",
        ndims - 1);
    return i < 0 ? i + ndims : i;
  });

  // remove duplicates, and put into a set for searching
  std::set<int> axes_set(axes.begin(), axes.end());

  bool found_rfactor = false;
  // Axes marked as rfactor, these will be removed from this domain
  std::vector<bool> rfactor_axes(in_td->nDims(), false);
  for (int i{0}; i < ndims; i++) {
    bool in_set = axes_set.find(i) != axes_set.end();
    IterDomain* orig_axis = in_td->axis(i);

    if (in_set) {
      TORCH_CHECK(
          orig_axis->isReduction(),
          "Tried to rFactor an axis that is not a reduction.");
      rfactor_axes[i] = true;
      found_rfactor = true;
    }
  }

  TORCH_CHECK(found_rfactor, "Could not find axis to rfactor out.");
  auto root_rfactor_axes = TransformIter::getRootInfluence(in_td, rfactor_axes);

  // Root axes involved in rfactor, these axes should not be replayed, they need
  // to be either removed completely, or part of the root domain
  auto root_dom = TransformIter::getRoot(in_td);
  TORCH_INTERNAL_ASSERT(
      root_rfactor_axes.size() == root_dom->nDims(),
      "Error backpropagating influence of rfactor.");

  // Forward propagate influence back to the end we want to mark everything
  // that's part of the rfactor
  rfactor_axes = TransformIter::replayInfluence(
      TransformIter::getHistory(in_td), root_rfactor_axes);

  // Axes part of rfactor we need to keep
  std::vector<IterDomain*> rfactor_axes_keep;

  for (int i{0}; i < ndims; i++) {
    if (rfactor_axes[i] && axes_set.find(i) == axes_set.end()) {
      TORCH_INTERNAL_ASSERT(
          in_td->axis(i)->isReduction(),
          "Error occured when tracking rfactor axes.");
      rfactor_axes_keep.push_back(in_td->axis(i));
    }
  }

  int root_ndims = (int)root_dom->nDims();
  std::vector<IterDomain*> domain_copy;
  // These are the axes that are not involved in the rfactor.
  for (int i{0}; i < root_ndims; i++) {
    if (!root_rfactor_axes[i]) {
      domain_copy.push_back(root_dom->axis(i));
    }
  }

  TORCH_INTERNAL_ASSERT(
      domain_copy.size() < root_dom->nDims(),
      "Error during rfactor, didn't get any rfactor axes.");

  // Setup axis map before we add back in the rfactor_axes
  std::vector<int> replay_axis_map(root_dom->nDims(), -1);
  {
    decltype(domain_copy.size()) it = 0;
    decltype(root_dom->nDims()) ir = 0;
    while (it < domain_copy.size() && ir < root_dom->nDims()) {
      if (root_rfactor_axes[ir]) {
        ir++;
      } else {
        replay_axis_map[ir++] = it++;
      }
    }
    TORCH_INTERNAL_ASSERT(
        it == domain_copy.size(),
        "Error during rfactor, missed an unmodified root domain.");
  }

  // Push back the rfactor axes we need to keep
  domain_copy.insert(
      domain_copy.end(), rfactor_axes_keep.begin(), rfactor_axes_keep.end());

  // TD that we will actually modify
  TensorDomain* replay_root_td = new TensorDomain(domain_copy);
  auto td = TransformIter::replay(
      replay_root_td, TransformIter::getHistory(in_td), replay_axis_map);

  return td;
}

} // namespace fuser
} // namespace jit
} // namespace torch
