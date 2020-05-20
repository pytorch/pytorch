#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

TensorDomain* IndexCompute::replayBackward(Split* split, TensorDomain*) {
  int ax = split->axis();
  TORCH_INTERNAL_ASSERT(
      ax >= 0 && (unsigned int)(ax + 1) < indices.size(),
      "Hit an invalid Split transformation during IndexCompute, axis is not within bounds.");
  indices[ax] = add(mul(indices[ax], split->factor()), indices[ax + 1]);
  indices.erase(indices.begin() + ax + 1);
  return split->in();
}

TensorDomain* IndexCompute::replayBackward(Merge* merge, TensorDomain*) {
  int ax = merge->axis();
  TORCH_INTERNAL_ASSERT(
      ax >= 0 && (unsigned int)ax < indices.size(),
      "Hit an invalid MERGE transformation during IndexCompute, axis is not within bounds.");

  Val* I = merge->in()->axis(ax + 1)->extent();
  Val* ind = indices[ax];
  indices[ax] = div(ind, I);
  indices.insert(indices.begin() + ax + 1, mod(ind, I));
  return merge->in();
}

TensorDomain* IndexCompute::replayBackward(Reorder* reorder, TensorDomain*) {
  // new2old[new_pos] = old_pos Generate new old2new map
  const std::vector<int>& new2old = reorder->new2old();

  std::vector<Val*> reordered_indices;

  // Reverse the map so we can simply push back into reordered_indices
  // old2new[old_pos] = new_pos
  std::vector<int> old2new(new2old.size(), -1);

  for (decltype(new2old.size()) i = 0; i < new2old.size(); i++) {
    int new_pos = i;
    int old_pos = new2old[i];
    TORCH_INTERNAL_ASSERT(
        new_pos >= 0 && (unsigned int)new_pos < indices.size() &&
            old_pos >= 0 && (unsigned int)old_pos < indices.size(),
        "Hit an invalid reorder transformation during IndexCompute,"
        " at least one move position is not within bounds.");
    old2new[old_pos] = new_pos;
  }
  for (auto new_pos : old2new) {
    // int new_pos = old2new[i];
    // int old_pos = i;
    // reordered_indices[old_pos] = indices[new_pos];
    reordered_indices.push_back(indices[new_pos]);
  }

  indices = reordered_indices;
  return reorder->in();
}

TensorDomain* IndexCompute::runBackward(std::vector<Expr*> history) {
  TensorDomain* running_td = nullptr;
  for (auto it = history.rbegin(); it != history.rend(); it++)
    running_td = TransformIter::replayBackward(*it, running_td);

  return running_td;
}

IndexCompute::IndexCompute(TensorDomain* td, std::vector<Val*> _indices)
    : indices(std::move(_indices)) {
  bool exclude_reduction = td->nDims() > indices.size();

  TORCH_INTERNAL_ASSERT(
      td->noReductions()->nDims() == indices.size() ||
          td->nDims() == indices.size(),
      "For IndexCompute the number of axes should match the number of dimensions"
      " in the TensorDomain.");

  // If we need to ignore the reduction dimensions because a tensor is
  // being consumed, not produced, then insert dummy dimensions in the
  // indices for bookkeeping while replaying split/merge/reorder operations.
  if (exclude_reduction)
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++)
      if (td->axis(i)->isReduction())
        indices.insert(indices.begin() + i, new Int(-1));

  TORCH_INTERNAL_ASSERT(
      indices.size() == td->nDims(),
      "Attempted to modify indices for IndexCompute, but didn't work.");

  // Run the split/merge/reorder operations backwards. This will
  // Modify std::vector<Int*> indices so it can be used to index
  // the root TensorDomain which should now match the physical axes.
  TensorDomain* root = TransformIter::getRoot(td);
  auto history = TransformIter::getHistory(td);
  if (exclude_reduction && td->hasRFactor()) {
    root = TransformIter::getRFactorRoot(td);
    auto rfactor_history = TransformIter::getHistory(root);
    history.erase(history.begin(), history.begin() + rfactor_history.size());
  }

  runBackward(history);

  TORCH_INTERNAL_ASSERT(
      root->nDims() == indices.size(),
      "Error during IndexCompute. The number of indices generated"
      " after running the transformations backwards should match"
      " the number of dimensions of the root TensorDomain.");
}

std::vector<Val*> IndexCompute::get(
    TensorDomain* td,
    const std::vector<Val*>& _indices) {
  IndexCompute ic(td, _indices);
  return ic.indices;
}

TensorIndex* Index::getGlobalProducerIndex(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  // This replay will ignore reduction dimensions on the producer
  auto pind =
      TransformReplay::replayPasC(producer->domain(), consumer->domain(), -1);

  TORCH_INTERNAL_ASSERT(
      loops.size() == pind->noReductions()->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  std::vector<Val*> indices(loops.size());
  std::transform(loops.begin(), loops.end(), indices.begin(), [](ForLoop* fl) {
    return fl->index();
  });
  std::vector<Val*> computed_inds = IndexCompute::get(pind, indices);

  auto root_producer = producer->getRootDomain();

  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == root_producer->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  for (decltype(computed_inds.size()) i{0}; i < computed_inds.size(); i++) {
    if (root_producer->axis(i)->isReduction())
      computed_inds.erase(computed_inds.begin() + i);
  }

  std::vector<Val*> strided_inds;
  for (decltype(computed_inds.size()) i{0}; i < computed_inds.size(); i++) {
    std::stringstream ss;
    ss << "T" << producer->name() << ".stride[" << i << "]";
    strided_inds.push_back(
        mul(computed_inds[i], new NamedScalar(ss.str(), DataType::Int)));
  }

  // Probably shouldn't ever hit this
  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new TensorIndex(producer, strided_inds);
}

// Producer index for either shared or local memory
TensorIndex* Index::getProducerIndex_impl(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  TORCH_INTERNAL_ASSERT(
      loops.size() == producer->domain()->noReductions()->nDims(),
      "Expected a tensor with ",
      loops.size(),
      " dimensions but got one with ",
      producer->nDims());

  std::vector<IterDomain*> ranges(loops.size());
  std::transform(loops.begin(), loops.end(), ranges.begin(), [](ForLoop* fl) {
    return fl->iter_domain();
  });

  std::vector<Val*> indices(loops.size());
  std::transform(loops.begin(), loops.end(), indices.begin(), [](ForLoop* fl) {
    return fl->index();
  });

  std::vector<Val*> used_inds;
  std::vector<IterDomain*> used_ranges;
  bool unrolled = false;
  for (decltype(loops.size()) i{0}; i < loops.size(); i++) {
    if (ranges[i]->parallel_method() == ParallelType::Unroll)
      unrolled = true;
    if (!unrolled && producer->hasComputeAt() &&
        i < producer->getComputeAtAxis())
      continue;
    if (producer->getMemoryType() == MemoryType::Shared &&
        ranges[i]->isBlockDim())
      continue;
    if (producer->getMemoryType() == MemoryType::Local && ranges[i]->isThread())
      continue;
    used_inds.push_back(indices[i]);
    used_ranges.push_back(ranges[i]);
  }

  for (decltype(used_inds.size()) i{0}; i < used_inds.size(); i++) {
    Val* ind = used_inds[i];
    for (decltype(used_ranges.size()) j{i + 1}; j < used_ranges.size(); j++)
      ind = mul(ind, used_ranges[j]->extent());
    used_inds[i] = ind;
  }
  if (used_inds.size() == 0)
    used_inds.push_back(new Int(0));

  return new TensorIndex(producer, used_inds);
}

TensorIndex* Index::getGlobalConsumerIndex(
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  // If we're initializing a reduction buffer, we won't have the reduction
  // loops. If we're actually performing the reduction, we will.

  std::vector<Val*> indices(loops.size());
  std::transform(loops.begin(), loops.end(), indices.begin(), [](ForLoop* fl) {
    return fl->index();
  });

  std::vector<Val*> computed_inds =
      IndexCompute::get(consumer->domain(), indices);

  TensorDomain* root_dom = consumer->getRootDomain();
  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == root_dom->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  for (decltype(root_dom->nDims()) i{0}; i < root_dom->nDims(); i++) {
    // Do this backwards so erase offset will be right
    auto axis = root_dom->nDims() - i - 1;
    if (root_dom->axis(axis)->isReduction())
      computed_inds.erase(computed_inds.begin() + axis);
  }

  std::vector<Val*> strided_inds;
  for (decltype(computed_inds.size()) i{0}; i < computed_inds.size(); i++) {
    std::stringstream ss;
    ss << "T" << consumer->name() << ".stride[" << i << "]";
    strided_inds.push_back(
        mul(computed_inds[i], new NamedScalar(ss.str(), DataType::Int)));
  }

  // Probably shouldn't ever hit this
  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new TensorIndex(consumer, strided_inds);
}

TensorIndex* Index::getConsumerIndex_impl(
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  // If we're initializing a reduction buffer, we won't have the reduction
  // loops. If we're actually performing the reduction, we will.

  bool have_reduction_iters = loops.size() == consumer->nDims();

  if (!have_reduction_iters) {
    TORCH_INTERNAL_ASSERT(
        // Init reduction space
        loops.size() == consumer->domain()->noReductions()->nDims(),
        "Expected a tensor with ",
        loops.size(),
        " dimensions but got one with ",
        consumer->domain()->noReductions()->nDims());
  } else {
    TORCH_INTERNAL_ASSERT(
        // Calling the reduction op
        loops.size() == consumer->nDims(),
        "Expected a tensor with ",
        loops.size(),
        " dimensions but got one with ",
        consumer->nDims());
  }

  std::vector<IterDomain*> ranges(loops.size());
  std::transform(loops.begin(), loops.end(), ranges.begin(), [](ForLoop* fl) {
    return fl->iter_domain();
  });

  std::vector<Val*> indices(loops.size());
  std::transform(loops.begin(), loops.end(), indices.begin(), [](ForLoop* fl) {
    return fl->index();
  });

  std::vector<Val*> used_inds;
  std::vector<IterDomain*> used_ranges;
  bool unrolled = false;
  for (decltype(loops.size()) i{0}; i < loops.size(); i++) {
    if (have_reduction_iters && consumer->axis(i)->isReduction())
      continue;
    if (ranges[i]->parallel_method() == ParallelType::Unroll)
      unrolled = true;
    if (!unrolled && consumer->hasComputeAt() &&
        i < consumer->getComputeAtAxis())
      continue;
    if (consumer->getMemoryType() == MemoryType::Shared &&
        ranges[i]->isBlockDim())
      continue;
    if (consumer->getMemoryType() == MemoryType::Local && ranges[i]->isThread())
      continue;

    used_inds.push_back(indices[i]);
    used_ranges.push_back(ranges[i]);
  }

  for (decltype(used_inds.size()) i{0}; i < used_inds.size(); i++) {
    Val* ind = used_inds[i];
    for (decltype(used_ranges.size()) j{i + 1}; j < used_ranges.size(); j++)
      ind = mul(ind, used_ranges[j]->extent());
    used_inds[i] = ind;
  }

  if (used_inds.size() == 0)
    used_inds.push_back(new Int(0));

  return new TensorIndex(consumer, used_inds);
}

// Producer is the inputs of an expression
TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  if (producer->getMemoryType() == MemoryType::Global)
    return getGlobalProducerIndex(producer, consumer, loops);
  return getProducerIndex_impl(producer, consumer, loops);
}

// Consumer is the output of an expression
TensorIndex* Index::getConsumerIndex(
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  if (consumer->getMemoryType() == MemoryType::Global)
    return getGlobalConsumerIndex(consumer, loops);
  return getConsumerIndex_impl(consumer, loops);
}

} // namespace fuser
} // namespace jit
} // namespace torch
