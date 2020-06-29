#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

void IndexCompute::handle(Split* split) {
  auto in_id = split->in();
  auto outer_id = split->outer();
  auto inner_id = split->inner();

  auto outer_it = index_map_.find(outer_id);
  auto inner_it = index_map_.find(inner_id);
  TORCH_INTERNAL_ASSERT(
      outer_it != index_map_.end() && inner_it != index_map_.end(),
      "Error in index compute, did not compute a necessary intermediate value.");

  auto outer_ind = outer_it->second;
  auto inner_ind = inner_it->second;

  auto ind = add(mul(outer_ind, split->factor()), inner_ind);
  index_map_[in_id] = ind;
}

void IndexCompute::handle(Merge* merge) {
  auto out_id = merge->out();
  auto outer_id = merge->outer();
  auto inner_id = merge->inner();

  auto out_it = index_map_.find(out_id);
  TORCH_INTERNAL_ASSERT(
      out_it != index_map_.end(),
      "Error in index compute, did not compute a necessary intermediate value.");

  auto out_ind = out_it->second;

  Val* I = inner_id->extent();
  Val* outer_ind = div(out_ind, I);
  Val* inner_ind = mod(out_ind, I);

  index_map_[outer_id] = outer_ind;
  index_map_[inner_id] = inner_ind;
}

void IndexCompute::handle(Expr* e) {
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid expr type found in transform traversal.");
  }
  BackwardVisitor::handle(e);
}

IndexCompute::IndexCompute(TensorDomain* td, const std::vector<Val*>& indices) {
  if (td->nDims() == 0 || indices.empty()) {
    indices_.push_back(new Int(0));
    return;
  }

  bool exclude_reduction = td->nDims() > indices.size();

  TORCH_INTERNAL_ASSERT(
      td->noReductions().size() == indices.size() ||
          td->nDims() == indices.size(),
      "For IndexCompute the number of axes should match the number of dimensions in the TensorDomain.");

  TORCH_INTERNAL_ASSERT(!td->hasRFactor(), "Not implemented yet.");

  {
    size_t i = 0;
    for (auto id : td->domain()) {
      if (exclude_reduction && id->isReduction())
        continue;
      index_map_[id] = indices[i++];
    }
  }

  std::vector<Val*> domain_vals(td->domain().begin(), td->domain().end());

  // Run the split/merge operations backwards. This will modify the index_map_
  // so it can be used to index the root TensorDomain. Each entry in the root
  // TensorDomain should have an entry in index_map_ We might not want to run
  // these indices at the root of the domain, but actually at the rfactor root.
  // Fortunately we can run them all the way back, but grab the indices from the
  // map at the rfactor IterDomains.
  traverseFrom(indices[0]->fusion(), domain_vals, false);

  std::vector<Val*> inds;
  for (auto id : td->rootDomain()) {
    if (exclude_reduction && id->isReduction())
      continue;
    auto it = index_map_.find(id);
    TORCH_INTERNAL_ASSERT(
        it != index_map_.end(),
        "Error during index compute, missed computing a value.");
    indices_.push_back(it->second);
  }
}

std::vector<Val*> IndexCompute::get(
    TensorDomain* td,
    const std::vector<Val*>& _indices) {
  IndexCompute ic(td, _indices);
  return ic.indices_;
}

TensorIndex* Index::getGlobalProducerIndex(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  // This replay will ignore reduction dimensions on the producer
  auto pind =
      TransformReplay::replayPasC(producer->domain(), consumer->domain(), -1);

  TORCH_INTERNAL_ASSERT(
      loops.size() == consumer->nDims(),
      "Dimensionality error in code generator while computing tensor indexes.");

  std::vector<ForLoop*> loops_adjusted;
  size_t it_c = 0, it_p = 0;
  while (it_c < consumer->nDims() && it_p < pind->noReductions().size()) {
    if (consumer->axis(it_c)->isBroadcast() &&
        !pind->noReductions()[it_p]->isBroadcast()) {
      it_c++;
    } else {
      loops_adjusted.push_back(loops[it_c]);
      it_c++;
      it_p++;
    }
  }

  TORCH_INTERNAL_ASSERT(
      loops_adjusted.size() == pind->noReductions().size(),
      "Dimensionality error in code generator while computing tensor indexes.");

  std::vector<Val*> indices(loops_adjusted.size());
  std::transform(
      loops_adjusted.begin(),
      loops_adjusted.end(),
      indices.begin(),
      [](ForLoop* fl) { return fl->index(); });
  std::vector<Val*> computed_inds = IndexCompute::get(pind, indices);

  auto root_domain = producer->getRootDomain();

  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == root_domain.size(),
      "Dimensionality error in code generator while computing indexing.");

  for (decltype(computed_inds.size()) i{0}; i < computed_inds.size(); i++) {
    if (root_domain[i]->isReduction() || root_domain[i]->isBroadcast())
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
      loops.size() == consumer->nDims(),
      "Dimensionality error in code generator while computing tensor indexes.");

  std::vector<ForLoop*> loops_adjusted;
  size_t it_c = 0, it_p = 0;
  while (it_c < consumer->nDims() && it_p < producer->nDims()) {
    if (consumer->axis(it_c)->isBroadcast() &&
        !producer->axis(it_p)->isBroadcast()) {
      it_c++;
    } else {
      loops_adjusted.push_back(loops[it_c]);
      it_c++;
      it_p++;
    }
  }

  TORCH_INTERNAL_ASSERT(
      loops_adjusted.size() == producer->domain()->noReductions().size(),
      "Expected a tensor with ",
      loops_adjusted.size(),
      " dimensions but got one with ",
      producer->nDims());

  std::vector<IterDomain*> ranges(loops_adjusted.size());
  std::transform(
      loops_adjusted.begin(),
      loops_adjusted.end(),
      ranges.begin(),
      [](ForLoop* fl) { return fl->iter_domain(); });

  std::vector<Val*> indices(loops_adjusted.size());
  std::transform(
      loops_adjusted.begin(),
      loops_adjusted.end(),
      indices.begin(),
      [](ForLoop* fl) {
        return fl->iter_domain()->isBroadcast() ? new Int(0) : fl->index();
      });

  std::vector<Val*> used_inds;
  std::vector<IterDomain*> used_ranges;
  bool unrolled = false;
  for (decltype(loops_adjusted.size()) i{0}; i < loops_adjusted.size(); i++) {
    if (ranges[i]->parallel_method() == ParallelType::Unroll)
      unrolled = true;
    if (!unrolled && producer->hasComputeAt() &&
        i < producer->getThisComputeAtAxis())
      continue;
    if (producer->getMemoryType() == MemoryType::Shared &&
        ranges[i]->isBlockDim())
      continue;
    if (producer->getMemoryType() == MemoryType::Local && ranges[i]->isThread())
      continue;
    if (ranges[i]->isBroadcast())
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

  auto root_dom = consumer->getRootDomain();
  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == TensorDomain::noReductions(root_dom).size() ||
          computed_inds.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  if (computed_inds.size() == root_dom.size())
    for (decltype(root_dom.size()) i{0}; i < root_dom.size(); i++) {
      // Do this backwards so erase offset will be right
      auto axis = root_dom.size() - i - 1;
      if (root_dom[axis]->isReduction() || root_dom[i]->isBroadcast())
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

// Consumer index for either shared or local memory
TensorIndex* Index::getConsumerIndex_impl(
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  // If we're initializing a reduction buffer, we won't have the reduction
  // loops. If we're actually performing the reduction, we will.

  bool have_reduction_iters = loops.size() == consumer->nDims();

  if (!have_reduction_iters) {
    TORCH_INTERNAL_ASSERT(
        // Init reduction space
        loops.size() == consumer->domain()->noReductions().size(),
        "Expected a tensor with ",
        loops.size(),
        " dimensions but got one with ",
        consumer->domain()->noReductions().size());
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
    return fl->iter_domain()->isBroadcast() ? new Int(0) : fl->index();
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
        i < consumer->getThisComputeAtAxis())
      continue;
    if (consumer->getMemoryType() == MemoryType::Shared &&
        ranges[i]->isBlockDim())
      continue;
    if (consumer->getMemoryType() == MemoryType::Local && ranges[i]->isThread())
      continue;
    if (ranges[i]->isBroadcast())
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
  TORCH_INTERNAL_ASSERT(
      loops.size() == consumer->nDims() ||
      loops.size() == consumer->domain()->noReductions().size());

  if (producer->getMemoryType() == MemoryType::Global)
    return getGlobalProducerIndex(producer, consumer, loops);
  return getProducerIndex_impl(producer, consumer, loops);
}

// Consumer is the output of an expression
TensorIndex* Index::getConsumerIndex(
    TensorView* consumer,
    const std::vector<ForLoop*>& loops) {
  TORCH_INTERNAL_ASSERT(
      loops.size() == consumer->nDims() ||
      loops.size() == consumer->domain()->noReductions().size());

  if (consumer->getMemoryType() == MemoryType::Global)
    return getGlobalConsumerIndex(consumer, loops);
  return getConsumerIndex_impl(consumer, loops);
}

} // namespace fuser
} // namespace jit
} // namespace torch
