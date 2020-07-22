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
  if (outer_it == index_map_.end() || inner_it == index_map_.end())
    return;

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
  if (out_it == index_map_.end())
    return;

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

IndexCompute::IndexCompute(
    const TensorDomain* td,
    const std::vector<Val*>& indices) {
  if (td->nDims() == 0 || indices.empty()) {
    indices_.push_back(new Int(0));
    return;
  }

  const bool exclude_reduction = td->nDims() > indices.size();

  TORCH_INTERNAL_ASSERT(
      td->noReductions().size() == indices.size() ||
          td->nDims() == indices.size(),
      "For IndexCompute the number of axes should match the number of dimensions in the TensorDomain.");

  {
    size_t i = 0;
    for (auto id : td->domain()) {
      if (exclude_reduction && id->isReduction())
        continue;
      index_map_[id] = indices[i++];
    }
  }

  const std::vector<Val*> domain_vals(td->domain().begin(), td->domain().end());

  // Run the split/merge operations backwards. This will modify the index_map_
  // so it can be used to index the root TensorDomain. Each entry in the root
  // TensorDomain should have an entry in index_map_ We might not want to run
  // these indices at the root of the domain, but actually at the rfactor root.
  // Fortunately we can run them all the way back, but grab the indices from the
  // map at the rfactor IterDomains.
  traverseFrom(indices[0]->fusion(), domain_vals, false);

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
    const TensorDomain* td,
    const std::vector<Val*>& _indices) {
  IndexCompute ic(td, _indices);
  return ic.indices_;
}

kir::TensorIndex* Index::getGlobalProducerIndex(
    const TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  // Grab indices from the loops
  std::vector<Val*> indices(loops.size());
  std::transform(
      loops.begin(), loops.end(), indices.begin(), [](kir::ForLoop* fl) {
        return fl->index();
      });

  // What would the consumer indices be if it was global, keeping in mind
  // reduction axes:
  const std::vector<Val*> c_inds =
      IndexCompute::get(consumer->domain(), indices);

  // Computed consumer indices should have everything we need for the producer
  std::vector<Val*> p_inds;
  auto p_root = TensorDomain::noReductions(producer->getRootDomain());
  // Number of root dims that are broadcasted
  size_t implicit_bcast_dims = 0;
  {
    auto c_root = consumer->getRootDomain();
    size_t it_c = 0, it_p = 0;
    while (it_c < c_root.size() && it_p < p_root.size()) {
      const bool is_bcast = p_root[it_p]->isBroadcast();
      if (c_root[it_c]->isBroadcast() && !p_root[it_p]->isBroadcast()) {
        it_c++;
      } else {
        if (!p_root[it_p]->isBroadcast()) {
          p_inds.push_back(c_inds[it_c]);
        } else {
          if (p_root[it_p]->getBroadcastType() == BroadcastType::WithStride) {
            p_inds.push_back(new Int(0));
          } else {
            implicit_bcast_dims++;
          }
        }
        it_c++;
        it_p++;
      }
    }
  }
  TORCH_INTERNAL_ASSERT(
      p_inds.size() == p_root.size() - implicit_bcast_dims,
      "Dimensionality error in code generator while computing tensor indices.");

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < p_inds.size(); i++) {
    std::stringstream ss;
    ss << "T" << producer->name() << ".stride[" << i << "]";
    strided_inds.push_back(
        mul(p_inds[i], new NamedScalar(ss.str(), DataType::Int)));
  }

  // Probably shouldn't ever hit this
  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(producer, strided_inds);
}

// Producer index for either shared or local memory
kir::TensorIndex* Index::getProducerIndex_impl(
    const TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  TORCH_INTERNAL_ASSERT(
      loops.size() == consumer->nDims(),
      "Dimensionality error in code generator while computing tensor indexes.");

  std::vector<kir::ForLoop*> loops_adjusted;
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
      [](kir::ForLoop* fl) { return fl->iter_domain(); });

  std::vector<Val*> indices(loops_adjusted.size());
  std::transform(
      loops_adjusted.begin(),
      loops_adjusted.end(),
      indices.begin(),
      [](kir::ForLoop* fl) {
        return fl->iter_domain()->isBroadcast() ? new Int(0) : fl->index();
      });

  std::vector<Val*> used_inds;
  std::vector<IterDomain*> used_ranges;
  bool unrolled = false;
  for (size_t i = 0; i < loops_adjusted.size(); i++) {
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
    if (producer->domain()->noReductions()[i]->isBroadcast())
      continue;

    used_inds.push_back(indices[i]);
    used_ranges.push_back(ranges[i]);
  }

  for (size_t i = 0; i < used_inds.size(); i++) {
    Val* ind = used_inds[i];
    for (size_t j = i + 1; j < used_ranges.size(); j++)
      ind = mul(ind, used_ranges[j]->extent());
    used_inds[i] = ind;
  }
  if (used_inds.size() == 0)
    used_inds.push_back(new Int(0));

  return new kir::TensorIndex(producer, used_inds);
}

kir::TensorIndex* Index::getGlobalConsumerIndex(
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  // If we're initializing a reduction buffer, we won't have the reduction
  // loops. If we're actually performing the reduction, we will.

  std::vector<Val*> indices(loops.size());
  std::transform(
      loops.begin(), loops.end(), indices.begin(), [](kir::ForLoop* fl) {
        return fl->index();
      });

  std::vector<Val*> computed_inds =
      IndexCompute::get(consumer->domain(), indices);

  auto root_dom = consumer->getRootDomain();
  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == TensorDomain::noReductions(root_dom).size() ||
          computed_inds.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  if (computed_inds.size() == root_dom.size()) {
    for (size_t i = 0; i < root_dom.size(); i++) {
      // Do this backwards so erase offset will be right
      auto axis = root_dom.size() - i - 1;
      if (root_dom[axis]->isReduction())
        computed_inds.erase(computed_inds.begin() + axis);
    }
  }

  {
    size_t root_i = 0, inds_i = 0;
    while (root_i < root_dom.size() && inds_i < computed_inds.size()) {
      if (root_dom[root_i]->isReduction()) {
        root_i++;
      } else {
        if (root_dom[root_i]->getBroadcastType() ==
            BroadcastType::WithoutStride) {
          computed_inds.erase(computed_inds.begin() + inds_i);
          root_i++;
        } else {
          if (root_dom[root_i]->getBroadcastType() ==
              BroadcastType::WithStride) {
            computed_inds[inds_i] = new Int(0);
          }
          root_i++;
          inds_i++;
        }
      }
    }
  }

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < computed_inds.size(); i++) {
    if (computed_inds[i]->isZeroInt()) {
      continue;
    }
    std::stringstream ss;
    ss << "T" << consumer->name() << ".stride[" << i << "]";
    strided_inds.push_back(
        mul(computed_inds[i], new NamedScalar(ss.str(), DataType::Int)));
  }

  // Probably shouldn't ever hit this
  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(consumer, strided_inds);
}

// Consumer index for either shared or local memory
kir::TensorIndex* Index::getConsumerIndex_impl(
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
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
  std::transform(
      loops.begin(), loops.end(), ranges.begin(), [](kir::ForLoop* fl) {
        return fl->iter_domain();
      });

  std::vector<Val*> indices(loops.size());
  std::transform(
      loops.begin(), loops.end(), indices.begin(), [](kir::ForLoop* fl) {
        return fl->iter_domain()->isBroadcast() ? new Int(0) : fl->index();
      });

  std::vector<Val*> used_inds;
  std::vector<IterDomain*> used_ranges;
  bool unrolled = false;
  {
    size_t c_i = 0, l_i = 0;
    while (c_i < consumer->nDims() && l_i < loops.size()) {
      if (consumer->axis(c_i)->isReduction()) {
        c_i++;
        if (have_reduction_iters)
          l_i++;
        continue;
      }
      if (ranges[l_i]->parallel_method() == ParallelType::Unroll)
        unrolled = true;

      if ((!unrolled && consumer->hasComputeAt() &&
           c_i < consumer->getThisComputeAtAxis()) ||
          (consumer->getMemoryType() == MemoryType::Shared &&
           ranges[l_i]->isBlockDim()) ||
          (consumer->getMemoryType() == MemoryType::Local &&
           ranges[l_i]->isThread()) ||
          (consumer->axis(c_i)->isBroadcast())) {
        c_i++;
        l_i++;
        continue;
      }

      used_inds.push_back(indices[l_i]);
      used_ranges.push_back(ranges[l_i]);
      l_i++;
      c_i++;
    }
  }

  for (size_t i = 0; i < used_inds.size(); i++) {
    Val* ind = used_inds[i];
    for (size_t j = i + 1; j < used_ranges.size(); j++)
      ind = mul(ind, used_ranges[j]->extent());
    used_inds[i] = ind;
  }

  if (used_inds.size() == 0)
    used_inds.push_back(new Int(0));

  return new kir::TensorIndex(consumer, used_inds);
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    const TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  TORCH_INTERNAL_ASSERT(
      loops.size() == consumer->nDims() ||
      loops.size() == consumer->domain()->noReductions().size());

  if (producer->domain()->noReductions().size() == 0) {
    return new kir::TensorIndex(producer, {});
  }

  if (producer->getMemoryType() == MemoryType::Global)
    return getGlobalProducerIndex(producer, consumer, loops);
  return getProducerIndex_impl(producer, consumer, loops);
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  TORCH_INTERNAL_ASSERT(
      loops.size() == consumer->nDims() ||
      loops.size() == consumer->domain()->noReductions().size());

  if (consumer->domain()->noReductions().size() == 0) {
    return new kir::TensorIndex(consumer, {});
  }

  if (consumer->getMemoryType() == MemoryType::Global)
    return getGlobalConsumerIndex(consumer, loops);
  return getConsumerIndex_impl(consumer, loops);
}

} // namespace fuser
} // namespace jit
} // namespace torch
