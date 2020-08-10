#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

struct ContigMergeInfo {
  bool mergeable = false;
  std::unordered_set<IterDomain*> zero_root_ids;
  IterDomain* last_root_id = nullptr;
};

ContigMergeInfo isContiguousMerge(
    Merge* merge,
    const std::vector<IterDomain*>& root_domain,
    const std::vector<bool>& root_contiguity) {
  // Check if all inputs to merge operation are contiguous in root domain
  // (consecutive, and marked contiguous)
  ContigMergeInfo contig_merge_info;

  auto merge_inputs = IterVisitor::getInputsTo({merge->out()});

  std::unordered_set<IterDomain*> merge_input_ids;
  for (auto inp : merge_inputs) {
    if (inp->getValType().value() == ValType::IterDomain) {
      merge_input_ids.emplace(inp->as<IterDomain>());
    } else {
      TORCH_INTERNAL_ASSERT(
          inp->isAnInt(), "Found invalid input to merge heirarchy, ", inp, ".");
    }
  }

  contig_merge_info.zero_root_ids = merge_input_ids;

  bool contig_dims = true;

  IterDomain* start = nullptr;
  for (size_t i = 0; i < root_domain.size(); i++) {
    auto id = root_domain[i];

    if (merge_inputs.find(id) == merge_inputs.end()) {
      if (start != nullptr) {
        contig_dims = false;
      }
    }

    if (!root_contiguity[i]) {
      contig_dims = false;
    }

    contig_merge_info.last_root_id = id;

    start = start == nullptr ? id : start;

    if (start->getIterType() != id->getIterType()) {
      break;
    }

    merge_input_ids.erase(root_domain[i]);
  }

  if (!(merge_input_ids.empty() && contig_dims)) {
    contig_merge_info.zero_root_ids.clear();
    contig_merge_info.last_root_id = nullptr;
    return contig_merge_info;
  }

  contig_merge_info.zero_root_ids.erase(contig_merge_info.last_root_id);

  auto exprs = ExprSort::getExprs(
      merge->out()->fusion(), std::vector<Val*>({merge->out()}));
  std::unordered_set<IterDomain*> expr_inputs;
  std::unordered_set<IterDomain*> expr_outputs;

  for (auto expr : exprs) {
    if (expr->getExprType().value() == ExprType::Split) {
      contig_merge_info.zero_root_ids.clear();
      contig_merge_info.last_root_id = nullptr;
      return contig_merge_info;
    }

    for (auto inp : expr->inputs()) {
      if (inp->getValType().value() == ValType::IterDomain) {
        expr_inputs.emplace(inp->as<IterDomain>());
      }
    }
    for (auto out : expr->outputs()) {
      if (out->getValType().value() == ValType::IterDomain) {
        expr_outputs.emplace(out->as<IterDomain>());
      }
    }
  }

  expr_outputs.erase(merge->out()->as<IterDomain>());

  for (auto val : root_domain) {
    expr_inputs.erase(val->as<IterDomain>());
  }

  std::unordered_set<IterDomain*> inp_out_diff;
  std::set_symmetric_difference(
      expr_inputs.begin(),
      expr_inputs.end(),
      expr_outputs.begin(),
      expr_outputs.end(),
      std::inserter(inp_out_diff, inp_out_diff.begin()));

  if (inp_out_diff.empty()) {
    contig_merge_info.mergeable = true;
  } else {
    contig_merge_info.zero_root_ids.clear();
    contig_merge_info.last_root_id = nullptr;
  }
  return contig_merge_info;
}

} // namespace

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

  auto ind = kir::addExpr(
      kir::mulExpr(outer_ind, kir::lowerValue(split->factor())), inner_ind);
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

  const auto mergable_info =
      isContiguousMerge(merge, td_->rootDomain(), root_contiguity_);

  if (mergable_info.mergeable) {
    TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(out_ind));
    index_map_[mergable_info.last_root_id] = out_ind;
    for (auto root_ind : mergable_info.zero_root_ids) {
      index_map_[root_ind] = new kir::Int(0);
    }
    return;
  }

  Val* extent = kir::lowerValue(inner_id->extent());
  index_map_[outer_id] = kir::divExpr(out_ind, extent);
  index_map_[inner_id] = kir::modExpr(out_ind, extent);
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
    const TensorDomain* _td,
    const std::vector<Val*>& indices,
    std::vector<bool> _root_contiguity)
    : td_(_td), root_contiguity_(std::move(_root_contiguity)) {
  if (td_->nDims() == 0 || indices.empty()) {
    indices_.push_back(new kir::Int(0));
    return;
  }

  const bool exclude_reduction = td_->nDims() > indices.size();

  TORCH_INTERNAL_ASSERT(
      td_->noReductions().size() == indices.size() ||
          td_->nDims() == indices.size(),
      "For IndexCompute the number of axes should match the number of dimensions in the TensorDomain.");

  {
    size_t i = 0;
    for (auto id : td_->domain()) {
      if (exclude_reduction && id->isReduction())
        continue;
      TORCH_CHECK(kir::isLoweredScalar(indices[i]));
      index_map_[id] = indices[i++];
    }
  }

  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  // Run the split/merge operations backwards. This will modify the index_map_
  // so it can be used to index the root TensorDomain. Each entry in the root
  // TensorDomain should have an entry in index_map_ We might not want to run
  // these indices at the root of the domain, but actually at the rfactor root.
  // Fortunately we can run them all the way back, but grab the indices from the
  // map at the rfactor IterDomains.
  traverseFrom(indices[0]->fusion(), domain_vals, false);

  for (auto id : td_->rootDomain()) {
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
    const std::vector<Val*>& _indices,
    const std::vector<bool>& _root_contiguity) {
  IndexCompute ic(td, _indices, _root_contiguity);
  return ic.indices_;
}

std::vector<bool> IndexCompute::contiguityAnd(
    const std::vector<bool>& contig1,
    const std::vector<bool>& contig2) {
  TORCH_INTERNAL_ASSERT(
      contig1.size() == contig2.size(),
      "Called contiguityAnd with mismatched vectors.");

  std::vector<bool> contig_result;
  std::transform(
      contig1.begin(),
      contig1.end(),
      contig2.begin(),
      std::back_inserter(contig_result),
      std::logical_and<>());
  return contig_result;
}

std::vector<bool> IndexCompute::contiguityPasC(
    TensorDomain* producer,
    TensorDomain* consumer) {
  const std::vector<bool>& producer_contiguity = producer->contiguity();
  std::vector<bool> as_consumer_contiguity;

  auto c_root = consumer->rootDomain();
  auto p_root = producer->rootDomain();

  size_t p_ind = 0;
  size_t c_ind = 0;
  while (p_ind < p_root.size()) {
    if (p_root[p_ind]->isReduction()) {
      p_ind++;
    } else if (
        c_root[c_ind]->isBroadcast() &&
        p_root[p_ind]->getIterType() != c_root[c_ind]->getIterType()) {
      c_ind++;
      as_consumer_contiguity.push_back(false);
    } else {
      as_consumer_contiguity.push_back(producer_contiguity[p_ind]);
      c_ind++;
      p_ind++;
    }
  }

  while (c_ind < c_root.size()) {
    as_consumer_contiguity.push_back(false);
    c_ind++;
  }

  return as_consumer_contiguity;
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
  // reduction axes. We have to do the indexing based on consumer because we
  // could hit instances where we have a loop nest generated based on:
  // consumer[b{1}, i1, i2] with consumer->merge(0) => consumer[b{1}*i1, i2],
  // but producer would just be producer[i1, i2]. It would be very hard to
  // generate indices directly on producer, but if we do it on consumer, and
  // grab the root axes we need (i1 and i2), it's easy to do.
  const std::vector<Val*> c_inds = IndexCompute::get(
      consumer->domain(),
      indices,
      IndexCompute::contiguityPasC(producer->domain(), consumer->domain()));

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
          if (p_root[it_p]->getIterType() == IterType::BroadcastWithStride) {
            p_inds.push_back(new kir::Int(0));
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

  bool inner_most_dim_contig =
      producer->getRootDomain()[producer->getRootDomain().size() - 1]
              ->getIterType() == IterType::Iteration &&
      producer->domain()->contiguity()[producer->getRootDomain().size() - 1];

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < p_inds.size(); i++) {
    if (p_inds[i]->isZeroInt()) {
      strided_inds.push_back(p_inds[i]);
    } else if (i == p_inds.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(p_inds[i]);
    } else {
      std::stringstream ss;
      ss << "T" << producer->name() << ".stride[" << i << "]";
      strided_inds.push_back(kir::mulExpr(
          p_inds[i], new kir::NamedScalar(ss.str(), DataType::Int)));
    }
  }

  // Probably shouldn't ever hit this
  if (strided_inds.size() == 0) {
    strided_inds.push_back(new kir::Int(0));
  }

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

  std::vector<kir::IterDomain*> ranges(loops_adjusted.size());
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
        return fl->iter_domain()->isBroadcast() ? new kir::Int(0) : fl->index();
      });

  std::vector<Val*> used_inds;
  std::vector<kir::IterDomain*> used_ranges;
  bool unrolled = false;
  for (size_t i = 0; i < loops_adjusted.size(); i++) {
    if (ranges[i]->getParallelType() == ParallelType::Unroll)
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
    for (size_t j = i + 1; j < used_ranges.size(); j++) {
      ind = kir::mulExpr(ind, used_ranges[j]->extent());
    }
    used_inds[i] = ind;
  }
  if (used_inds.size() == 0) {
    used_inds.push_back(new kir::Int(0));
  }

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

  std::vector<Val*> computed_inds = IndexCompute::get(
      consumer->domain(), indices, consumer->domain()->contiguity());

  auto root_dom = consumer->getRootDomain();
  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == TensorDomain::noReductions(root_dom).size() ||
          computed_inds.size() == root_dom.size(),
      "Dimensionality error in code generator while computing indexing.");

  // Remove indices associated with reductions.
  if (computed_inds.size() == root_dom.size()) {
    for (size_t i = 0; i < root_dom.size(); i++) {
      // Do this backwards so erase offset will be right
      auto axis = root_dom.size() - i - 1;
      if (root_dom[axis]->isReduction())
        computed_inds.erase(computed_inds.begin() + axis);
    }
  }

  // Number of root dims that are broadcasted
  size_t implicit_bcast_dims = 0;

  {
    size_t root_i = 0, inds_i = 0;
    while (root_i < root_dom.size() && inds_i < computed_inds.size()) {
      if (root_dom[root_i]->isReduction()) {
        root_i++;
      } else {
        if (root_dom[root_i]->getIterType() ==
            IterType::BroadcastWithoutStride) {
          computed_inds.erase(computed_inds.begin() + inds_i);
          root_i++;
          implicit_bcast_dims++;
        } else {
          if (root_dom[root_i]->getIterType() ==
              IterType::BroadcastWithStride) {
            computed_inds[inds_i] = new kir::Int(0);
          }
          root_i++;
          inds_i++;
        }
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      computed_inds.size() ==
              TensorDomain::noReductions(root_dom).size() -
                  implicit_bcast_dims ||
          computed_inds.size() == root_dom.size() - implicit_bcast_dims,
      "Dimensionality error in code generator while computing tensor indices.");

  bool inner_most_dim_contig =
      consumer->getRootDomain()[consumer->getRootDomain().size() - 1]
              ->getIterType() == IterType::Iteration &&
      consumer->domain()->contiguity()[consumer->getRootDomain().size() - 1];

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < computed_inds.size(); i++) {
    if (computed_inds[i]->isZeroInt()) {
      strided_inds.push_back(computed_inds[i]);
    } else if (i == computed_inds.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(computed_inds[i]);
    } else {
      std::stringstream ss;
      ss << "T" << consumer->name() << ".stride[" << i << "]";
      strided_inds.push_back(kir::mulExpr(
          computed_inds[i], new kir::NamedScalar(ss.str(), DataType::Int)));
    }
  }

  // Probably shouldn't ever hit this
  if (strided_inds.size() == 0) {
    strided_inds.push_back(new kir::Int(0));
  }

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

  std::vector<kir::IterDomain*> ranges(loops.size());
  std::transform(
      loops.begin(), loops.end(), ranges.begin(), [](kir::ForLoop* fl) {
        return fl->iter_domain();
      });

  std::vector<Val*> indices(loops.size());
  std::transform(
      loops.begin(), loops.end(), indices.begin(), [](kir::ForLoop* fl) {
        return fl->iter_domain()->isBroadcast() ? new kir::Int(0) : fl->index();
      });

  std::vector<Val*> used_inds;
  std::vector<kir::IterDomain*> used_ranges;
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
      if (ranges[l_i]->getParallelType() == ParallelType::Unroll)
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

      TORCH_CHECK(kir::isLoweredScalar(indices[l_i]));
      used_inds.push_back(indices[l_i]);
      used_ranges.push_back(ranges[l_i]);
      l_i++;
      c_i++;
    }
  }

  for (size_t i = 0; i < used_inds.size(); i++) {
    Val* ind = used_inds[i];
    for (size_t j = i + 1; j < used_ranges.size(); j++) {
      ind = kir::mulExpr(ind, used_ranges[j]->extent());
    }
    used_inds[i] = ind;
  }

  if (used_inds.size() == 0) {
    used_inds.push_back(new kir::Int(0));
  }

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

  if (producer->getMemoryType() == MemoryType::Global) {
    return getGlobalProducerIndex(producer, consumer, loops);
  }

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

  if (consumer->getMemoryType() == MemoryType::Global) {
    return getGlobalConsumerIndex(consumer, loops);
  }

  return getConsumerIndex_impl(consumer, loops);
}

} // namespace fuser
} // namespace jit
} // namespace torch
