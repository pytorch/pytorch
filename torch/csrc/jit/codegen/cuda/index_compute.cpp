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

  const auto mergable_info =
      isContiguousMerge(merge, td_->rootDomain(), root_contiguity_);

  if (mergable_info.mergeable) {
    index_map_[mergable_info.last_root_id] = out_ind;
    for (auto root_ind : mergable_info.zero_root_ids) {
      index_map_[root_ind] = new Int(0);
    }
    return;
  }

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
    const TensorDomain* _td,
    const std::vector<Val*>& indices,
    std::vector<bool> _root_contiguity)
    : td_(_td), root_contiguity_(std::move(_root_contiguity)) {
  if (td_->nDims() == 0 || indices.empty()) {
    indices_.push_back(new Int(0));
    return;
  }

  // We may or may not have indices associated with reductions.
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
    if (exclude_reduction && id->isReduction()) {
      continue;
    } else if (id->getIterType() == IterType::BroadcastWithStride) {
      indices_.push_back(new Int(0));
    } else {
      auto it = index_map_.find(id);
      TORCH_INTERNAL_ASSERT(
          it != index_map_.end(),
          "Error during index compute, missed computing a value.");
      indices_.push_back(it->second);
    }
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

// TODO: use new mapping functions
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

namespace {
// Note returned vector is relative to the producer root (rfactor domain is
// taken here for producer if there is one as this is the root domain relative
// to consumer)
//
// Consider: T3[ iS{( 1 * i5 )}, iS{i7} ] compute_at( 4, 1 )
//                 = broadcast( T1[ iS{i5}, iS{i7} ] )
// which could generate the loop nest:
//   for(size_t i18 = 0; i18 < ( T4.size[0] * T4.size[1] ); ++i18 ) {
//     float T3[T1.size[2]];
//     for(size_t i19 = 0; i19 < T3.size[2]; ++i19 ) {
//       T2[ 0 ]
//         = T1[...];
//
// Here the first dimension to index T1 must be i18 % T4.size[1], because T3 has
// a dimension being broadcasted to the extent of T4. This function is looking
// for these types of cases: where there's a dimension merged into an entry into
// consumer->domain(), but this dimension does not exist in producer_root.
// Then we need these dimensions mapped to producer_root, so we know which ones
// we need to access with modulo. We could go consumer->domain() =>
// consumer->rootDomain() => producer_root however producer_root could =
// producer->rfactorDomain() then we still might have to map to
// producer->rootDomain(). Therefore we might as well go consumer->domain() =>
// producer->domain() => producer->rootDomain().
std::vector<bool> getBCastMergedIndices(
    const TensorDomain* producer,
    const TensorDomain* consumer) {
  auto c_root = consumer->rootDomain();
  auto p_root = producer->hasRFactor() ? producer->rfactorDomain()
                                       : producer->rootDomain();

  auto root_c2p_idmap = TensorDomain::mapRootCtoP(consumer, producer);

  std::unordered_set<IterDomain*> bcast_not_in_P;
  for (auto c_id : c_root) {
    if (c_id->isBroadcast() &&
        root_c2p_idmap.find(c_id) == root_c2p_idmap.end()) {
      bcast_not_in_P.emplace(c_id);
    }
  }

  // If there are no broadcasts in consumer_root that are not in producer_root,
  // we have nothing to track here.
  if (bcast_not_in_P.empty()) {
    return std::vector<bool>(p_root.size(), false);
  }

  // We want to know what domains in consumer have a merged root broadcast
  // domain not present in producer root. We then want to map that to the
  // consumer_root axes impacted by this (the non-bcast axes merged with these
  // bcast axes). Then we want to map this to producer_root.

  std::vector<bool> c_bcast_merged(consumer->nDims(), false);

  for (size_t c_i = 0; c_i < consumer->nDims(); c_i++) {
    auto c_id = consumer->axis(c_i);
    bool missing_bcast = false;
    bool non_missing_bcast = false;

    auto c_id_inps = ir_utils::iterDomainInputsOf({c_id});

    for (auto inp : c_id_inps) {
      if (bcast_not_in_P.find(inp) != bcast_not_in_P.end()) {
        missing_bcast = true;
      } else {
        non_missing_bcast = true;
      }
    }

    // this domain c_i is guilty.
    c_bcast_merged[c_i] = missing_bcast && non_missing_bcast;
  }

  // If these missing axes aren't merged with non-missing axes, we have nothing
  // to track here.
  if (std::none_of(c_bcast_merged.begin(), c_bcast_merged.end(), [](bool b) {
        return b;
      })) {
    return std::vector<bool>(p_root.size(), false);
  }

  // map c_bcast_merged to producer
  std::vector<bool> p_bcast_merged(producer->nDims(), false);
  std::vector<std::pair<int, int>> pc_map =
      TensorDomain::mapDomainPandC(consumer->domain(), producer->domain());

  for (std::pair<int, int> entry : pc_map) {
    int p_i = entry.first;
    int c_i = entry.second;
    p_bcast_merged[p_i] = c_bcast_merged[c_i];
  }

  // map p_bcast_merged to producer->root
  std::vector<bool> p_root_bcast_merged(producer->nDims(), false);
  // map producer root IterDomain to it's position in producer->rootDomain()
  std::unordered_map<IterDomain*, int> p_root_id_to_index;
  for (size_t p_i = 0; p_i < producer->rootDomain().size(); p_i++) {
    p_root_id_to_index[producer->rootDomain()[p_i]] = p_i;
  }

  for (size_t p_i = 0; p_i < p_bcast_merged.size(); p_i++) {
    if (!p_bcast_merged[p_i])
      continue;
    IterDomain* id = producer->axis((int)p_i);
    auto id_inps = ir_utils::iterDomainInputsOf({id});
    for (auto inp : id_inps) {
      p_root_bcast_merged[p_root_id_to_index.at(inp)] = true;
    }
  }

  return p_root_bcast_merged;
}
} // namespace
kir::TensorIndex* Index::getGlobalProducerIndex(
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  // producer_tv->domain() is not replayed as the loop strucutre we were
  // provided, so replay it to match consumer_tv which is.
  auto producer = TransformReplay::replayPasC(
                      producer_tv->domain(), consumer_tv->domain(), -1)
                      .first;

  std::vector<int> p2c(producer->nDims(), false);
  auto pc_map = TensorDomain::mapDomainPandC(
      producer->domain(), consumer_tv->domain()->domain());
  for (auto entry : pc_map) {
    int p_i = entry.first;
    int c_i = entry.second;
    p2c[p_i] = c_i;
  }

  std::vector<Val*> indices;
  for (size_t i = 0; i < producer->domain().size(); i++) {
    indices.push_back(loops[p2c[i]]->index());
  }

  std::vector<Val*> computed_inds =
      IndexCompute::get(producer, indices, producer_tv->domain()->contiguity());

  auto p_root = TensorDomain::noReductions(producer->rootDomain());

  {
    // remove implicit bcast dims from root
    std::vector<IterDomain*> without_implicit_bcast;

    size_t implicit_bcast_dims = 0;
    for (auto id : p_root) {
      if (id->getIterType() != IterType::BroadcastWithoutStride) {
        without_implicit_bcast.push_back(id);
      }
    }
    p_root = without_implicit_bcast;
  }

  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == p_root.size(),
      "Dimensionality error in code generator while computing tensor indices.");

  bool inner_most_dim_contig =
      p_root[p_root.size() - 1]->getIterType() == IterType::Iteration &&
      producer->contiguity()[p_root.size() - 1];

  // This function is projected as consumer->domain() => consumer->rootDomain()
  // => producer->rootDomain()
  auto p_root_bcast_merged =
      getBCastMergedIndices(producer, consumer_tv->domain());

  std::vector<Val*> strided_inds;
  for (size_t p_i = 0; p_i < p_root.size(); p_i++) {
    Val* extent = nullptr;
    if (computed_inds[p_i]->isZeroInt()) {
      // If collapsing a dim, but need to module, we need extents multiplied
      // together
      if (p_root[p_i]->getIterType() == IterType::Iteration) {
        if (extent == nullptr) {
          extent = p_root[p_i]->extent();
        } else {
          extent = mul(extent, p_root[p_i]->extent());
        }
      }
      continue;
    }

    auto maybe_modulo = computed_inds[p_i];
    if (p_root_bcast_merged[p_i]) {
      maybe_modulo =
          mod(computed_inds[p_i],
              extent == nullptr ? p_root[p_i]->extent()
                                : mul(extent, p_root[p_i]->extent()));
    }

    if (p_i == computed_inds.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(maybe_modulo);
    } else {
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << p_i << "]";
      strided_inds.push_back(
          mul(maybe_modulo, new NamedScalar(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new kir::TensorIndex(producer_tv, strided_inds);
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

  std::vector<Val*> computed_inds = IndexCompute::get(
      consumer->domain(), indices, consumer->domain()->contiguity());

  auto root_dom = consumer->getRootDomain();

  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == root_dom.size() ||
          computed_inds.size() == TensorDomain::noReductions(root_dom).size(),
      "Dimensionality error in code generator while computing indexing.");
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
            computed_inds[inds_i] = new Int(0);
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
      continue;
    } else if (i == computed_inds.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(computed_inds[i]);
    } else {
      std::stringstream ss;
      ss << "T" << consumer->name() << ".stride[" << i << "]";
      strided_inds.push_back(
          mul(computed_inds[i], new NamedScalar(ss.str(), DataType::Int)));
    }
  }

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
