#include <torch/csrc/jit/codegen/cuda/index_compute.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_reference_replay.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// A merge is contiguous if:
//   Inputs of outer are to the left in the root domain of the inputs of RHS.
//   All inputs are contiguous in the root domain:
//     - All marked as contiguous
//     - Only gaps between inputs are broadcast or reductoin dims
//   There are no split transformations performed on outer or inner
//   All transformations on outer or inner are contiguous merges
// If this criteria holds, then we can index the input root domains of this
// merge with the indexing provided to the output of the merge in the backward
// index pass

class ContigIDs : public OptInDispatch {
 private:
  using OptInDispatch::handle;

  // Mark if ids are result of contigous merges
  std::unordered_set<kir::IterDomain*> contig_ids;
  // Given contiguous domain, return all iter domains within its history.
  std::unordered_map<kir::IterDomain*, std::unordered_set<kir::IterDomain*>>
      within_contig_ids;
  const std::vector<IterDomain*>& root_domain_;
  const std::vector<bool>& root_contiguity_;
  std::unordered_map<IterDomain*, bool> is_contig_root;

  bool inRoot(const std::vector<IterDomain*>& ids) {
    return std::all_of(ids.begin(), ids.end(), [this](IterDomain* id) {
      return is_contig_root.find(id) != is_contig_root.end();
    });
  }

  bool isContig(kir::IterDomain* id) {
    return contig_ids.find(id) != contig_ids.end();
  }

  // Split outputs are not conitguous, don't need to do anything.
  void handle(Split*) override {}

  void handle(Merge* merge) override {
    const auto gpu_lower = GpuLower::current();

    // If either input is non-contiguous so is output.
    const auto inner = merge->inner();
    const auto outer = merge->outer();
    if (!isContig(gpu_lower->lowerValue(inner)->as<kir::IterDomain>()) ||
        !isContig(gpu_lower->lowerValue(outer)->as<kir::IterDomain>())) {
      return;
    }

    // Grab inputs, make sure they're in root domain, check if they're
    // contiguous.

    auto lhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({outer}, root_domain_);
    auto rhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({inner}, root_domain_);

    TORCH_INTERNAL_ASSERT(
        inRoot(lhs_inputs) && inRoot(rhs_inputs),
        "Found an invalid merge operation, inputs of its arguments are not in the root domain.");

    std::deque<IterDomain*> ordered_inputs(
        lhs_inputs.begin(), lhs_inputs.end());
    ordered_inputs.insert(
        ordered_inputs.end(), rhs_inputs.begin(), rhs_inputs.end());

    // If any root input is not contig, output is not contig
    if (!(std::all_of(
            ordered_inputs.begin(),
            ordered_inputs.end(),
            [this](IterDomain* id) {
              return is_contig_root.at(id) && !id->isBroadcast() &&
                  !id->isReduction();
            }))) {
      return;
    }

    std::deque<IterDomain*> root_copy(root_domain_.begin(), root_domain_.end());

    // Forward to first matching argument
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() != ordered_inputs.front()) {
        root_copy.pop_front();
      } else {
        break;
      }
    }

    // Forward through all matching arguments
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() == ordered_inputs.front()) {
        root_copy.pop_front();
        ordered_inputs.pop_front();
        // This is no longer causing an error in:
        // ReductionSchedulerMultiDimNonFastest TODO: test reenablement to make
        // sure it does what's expected
        //  } else if (
        //     root_copy.front()->isReduction() ||
        //     root_copy.front()->isBroadcast()) {
        //   root_copy.pop_front();
      } else {
        break;
      }
    }

    // If we matched all inputs, the output is contiguous. Only want to keep the
    // top contig ID, lower ids should be placed in the "within_contig_ids" map
    // of top id.
    auto kir_inner =
        gpu_lower->lowerValue(merge->inner())->as<kir::IterDomain>();
    auto kir_outer =
        gpu_lower->lowerValue(merge->outer())->as<kir::IterDomain>();
    auto kir_out = gpu_lower->lowerValue(merge->out())->as<kir::IterDomain>();
    if (ordered_inputs.empty()) {
      if (contig_ids.find(kir_inner) != contig_ids.end()) {
        contig_ids.erase(kir_inner);
      }

      if (contig_ids.find(kir_outer) != contig_ids.end()) {
        contig_ids.erase(kir_outer);
      }

      contig_ids.emplace(kir_out);

      std::unordered_set<kir::IterDomain*> within_out;
      within_out.emplace(kir_inner);
      if (within_contig_ids.find(kir_inner) != within_contig_ids.end()) {
        auto in_inner = within_contig_ids.at(kir_inner);
        within_out.insert(in_inner.begin(), in_inner.end());
        within_contig_ids.erase(kir_inner);
      }

      within_out.emplace(kir_outer);
      if (within_contig_ids.find(kir_outer) != within_contig_ids.end()) {
        auto in_outer = within_contig_ids.at(kir_outer);
        within_out.insert(in_outer.begin(), in_outer.end());
        within_contig_ids.erase(kir_outer);
      }

      within_contig_ids[kir_out] = within_out;
    }
  }

 public:
  ContigIDs() = delete;

  // Check through thie history of ids whose inputs map to root_domain with
  // contiguity root_contiguity. Return unordered_set of all merges that are
  // contiguous.
  ContigIDs(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& root_domain,
      const std::vector<bool>& root_contiguity)
      : root_domain_(root_domain), root_contiguity_(root_contiguity) {
    if (ids.empty()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        root_domain_.size() == root_contiguity_.size(),
        "Arguments don't match ",
        root_domain_.size(),
        " != ",
        root_contiguity_.size());

    const auto gpu_lower = GpuLower::current();

    for (size_t i = 0; i < root_domain_.size(); i++) {
      if (root_contiguity_[i]) {
        auto kir_root_domain_i =
            gpu_lower->lowerValue(root_domain_[i])->as<kir::IterDomain>();
        contig_ids.emplace(kir_root_domain_i);
        within_contig_ids[kir_root_domain_i] =
            std::unordered_set<kir::IterDomain*>();
      }
      is_contig_root[root_domain_[i]] = root_contiguity_[i];
    }

    auto exprs = ExprSort::getExprs(ids[0]->fusion(), {ids.begin(), ids.end()});

    for (auto expr : exprs) {
      handle(expr);
    }
  }

  const std::unordered_set<kir::IterDomain*> contigIDs() const {
    return contig_ids;
  }

  const std::
      unordered_map<kir::IterDomain*, std::unordered_set<kir::IterDomain*>>
      withinContigIDs() const {
    return within_contig_ids;
  }
};

} // namespace

void IndexCompute::handle(Split* split) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto in_id = gpu_lower->lowerValue(split->in())->as<kir::IterDomain>();
  auto outer_id = gpu_lower->lowerValue(split->outer())->as<kir::IterDomain>();
  auto inner_id = gpu_lower->lowerValue(split->inner())->as<kir::IterDomain>();

  auto outer_it = index_map_.find(outer_id);
  auto inner_it = index_map_.find(inner_id);
  if (outer_it == index_map_.end() || inner_it == index_map_.end())
    return;

  const auto outer_ind = outer_it->second;
  const auto inner_ind = inner_it->second;

  const bool outer_zero = outer_ind->isZeroInt();
  const bool inner_zero = inner_ind->isZeroInt();

  const bool outer_bcast = outer_id->isBroadcast();
  const bool inner_bcast = inner_id->isBroadcast();

  const bool outer_vect =
      split->outer()->getParallelType() == ParallelType::Vectorize;
  const bool inner_vect =
      split->inner()->getParallelType() == ParallelType::Vectorize;

  // We want to mark as zero merged in if we're working with shared or local
  // memory, and the dimension we're working with is not part of the allocation,
  // as we have special propagation rules for that scenario. If zero indexing is
  // from a vectorized ID or broadcast do not propagate in zero merged manner,
  // so don't mark. This logic is important for vector support on global memory.

  // Maybe clear in_id as it could have been mapped over from another
  // IndexCompute. Uncertain if this is needed but seems to be safe.
  bool zero_merged_in = hasZeroMerged(in_id);
  zero_merged_in =
      zero_merged_in || hasZeroMerged(inner_id) || hasZeroMerged(outer_id);
  zero_merged_in =
      zero_merged_in || (outer_zero && (!outer_bcast && !outer_vect));
  zero_merged_in =
      zero_merged_in || (inner_zero && (!inner_bcast && !inner_vect));

  if (zero_merged_in) {
    zero_merged_in_.emplace(in_id);
  }
  if (zero_merged_in && outer_zero && inner_zero) {
    index_map_[in_id] = ir_builder.create<kir::Int>(0);
    extent_map_[in_id] = ir_builder.create<kir::Int>(0);
  } else if (zero_merged_in && outer_zero) {
    index_map_[in_id] = inner_ind;
    extent_map_[in_id] = getExtent(inner_id);
  } else if (zero_merged_in && inner_zero) {
    index_map_[in_id] = outer_ind;
    extent_map_[in_id] = getExtent(outer_id);
  } else {
    index_map_[in_id] = ir_builder.addExpr(
        ir_builder.mulExpr(outer_ind, getExtent(inner_id)), inner_ind);
    if (extent_map_.find(outer_id) != extent_map_.end() ||
        extent_map_.find(inner_id) != extent_map_.end()) {
      extent_map_[in_id] =
          ir_builder.mulExpr(getExtent(outer_id), getExtent(inner_id));
    }
  }
}

void IndexCompute::handle(Merge* merge) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto out_id = gpu_lower->lowerValue(merge->out())->as<kir::IterDomain>();
  auto outer_id = gpu_lower->lowerValue(merge->outer())->as<kir::IterDomain>();
  auto inner_id = gpu_lower->lowerValue(merge->inner())->as<kir::IterDomain>();

  auto out_it = index_map_.find(out_id);
  if (out_it == index_map_.end()) {
    return;
  }
  auto out_ind = out_it->second;

  auto zero = ir_builder.create<kir::Int>(0);

  if (out_ind->isZeroInt()) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = zero;
    extent_map_[outer_id] = zero;
    extent_map_[inner_id] = zero;
    return;
  }

  if (!hasZeroMerged(out_id) && contig_ids.find(out_id) != contig_ids.end()) {
    // Contiguous indexing path
    auto input_ids = ir_utils::iterDomainInputsOfOrderedAs(
        {merge->out()}, td_->getRootDomain());

    // Shouldn't hit this, but don't want to segfault if somehow we do.
    TORCH_INTERNAL_ASSERT(!input_ids.empty());

    for (auto root_id : input_ids) {
      index_map_[gpu_lower->lowerValue(root_id)->as<kir::IterDomain>()] = zero;
    }

    index_map_[gpu_lower->lowerValue(*(input_ids.end() - 1))
                   ->as<kir::IterDomain>()] = out_ind;
    return;
  }

  const auto inner_extent = getExtent(inner_id);
  const auto outer_extent = getExtent(outer_id);

  if (inner_id->isBroadcast() && inner_extent->isOneInt()) {
    // Propagate away from broadcast dims
    index_map_[outer_id] = out_ind;
    index_map_[inner_id] = zero;

    extent_map_[outer_id] = getExtent(out_id);
  } else if (outer_id->isBroadcast() && outer_extent->isOneInt()) {
    // Propagate away from broadcast dims
    index_map_[outer_id] = zero;
    index_map_[inner_id] = out_ind;

    extent_map_[inner_id] = getExtent(out_id);
  } else if (hasZeroMerged(out_id)) {
    // Don't propagate to inner id if it's comprised of only broadcast root
    // domains, unless outer is also all broadcast domains. Index shouldn't be
    // anything but zero if both inner and outer are all broadcast domains, but
    // didn't add a hard check for this. See FusionAdvancedIndexing5_CUDA
    if (!inner_id->isBroadcast() && !outer_id->isBroadcast()) {
      // If neither dimension is a broadcast (should be true for reference
      // indexing) pick the preferred path or the inner path.
      if (preferred_paths_.find(outer_id) != preferred_paths_.end() &&
          preferred_paths_.find(inner_id) == preferred_paths_.end()) {
        // Marked that we should prop through outer, not inner.
        index_map_[outer_id] = out_ind;
        extent_map_[outer_id] = getExtent(out_id);
        index_map_[inner_id] = zero;
        extent_map_[inner_id] = zero;
      } else {
        // Prop through inner
        index_map_[inner_id] = out_ind;
        extent_map_[inner_id] = getExtent(out_id);
        index_map_[outer_id] = zero;
        extent_map_[outer_id] = zero;
      }
    } else if (inner_id->isBroadcast() && !outer_id->isBroadcast()) {
      // Inner is broadcast and outer isn't, prop through outer
      index_map_[outer_id] = out_ind;
      extent_map_[outer_id] = getExtent(out_id);
      index_map_[inner_id] = zero;
      extent_map_[inner_id] = zero;
    } else {
      // Default to propagating through inner
      index_map_[inner_id] = out_ind;
      extent_map_[inner_id] = getExtent(out_id);
      index_map_[outer_id] = zero;
      extent_map_[outer_id] = zero;
    }
    zero_merged_in_.emplace(inner_id);
    zero_merged_in_.emplace(outer_id);
  } else {
    index_map_[outer_id] = ir_builder.divExpr(out_ind, inner_extent);
    index_map_[inner_id] = ir_builder.modExpr(out_ind, inner_extent);
  }
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

// Otherwise warning on runBackward as it hides an overloaded virtual
// using TransformIter::runBackward;
IndexCompute::IndexCompute(
    const TensorDomain* _td,
    std::unordered_map<kir::IterDomain*, kir::Val*> initial_index_map,
    std::unordered_map<kir::IterDomain*, kir::Val*> extent_map,
    std::unordered_set<kir::IterDomain*> zero_merged_in,
    const std::vector<bool>& root_contiguity,
    std::unordered_set<kir::IterDomain*> preferred_paths)
    : td_(_td),
      index_map_(std::move(initial_index_map)),
      extent_map_(std::move(extent_map)),
      zero_merged_in_(std::move(zero_merged_in)),
      preferred_paths_(std::move(preferred_paths)) {
  FUSER_PERF_SCOPE("IndexCompute::IndexCompute");

  // Make sure we recompute any indices we can that map to a contiguous access
  // in physical memory.
  if (std::any_of(root_contiguity.begin(), root_contiguity.end(), [](bool b) {
        return b;
      })) {
    ContigIDs contig_finder(
        td_->domain(), td_->getRootDomain(), root_contiguity);
    contig_ids = contig_finder.contigIDs();
    auto within_contig = contig_finder.withinContigIDs();
    for (auto contig_id : contig_ids) {
      if (index_map_.find(contig_id) != index_map_.end()) {
        TORCH_INTERNAL_ASSERT(
            within_contig.find(contig_id) != within_contig.end());
        for (auto id : within_contig.at(contig_id)) {
          index_map_.erase(id);
        }
      }
    }
  }
}

void IndexCompute::run() {
  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  traverseFrom(td_->fusion(), domain_vals, false);
}

kir::Val* IndexCompute::getExtent(kir::IterDomain* id) {
  if (extent_map_.find(id) != extent_map_.end()) {
    return extent_map_.at(id);
  } else {
    return id->extent();
  }
}

bool IndexCompute::hasZeroMerged(kir::IterDomain* id) {
  return zero_merged_in_.find(id) != zero_merged_in_.end();
}

IndexCompute IndexCompute::updateIndexCompute(
    const TensorDomain* new_td,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    const std::vector<bool>& root_contiguity) {
  FUSER_PERF_SCOPE("updateIndexCompute");

  const auto gpu_lower = GpuLower::current();

  std::unordered_map<kir::IterDomain*, kir::Val*> updated_index_map;
  std::unordered_map<kir::IterDomain*, kir::Val*> updated_extent_map;
  std::unordered_set<kir::IterDomain*> updated_zero_merged_in;

  for (auto id_entry : id_map) {
    kir::IterDomain* prev_id =
        gpu_lower->lowerValue(id_entry.first)->as<kir::IterDomain>();
    kir::IterDomain* new_id =
        gpu_lower->lowerValue(id_entry.second)->as<kir::IterDomain>();

    if (index_map_.find(prev_id) != index_map_.end()) {
      updated_index_map[new_id] = index_map_.at(prev_id);
    }

    updated_extent_map[new_id] = getExtent(prev_id);

    if (zero_merged_in_.find(prev_id) != zero_merged_in_.end()) {
      updated_zero_merged_in.emplace(new_id);
    }
  }

  IndexCompute updated_index_compute(
      new_td,
      updated_index_map,
      updated_extent_map,
      updated_zero_merged_in,
      root_contiguity);
  updated_index_compute.run();

  return updated_index_compute;
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

// TODO: How does contiguity and rfactor interact?
std::vector<bool> IndexCompute::contiguityPasC(
    kir::TensorView* producer,
    kir::TensorView* consumer) {
  FUSER_PERF_SCOPE("contiguityPasC");

  auto producer_tv = producer->fuserTv();
  auto consumer_tv = consumer->fuserTv();

  const std::vector<bool>& producer_contiguity =
      producer_tv->domain()->contiguity();
  std::vector<bool> as_consumer_contiguity(
      consumer_tv->getRootDomain().size(), false);

  auto pairwiseMap = PairwiseRootDomainMap(producer_tv, consumer_tv);
  auto p2c_root_map = pairwiseMap.mapProducerToConsumer(
      producer_tv->domain(), consumer_tv->domain());

  for (size_t p_root_i = 0; p_root_i < producer_tv->getRootDomain().size();
       p_root_i++) {
    auto p_root_id = producer_tv->getRootDomain()[p_root_i];
    auto c_root_it = p2c_root_map.find(p_root_id);
    if (c_root_it == p2c_root_map.end()) {
      continue;
    }
    auto c_root_id = c_root_it->second;
    auto c_root_i = std::distance(
        consumer_tv->getRootDomain().begin(),
        std::find(
            consumer_tv->getRootDomain().begin(),
            consumer_tv->getRootDomain().end(),
            c_root_id));

    if (p_root_id->isReduction() ||
        (c_root_id->isBroadcast() &&
         p_root_id->getIterType() != c_root_id->getIterType())) {
      continue;
    } else {
      as_consumer_contiguity[c_root_i] = producer_contiguity[p_root_i];
    }
  }

  return as_consumer_contiguity;
}

namespace {
// Map indices down to the leaf domains for applying swizzle
class UpdateLeafIndices : public IterVisitor {
 public:
  UpdateLeafIndices(
      const TensorDomain* td,
      std::unordered_map<kir::IterDomain*, kir::Val*> initial_index_map,
      std::unordered_map<kir::IterDomain*, kir::Val*> extent_map)
      : td_(td),
        index_map_(std::move(initial_index_map)),
        extent_map_(std::move(extent_map)) {
    const std::vector<Val*> domain_vals(
        td_->domain().begin(), td_->domain().end());

    traverseFrom(td_->fusion(), domain_vals, false);
  }

  const std::unordered_map<kir::IterDomain*, kir::Val*>& indexMap() const {
    return index_map_;
  }

  const std::unordered_map<kir::IterDomain*, kir::Val*>& extentMap() const {
    return extent_map_;
  }

 private:
  using IterVisitor::handle;

  void handle(Split* split) override {
    const auto gpu_lower = GpuLower::current();

    auto in_id = gpu_lower->lowerValue(split->in())->as<kir::IterDomain>();
    auto outer_id =
        gpu_lower->lowerValue(split->outer())->as<kir::IterDomain>();
    auto inner_id =
        gpu_lower->lowerValue(split->inner())->as<kir::IterDomain>();

    // Nothing need to be done when mappings for the output axes
    // already exist.
    if (index_map_.find(outer_id) != index_map_.end()) {
      TORCH_INTERNAL_ASSERT(
          index_map_.find(inner_id) != index_map_.end(),
          "Outer exists but inner not found");
      return;
    }

    kir::IrBuilder ir_builder(gpu_lower->kernel());
    auto factor = gpu_lower->lowerValue(split->factor());
    index_map_[inner_id] = ir_builder.modExpr(index_map_[in_id], factor);
    extent_map_[inner_id] = factor;
    index_map_[outer_id] = ir_builder.divExpr(index_map_[in_id], factor);
    extent_map_[outer_id] = ir_builder.ceilDivExpr(getExtent(in_id), factor);
  }

  void handle(Merge* merge) override {
    const auto gpu_lower = GpuLower::current();

    auto out_id = gpu_lower->lowerValue(merge->out())->as<kir::IterDomain>();
    auto outer_id =
        gpu_lower->lowerValue(merge->outer())->as<kir::IterDomain>();
    auto inner_id =
        gpu_lower->lowerValue(merge->inner())->as<kir::IterDomain>();

    // Nothing need to be done when mappings for the output axes
    // already exist.
    if (index_map_.find(out_id) != index_map_.end()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        index_map_.find(outer_id) != index_map_.end(), "Outer ID not found");
    TORCH_INTERNAL_ASSERT(
        index_map_.find(inner_id) != index_map_.end(), "Inner ID not found");

    kir::IrBuilder ir_builder(gpu_lower->kernel());
    index_map_[out_id] = ir_builder.mulExpr(
        index_map_[inner_id],
        ir_builder.mulExpr(index_map_[outer_id], getExtent(inner_id)));

    extent_map_[out_id] =
        ir_builder.mulExpr(getExtent(outer_id), getExtent(inner_id));
  }

  // return extent_map_[id] if exists, else return id->extent()
  kir::Val* getExtent(kir::IterDomain* id) {
    if (extent_map_.find(id) != extent_map_.end()) {
      return extent_map_.at(id);
    } else {
      return id->extent();
    }
  }

 private:
  const TensorDomain* td_;
  std::unordered_map<kir::IterDomain*, kir::Val*> index_map_;
  std::unordered_map<kir::IterDomain*, kir::Val*> extent_map_;
};

} // namespace

IndexSwizzle::IndexSwizzle(
    const TensorView* tv,
    std::unordered_map<kir::IterDomain*, kir::Val*> initial_index_map,
    std::unordered_map<kir::IterDomain*, kir::Val*> extent_map,
    std::unordered_set<kir::IterDomain*> zero_merged_in)
    : IndexCompute(
          tv->domain(),
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_merged_in),
          std::vector<bool>(tv->getRootDomain().size(), false)),
      tv_(tv),
      swizzle_type_(tv->swizzleType()),
      ids_to_swizzle_(tv->axesToSwizzle()) {}

void IndexSwizzle::run() {
  TORCH_INTERNAL_ASSERT(
      swizzle_type_ == SwizzleType::NoSwizzle ||
          swizzle_type_ == SwizzleType::Transpose,
      "Invalid swizzle type");
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());
  if (swizzle_type_ == SwizzleType::Transpose) {
    // Shifts the second axis by the first axis as ((idx_1 + idx_2) %
    // ext). Alternatively, ((idx_1 - idx_2) & (ext - 1)) would also
    // work if ext is a power of two. Practically, ext should be 32 if
    // the data type of the tensor is float, so the latter approach
    // should also be fine.
    TORCH_INTERNAL_ASSERT(tv_->getMemoryType() == MemoryType::Shared);
    TORCH_INTERNAL_ASSERT(tv_->axesToSwizzle().size() == 2);

    UpdateLeafIndices update_leaves(td_, indexMap(), extentMap());
    index_map_ = update_leaves.indexMap();
    extent_map_ = update_leaves.extentMap();

    IterDomain* id_to_swizzle_i = ids_to_swizzle_.at(0);
    IterDomain* id_to_swizzle_j = ids_to_swizzle_.at(1);
    kir::IterDomain* id_to_swizzle_i_kir =
        gpu_lower->lowerValue(id_to_swizzle_i)->as<kir::IterDomain>();
    kir::IterDomain* id_to_swizzle_j_kir =
        gpu_lower->lowerValue(id_to_swizzle_j)->as<kir::IterDomain>();

    if (indexMap().find(id_to_swizzle_i_kir) != indexMap().end() &&
        indexMap().find(id_to_swizzle_j_kir) != indexMap().end()) {
      auto idx_to_swizzle_i = indexMap().at(id_to_swizzle_i_kir);
      auto idx_to_swizzle_j = indexMap().at(id_to_swizzle_j_kir);

      auto swizzled_idx = ir_builder.modExpr(
          ir_builder.addExpr(idx_to_swizzle_i, idx_to_swizzle_j),
          id_to_swizzle_j_kir->rawExtent());
      index_map_[id_to_swizzle_j_kir] = swizzled_idx;
      swizzled_ids_.insert(id_to_swizzle_j);
      IndexCompute::run();
    }
  }
}

void IndexSwizzle::handle(Expr* e) {
  auto out_ids = ir_utils::filterByType<IterDomain>(e->outputs());
  bool needs_update =
      std::any_of(out_ids.begin(), out_ids.end(), [this](IterDomain* id) {
        return swizzled_ids_.find(id) != swizzled_ids_.end();
      });
  if (!needs_update) {
    return;
  }

  IndexCompute::handle(e);
  for (auto input : ir_utils::filterByType<IterDomain>(e->inputs())) {
    swizzled_ids_.insert(input);
  }
}

kir::TensorIndex* Index::getGlobalProducerIndex(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("getGlobalProducerIndex");
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Get a reference tensor replayed as existing loop structure
  auto reference = IndexReferenceReplay::getReference(loops);
  auto reference_domain = reference.domain;
  auto reference_id_map = reference.concrete_to_id;

  // Replay producer to look like consumer so we can index on producer since our
  // loop nests look like consumer
  auto pairwiseMap = PairwiseRootDomainMap(producer_tv, consumer_tv);
  auto producerAsC =
      TransformReplay::replayPasC(
          producer_tv->domain(), consumer_tv->domain(), -1, pairwiseMap)
          .first;

  // Make the producer_tv look like consumer while performing indexing math
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // Map reference tensor to producer
  std::unordered_map<IterDomain*, IterDomain*> root_ref_to_producer;
  for (auto p_root : producer_tv->getMaybeRFactorDomain()) {
    auto concrete_id = gpu_lower->caIndexMap().getConcreteMappedID(p_root);
    auto ref_id_it = reference_id_map.find(concrete_id);
    if (ref_id_it != reference_id_map.end()) {
      root_ref_to_producer[ref_id_it->second] = p_root;
    }
  }

  // Index into the reference tensor. Reference indexing will handle vectorized
  // dims where index should be set to 0
  auto ref_compute = getReferenceIndexing(loops, reference_domain);

  // Replay producer as reference to get reference to producer ID map
  BestEffortReplay replay_producer_as_ref(
      producer_tv->domain()->domain(),
      reference_domain->domain(),
      root_ref_to_producer,
      false);

  const auto& ref_2_producer = replay_producer_as_ref.getReplay();

  // Forward vectorized IDs to index into producer correctly
  for (auto entry : ref_2_producer) {
    auto ref_id = entry.first;
    auto p_id = entry.second;
    if (ref_id->getParallelType() == ParallelType::Vectorize) {
      p_id->parallelize(ParallelType::Vectorize);
    }
  }

  // Index into producer using reference indexing
  auto producer_indexing = ref_compute.updateIndexCompute(
      producer_tv->domain(),
      ref_2_producer,
      producer_tv->domain()->contiguity());

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();

  bool inner_most_dim_contig =
      root_dom[root_dom.size() - 1]->getIterType() == IterType::Iteration &&
      producer_tv->domain()->contiguity()[root_dom.size() - 1];

  // Global striding
  int64_t stride_i = 0;
  std::vector<kir::Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
      // If the domain is derived from a trivial reduction, no indexing to
      // create. Also, the domain at this branch must not be a
      // reduction, so the stride index should be incremented.
    } else if (
        root_dom[i]->getIterType() == IterType::BroadcastWithStride ||
        gpu_lower->isDerivedFromTrivialReduction(root_dom[i])) {
      stride_i++;
      continue;
    }

    auto kir_root_dom_i =
        gpu_lower->lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        producer_indexing.indexMap().find(kir_root_dom_i) !=
            producer_indexing.indexMap().end(),
        "Couldn't find root mapping for TV",
        producer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir::toString(kir_root_dom_i));

    auto root_ind = producer_indexing.indexMap().at(kir_root_dom_i);
    if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(root_ind);
    } else if (root_ind->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(ir_builder.mulExpr(
          root_ind,
          ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(ir_builder.create<kir::Int>(0));

  return ir_builder.create<kir::TensorIndex>(producer_tv, strided_inds);
}

namespace {

// Used for local and shared index mapping
std::unordered_map<kir::ForLoop*, kir::Val*> indexMapFromTV(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::pair<kir::ForLoop*, int64_t>& alloc_point) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto alloc_loop = alloc_point.first;

  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  const auto zero = ir_builder.create<kir::Int>(0);

  const bool is_global = tv->getMemoryType() == MemoryType::Global;
  const bool is_shared = tv->getMemoryType() == MemoryType::Shared;
  const bool is_local = tv->getMemoryType() == MemoryType::Local;

  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map;

  for (auto loop : loops) {
    kir::Val* idx = nullptr;
    // See also LoopNestGenerator::pushAlloc.
    if (!within_alloc) {
      if ((loop->iter_domain()->isThreadDim() && is_shared) ||
          (loop->iter_domain()->isThread() && is_global)) {
        idx = loop->index();
      } else {
        idx = zero;
      }
    } else if (
        (loop->iter_domain()->isBlockDim() && is_shared) ||
        (loop->iter_domain()->isThread() && is_local) ||
        (loop->iter_domain()->parallelType() == ParallelType::Vectorize)) {
      idx = zero;
    } else {
      idx = loop->index();
    }

    loop_to_ind_map[loop] = idx;

    if (!within_alloc && loop == alloc_loop) {
      within_alloc = true;
    }
  }
  return loop_to_ind_map;
}

} // namespace

// Producer index for either shared or local memory
kir::TensorIndex* Index::getProducerIndex_impl(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Get a reference tensor replayed as existing loop structure
  auto reference = IndexReferenceReplay::getReference(loops);
  auto reference_domain = reference.domain;
  auto reference_id_map = reference.concrete_to_id;

  // Replay producer to look like consumer so we can index on producer since our
  // loop nests look like consumer
  auto pairwiseMap = PairwiseRootDomainMap(producer_tv, consumer_tv);
  auto producerAsC =
      TransformReplay::replayPasC(
          producer_tv->domain(), consumer_tv->domain(), -1, pairwiseMap)
          .first;

  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // Produce mapping between consumer and producer, this is used to figure out
  // the allocation point of the producer relative to the loop nests generated
  // by the consumer
  auto c2p_root_map = pairwiseMap.mapConsumerToProducer(
      consumer_tv->domain(), producer_tv->domain());

  //  We want to play producer as consumer instead of the other way around since
  //  consumer may have some broadcasted axes producer doesn't have merged into
  //  loops producer may use. If we did consumer as producer we wouldn't have
  //  this information in the mapping.
  BestEffortReplay replay_PasC(
      producer_tv->domain()->domain(),
      consumer_tv->domain()->domain(),
      c2p_root_map,
      true);

  auto c2p_map = replay_PasC.getReplay();

  // Grab consumer domain entries and reverse replay map. TODO: Maybe
  // TransformReplay::replayPasC could return this map
  decltype(c2p_map) p2c_map;
  for (auto id : consumer_tv->domain()->domain()) {
    auto c2p_it = c2p_map.find(id);
    if (c2p_it != c2p_map.end()) {
      auto c_id = c2p_it->first;
      auto p_id = c2p_it->second;
      p2c_map[p_id] = c_id;
    }
  }

  // Find allocation point of producer relative to loop nests. P2C map is
  // required because producer was replayed as consumer, so we can't use the
  // regular compute at maps to line up its iter domains with the for loops.
  auto alloc_point =
      loop_utils::getAllocPoint(producer_tv, loops, p2c_map, true);
  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map =
      indexMapFromTV(producer_tv, loops, alloc_point);

  // Map loop nests to indicies, zeroing out those not used due to locality of
  // memory
  std::unordered_map<kir::IterDomain*, kir::Val*> ref_id_to_ind_map;

  // Due to rfactor/initialization reference_domain may be bigger than loop nest
  // structure, ignore IterDomains that aren't present in the loop nest when
  // indexing reference.
  TORCH_INTERNAL_ASSERT(loops.size() <= reference_domain->nDims());
  for (size_t loop_i = 0; loop_i < loops.size(); loop_i++) {
    auto ref_axis = gpu_lower->lowerValue(reference_domain->axis(loop_i))
                        ->as<kir::IterDomain>();
    ref_id_to_ind_map[ref_axis] = loop_to_ind_map[loops[loop_i]];
  }

  // Map reference tensor to producer
  std::unordered_map<IterDomain*, IterDomain*> root_ref_to_producer;
  for (auto p_root : producer_tv->getMaybeRFactorDomain()) {
    auto concrete_id = gpu_lower->caIndexMap().getConcreteMappedID(p_root);
    auto ref_id_it = reference_id_map.find(concrete_id);
    if (ref_id_it != reference_id_map.end()) {
      root_ref_to_producer[ref_id_it->second] = p_root;
    }
  }

  // Grab roots that map into producer and save them into the preferred roots
  // set for references indexing
  std::unordered_set<IterDomain*> preferred_roots;
  for (auto entry : root_ref_to_producer) {
    if (entry.second->isBroadcast() || entry.second->isReduction()) {
      continue;
    }
    preferred_roots.emplace(entry.first);
  }

  // Make sure propagation of indexing while mixing with 0 indicies we propagate
  // in a way that the producer will be able to see what's going on (propagating
  // into common roots of reference and producer).
  auto preferred_paths = buildPreferredPaths(reference_domain, preferred_roots);

  // Index into the reference tensor
  auto ref_compute = getReferenceIndexing(
      loops, reference_domain, ref_id_to_ind_map, preferred_paths);

  // Directly replay the producer as the reference to get the mapping of
  // reference to producer we will use to map the indexing into producer
  BestEffortReplay replay_producer_as_ref(
      producer_tv->domain()->domain(),
      reference_domain->domain(),
      root_ref_to_producer,
      false);

  const auto& ref_2_producer = replay_producer_as_ref.getReplay();

  // Forward vectorized IDs to index into producer correctly
  for (auto entry : ref_2_producer) {
    auto ref_id = entry.first;
    auto p_id = entry.second;
    if (ref_id->getParallelType() == ParallelType::Vectorize) {
      p_id->parallelize(ParallelType::Vectorize);
    }
  }

  // Index into producer using reference indexing
  auto producer_indexing = ref_compute.updateIndexCompute(
      producer_tv->domain(),
      ref_2_producer,
      producer_tv->domain()->contiguity());

  IndexSwizzle index_swizzle(
      producer_tv,
      producer_indexing.indexMap(),
      producer_indexing.extentMap(),
      producer_indexing.zeroMergedIn());

  index_swizzle.run();

  auto index_map = index_swizzle.indexMap();
  auto extent_map = producer_indexing.extentMap();

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();
  std::vector<kir::Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast() ||
        gpu_lower->isDerivedFromTrivialReduction(root_dom[i])) {
      continue;
    }

    auto kir_root_dom_i =
        gpu_lower->lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        producer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir::toString(kir_root_dom_i));

    const auto root_ind_i = index_map.at(kir_root_dom_i);

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    kir::Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction() ||
          gpu_lower->isDerivedFromTrivialReduction(root_dom[j])) {
        continue;
      }

      auto kir_root_dom_j =
          gpu_lower->lowerValue(root_dom[j])->as<kir::IterDomain>();

      TORCH_INTERNAL_ASSERT(
          index_map.find(kir_root_dom_j) != index_map.end(),
          "Couldn't find root mapping for TV",
          consumer_tv->name(),
          " dim: ",
          i,
          " id: ",
          root_dom[i]);

      auto root_ind_j = index_map.at(kir_root_dom_j);
      auto root_ext_j = extent_map.find(kir_root_dom_j) == extent_map.end()
          ? kir_root_dom_j->extent()
          : extent_map.at(kir_root_dom_j);

      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = ir_builder.mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds.push_back(ir_builder.mulExpr(root_ind_i, stride));
    } else {
      strided_inds.push_back(root_ind_i);
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(ir_builder.create<kir::Int>(0));

  return ir_builder.create<kir::TensorIndex>(producer_tv, strided_inds);
}

kir::TensorIndex* Index::getGlobalConsumerIndex(
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("getGlobalConsumerIndex");
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Get a reference tensor replayed as existing loop structure
  auto reference = IndexReferenceReplay::getReference(loops);
  auto reference_domain = reference.domain;
  auto reference_id_map = reference.concrete_to_id;

  // Map reference tensor to consumer
  std::unordered_map<IterDomain*, IterDomain*> root_ref_to_consumer;
  for (auto c_root : consumer_tv->getMaybeRFactorDomain()) {
    auto concrete_id = gpu_lower->caIndexMap().getConcreteMappedID(c_root);
    auto ref_id_it = reference_id_map.find(concrete_id);
    if (ref_id_it != reference_id_map.end()) {
      root_ref_to_consumer[ref_id_it->second] = c_root;
    }
  }

  BestEffortReplay replay_consumer_as_ref(
      consumer_tv->domain()->domain(),
      reference_domain->domain(),
      root_ref_to_consumer,
      false);

  const auto& ref_2_consumer = replay_consumer_as_ref.getReplay();

  // Index into the reference tensor. Reference indexing will handle vectorized
  // dims where index should be set to 0
  auto ref_compute = getReferenceIndexing(loops, reference_domain);

  // Index into consumer using reference indexing
  auto consumer_indexing = ref_compute.updateIndexCompute(
      consumer_tv->domain(),
      ref_2_consumer,
      consumer_tv->domain()->contiguity());

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  bool inner_most_dim_contig =
      root_dom[root_dom.size() - 1]->getIterType() == IterType::Iteration &&
      consumer_tv->domain()->contiguity()[root_dom.size() - 1];

  int64_t stride_i = 0;
  std::vector<kir::Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
      // See a comment in indexing to root domains in getGlobalProducerIndex.
    } else if (
        root_dom[i]->getIterType() == IterType::BroadcastWithStride ||
        gpu_lower->isDerivedFromTrivialReduction(root_dom[i])) {
      stride_i++;
      continue;
    }

    auto kir_root_dom_i =
        gpu_lower->lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        consumer_indexing.indexMap().find(kir_root_dom_i) !=
            consumer_indexing.indexMap().end(),
        "Couldn't find root mapping for TV",
        consumer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir::toString(kir_root_dom_i));
    auto ind = consumer_indexing.indexMap().at(kir_root_dom_i);

    if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(ind);
    } else if (ind->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << consumer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(ir_builder.mulExpr(
          ind, ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(ir_builder.create<kir::Int>(0));

  return ir_builder.create<kir::TensorIndex>(consumer_tv, strided_inds);
}

// Consumer index for either shared or local memory
kir::TensorIndex* Index::getConsumerIndex_impl(
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Get a reference tensor replayed as existing loop structure
  auto reference = IndexReferenceReplay::getReference(loops);
  auto reference_domain = reference.domain;
  auto reference_id_map = reference.concrete_to_id;

  auto alloc_point = loop_utils::getAllocPoint(consumer_tv, loops);
  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map =
      indexMapFromTV(consumer_tv, loops, alloc_point);

  // Map loop nests to indicies, zeroing out those not used due to locality of
  // memory
  std::unordered_map<kir::IterDomain*, kir::Val*> ref_id_to_ind_map;

  // Due to rfactor/initialization reference_domain may be bigger than loop nest
  // structure, ignore IterDomains that aren't present in the loop nest when
  // indexing reference.
  TORCH_INTERNAL_ASSERT(loops.size() <= reference_domain->nDims());
  for (size_t loop_i = 0; loop_i < loops.size(); loop_i++) {
    auto ref_axis = gpu_lower->lowerValue(reference_domain->axis(loop_i))
                        ->as<kir::IterDomain>();
    ref_id_to_ind_map[ref_axis] = loop_to_ind_map[loops[loop_i]];
  }

  // Map reference tensor to consumer
  std::unordered_map<IterDomain*, IterDomain*> root_ref_to_consumer;
  for (auto c_root : consumer_tv->getMaybeRFactorDomain()) {
    auto concrete_id = gpu_lower->caIndexMap().getConcreteMappedID(c_root);
    auto ref_id_it = reference_id_map.find(concrete_id);
    if (ref_id_it != reference_id_map.end()) {
      root_ref_to_consumer[ref_id_it->second] = c_root;
    }
  }

  // Grab roots that map into consumer and save them into the preferred roots
  // set for references indexing
  std::unordered_set<IterDomain*> preferred_roots;
  for (auto entry : root_ref_to_consumer) {
    if (entry.second->isBroadcast() || entry.second->isReduction()) {
      continue;
    }
    preferred_roots.emplace(entry.first);
  }

  // Make sure propagation of indexing while mixing with 0 indicies we propagate
  // in a way that consumer will be able to see what's going on.
  auto preferred_paths = buildPreferredPaths(reference_domain, preferred_roots);

  // Index into the reference tensor
  auto ref_compute = getReferenceIndexing(
      loops, reference_domain, ref_id_to_ind_map, preferred_paths);

  BestEffortReplay replay_consumer_as_ref(
      consumer_tv->domain()->domain(),
      reference_domain->domain(),
      root_ref_to_consumer,
      false);

  const auto& ref_2_consumer = replay_consumer_as_ref.getReplay();

  // Index into consumer using reference indexing
  auto consumer_indexing = ref_compute.updateIndexCompute(
      consumer_tv->domain(),
      ref_2_consumer,
      consumer_tv->domain()->contiguity());

  IndexSwizzle index_swizzle(
      consumer_tv,
      consumer_indexing.indexMap(),
      consumer_indexing.extentMap(),
      consumer_indexing.zeroMergedIn());

  index_swizzle.run();

  auto index_map = index_swizzle.indexMap();
  auto extent_map = consumer_indexing.extentMap();

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();
  std::vector<kir::Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast() ||
        gpu_lower->isDerivedFromTrivialReduction(root_dom[i])) {
      continue;
    }

    auto kir_root_dom_i =
        gpu_lower->lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        consumer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir::toString(kir_root_dom_i));

    const auto root_ind_i = index_map.at(kir_root_dom_i);
    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    kir::Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction() ||
          gpu_lower->isDerivedFromTrivialReduction(root_dom[j])) {
        continue;
      }

      auto kir_root_dom_j =
          gpu_lower->lowerValue(root_dom[j])->as<kir::IterDomain>();

      TORCH_INTERNAL_ASSERT(
          index_map.find(kir_root_dom_j) != index_map.end(),
          "Couldn't find root mapping for TV",
          consumer_tv->name(),
          " dim: ",
          i,
          " id: ",
          root_dom[i]);

      auto root_ind_j = index_map.at(kir_root_dom_j);
      auto root_ext_j = extent_map.find(kir_root_dom_j) == extent_map.end()
          ? kir_root_dom_j->extent()
          : extent_map.at(kir_root_dom_j);
      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = ir_builder.mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds.push_back(ir_builder.mulExpr(root_ind_i, stride));
    } else {
      strided_inds.push_back(root_ind_i);
    }
  }

  if (strided_inds.size() == 0) {
    strided_inds.push_back(ir_builder.create<kir::Int>(0));
  }
  auto indexed = ir_builder.create<kir::TensorIndex>(consumer_tv, strided_inds);
  return indexed;
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("Index::getProducerIndex");
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  if (producer->domain()->noReductions().size() == 0) {
    return ir_builder.create<kir::TensorIndex>(
        producer, std::vector<kir::Val*>());
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
  FUSER_PERF_SCOPE("Index::getConsumerIndex");
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  if (consumer->domain()->noReductions().size() == 0) {
    return ir_builder.create<kir::TensorIndex>(
        consumer, std::vector<kir::Val*>());
  }

  if (consumer->getMemoryType() == MemoryType::Global) {
    return getGlobalConsumerIndex(consumer, loops);
  }
  return getConsumerIndex_impl(consumer, loops);
}

// Basically just copy getGlobalConsumerIndex, just don't do the striding and
// return std::vector of Vals
//
// TODO(kir): replace pair with struct
//
std::pair<std::vector<kir::Val*>, bool> Index::getConsumerRootPredIndices(
    const kir::TensorView* kir_consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::vector<bool>& root_contiguity,
    bool unswitch) {
  FUSER_PERF_SCOPE("Index::getConsumerRootPredIndices");

  auto consumer_tv = kir_consumer_tv->fuserTv();

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Get a reference tensor replayed as existing loop structure
  auto reference = IndexReferenceReplay::getReference(loops);
  auto reference_domain = reference.domain;
  auto reference_id_map = reference.concrete_to_id;

  // Map reference tensor to consumer
  std::unordered_map<IterDomain*, IterDomain*> root_ref_to_consumer;
  for (auto c_root : consumer_tv->getMaybeRFactorDomain()) {
    auto concrete_id = gpu_lower->caIndexMap().getConcreteMappedID(c_root);
    auto ref_id_it = reference_id_map.find(concrete_id);
    if (ref_id_it != reference_id_map.end()) {
      root_ref_to_consumer[ref_id_it->second] = c_root;
    }
  }

  BestEffortReplay replay_consumer_as_ref(
      consumer_tv->domain()->domain(),
      reference_domain->domain(),
      root_ref_to_consumer,
      false);

  const auto& ref_2_consumer = replay_consumer_as_ref.getReplay();

  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map;

  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  if (unswitch) {
    bool within_unswitch = false;
    const auto one = ir_builder.create<kir::Int>(1);
    for (auto loop : loops) {
      if (loop->iter_domain()->parallelType() == ParallelType::Unroll ||
          loop->iter_domain()->parallelType() == ParallelType::Unswitch ||
          loop->iter_domain()->parallelType() == ParallelType::Vectorize) {
        within_unswitch = true;
      }

      if (within_unswitch && !loop->iter_domain()->isThread()) {
        loop_to_ind_map[loop] =
            ir_builder.subExpr(loop->iter_domain()->extent(), one);
      }
    }
  }

  std::unordered_map<kir::IterDomain*, kir::Val*> ref_id_to_ind_map;
  // Due to rfactor/initialization reference_domain may be bigger than loop nest
  // structure
  TORCH_INTERNAL_ASSERT(loops.size() <= reference_domain->nDims());
  for (size_t loop_i = 0; loop_i < loops.size(); loop_i++) {
    auto ref_axis = gpu_lower->lowerValue(reference_domain->axis(loop_i))
                        ->as<kir::IterDomain>();
    ref_id_to_ind_map[ref_axis] = loop_to_ind_map[loops[loop_i]];
  }

  // Index into the reference tensor
  auto ref_compute =
      getReferenceIndexing(loops, reference_domain, ref_id_to_ind_map, {});

  // Index into consumer using reference indexing
  auto consumer_indexing = ref_compute.updateIndexCompute(
      consumer_tv->domain(), ref_2_consumer, root_contiguity);

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.

  // If we are generating a predicate for initialization, we should use
  // rfactor instead of root_dom. If we are generating a predicate for
  // actual reduction expr, reduction axes should have their indices
  // mapped to non-zero symbolic vals.
  bool buffer_init = false;
  for (auto consumer_id : kir_consumer_tv->domain()->domain()) {
    if (consumer_id->isReduction()) {
      if (consumer_indexing.indexMap().find(consumer_id) !=
          consumer_indexing.indexMap().end()) {
        if (!consumer_indexing.indexMap().at(consumer_id)->isZeroInt()) {
          buffer_init = false;
          break;
        }
      }
      buffer_init = true;
    }
  }

  // If we are initializing a reduction buffer and the tensor has a
  // rfactor root, the predicate should be based on the rfactor root.
  const auto root_domain =
      (buffer_init && kir_consumer_tv->domain()->hasRFactor())
      ? kir_consumer_tv->domain()->rfactorDomain()
      : kir_consumer_tv->domain()->rootDomain();

  const auto zero = ir_builder.create<kir::Int>(0);
  std::vector<kir::Val*> root_inds(root_domain.size(), zero);

  for (size_t i = 0; i < root_domain.size(); i++) {
    if (root_domain[i]->isBroadcast() ||
        gpu_lower->isDerivedFromTrivialReduction(root_domain[i])) {
      continue;
    }
    const auto it = consumer_indexing.indexMap().find(root_domain[i]);
    if (it != consumer_indexing.indexMap().end()) {
      root_inds[i] = it->second;
    }
  }

  return {root_inds, buffer_init};
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
