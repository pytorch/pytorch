#include <torch/csrc/jit/codegen/cuda/index_compute.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
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

    if ((!isContig(gpu_lower->lowerValue(inner)->as<kir::IterDomain>()) ||
         !isContig(gpu_lower->lowerValue(outer)->as<kir::IterDomain>()))) {
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
  // contiguous. Ignore root order is primarily used for predicate generation.
  // In this case we can linearize indexing of any ID that only consists of
  // merge operations.
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

    for (const auto i : c10::irange(root_domain_.size())) {
      if (root_contiguity_[i]) {
        auto kir_root_domain_i =
            gpu_lower->lowerValue(root_domain_[i])->as<kir::IterDomain>();
        contig_ids.emplace(kir_root_domain_i);
        within_contig_ids[kir_root_domain_i] =
            std::unordered_set<kir::IterDomain*>();
        is_contig_root[root_domain_[i]] = true;
      } else {
        is_contig_root[root_domain_[i]] = false;
      }
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

// Update the HaloInfo mappings for a reference tensor by propagating
// the halo information from the consumer tensor.
void updateHaloInfoForReference(
    const ReferenceTensor& reference,
    const TensorView* consumer_tv) {
  const auto gpu_lower = GpuLower::current();

  auto& halo_info = gpu_lower->haloInfo();

  auto* reference_domain = reference.domain;
  const auto& reference_concrete_map = reference.concrete_to_id;

  for (auto reference_root_axis : reference_domain->getRootDomain()) {
    // Set default
    halo_info.setRootAxisInfo(reference_root_axis, AxisHaloInfo());
    auto consumer_it = std::find_if(
        consumer_tv->getRootDomain().begin(),
        consumer_tv->getRootDomain().end(),
        [&](IterDomain* consumer_root) {
          auto concrete_id =
              gpu_lower->caIndexMap().getConcreteMappedID(consumer_root);
          auto it = reference_concrete_map.find(concrete_id);
          return it != reference_concrete_map.end() &&
              it->second == reference_root_axis;
        });
    // When no corresponding ID of the consumer exists, the reference
    // axis can be ignored
    if (consumer_it == consumer_tv->getRootDomain().end()) {
      continue;
    }
    auto consumer_root_axis = *consumer_it;
    auto root_axis_info =
        gpu_lower->haloInfo().getRootAxisInfo(consumer_root_axis);
    if (root_axis_info.width() == 0) {
      continue;
    }
    halo_info.setRootAxisInfo(reference_root_axis, root_axis_info);
  }

  halo_info.build(reference_domain);

  return;
}

// Get a map of IterDomains to halo-extended extents of corresponding
// reference IterDomains.
//
// ref_map: ref-to-consumer in consumer indexing; ref-to-producer in
// producer indexing
std::unordered_map<kir::IterDomain*, kir::Val*> getReferenceHaloExtentMap(
    const ReferenceTensor& reference,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, IterDomain*>& ref_map,
    const std::unordered_map<kir::IterDomain*, kir::Val*>& extent_map) {
  const auto gpu_lower = GpuLower::current();

  // First, update HaloInfo with the reference tensor, which reflects
  // the halo extents of the consumer tensor.
  updateHaloInfoForReference(reference, consumer_tv);

  const auto& halo_info = gpu_lower->haloInfo();

  std::unordered_map<kir::IterDomain*, kir::Val*> reference_halo_extent_map;

  // Propagate halo extents of the reference to the consumer or
  // producer tensor
  for (auto kv : ref_map) {
    auto ref_id = gpu_lower->lowerValue(kv.first)->as<kir::IterDomain>();
    auto producer_or_consumer_id =
        gpu_lower->lowerValue(kv.second)->as<kir::IterDomain>();
    auto extent = halo_info.getExtent(ref_id);
    if (extent == nullptr) {
      auto extent_it = extent_map.find(ref_id);
      if (extent_it != extent_map.end()) {
        extent = extent_it->second;
      } else {
        extent = ref_id->extent();
      }
    }
    reference_halo_extent_map[producer_or_consumer_id] = extent;
  }

  return reference_halo_extent_map;
}

//! Offset of an index of a producer axis with respect to its
//! corresponding consumer index
kir::Val* getProducerHaloOffset(
    const TensorView* producer_tv,
    size_t producer_axis,
    const TensorView* consumer_tv) {
  auto p2c =
      PairwiseRootDomainMap(producer_tv, consumer_tv)
          .mapProducerToConsumer(producer_tv->domain(), consumer_tv->domain());

  auto producer_id = producer_tv->getMaybeRFactorDomain()[producer_axis];

  auto it = p2c.find(producer_id);
  // p2c should always have a mapping for producer_id. The only case
  // where no mapping exists for a producer axis is when it is a
  // reduction axis. Since this function is only used for indexing
  // producer tensors, where reduction axes are skipped, producer_id
  // should never be a reduction axis.
  TORCH_INTERNAL_ASSERT(it != p2c.end());
  IterDomain* consumer_id = it->second;

  const auto& halo_map = GpuLower::current()->haloInfo();
  const auto p_pad = halo_map.getRootAxisInfo(producer_id).width(0);
  const auto c_pad = halo_map.getRootAxisInfo(consumer_id).width(0);

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  kir::Val* offset = (p_pad->isConst() && c_pad->isConst())
      ? ir_builder.create<kir::Int>(
            p_pad->value().value() - c_pad->value().value())
      : ir_builder.subExpr(p_pad, c_pad);

  // If the consumer is a result of shifting the producer, adjust the
  // producer index per the offsets argument of the shift op.
  if (auto shift_op = dynamic_cast<const ShiftOp*>(consumer_tv->definition())) {
    offset = ir_builder.subExpr(
        offset, ir_builder.create<kir::Int>(shift_op->offset(producer_axis)));
  }

  return offset;
}

//! Offset producer index when necessary
kir::Val* getProducerIndexWithHalo(
    const TensorView* producer_tv,
    size_t producer_axis,
    kir::Val* producer_index,
    const TensorView* consumer_tv) {
  const auto offset =
      getProducerHaloOffset(producer_tv, producer_axis, consumer_tv);

  if (offset->isZeroInt()) {
    return producer_index;
  }

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  producer_index = ir_builder.addExpr(producer_index, offset);

  return producer_index;
}

//! Offset a producer index of a gather expression
//!
//! Given an index of a producer root axis, build a new index
//! expression that accesses a window position that the current loop
//! structure refers to.
kir::Val* getProducerIndexWithGather(
    size_t producer_root_axis,
    kir::Val* producer_index,
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::unordered_map<kir::IterDomain*, kir::Val*>& ref_index_map,
    const std::unordered_map<IterDomain*, IterDomain*>& ref_concrete_map) {
  auto gather_op = dynamic_cast<const GatherOp*>(consumer_tv->definition());

  // Just return the producer index as is if this is not a gather
  if (gather_op == nullptr) {
    return producer_index;
  }

  // Consumer axis that corresponds to the producer axis
  int consumer_axis = -1;
  for (size_t i = 0; i <= producer_root_axis; ++i) {
    if (producer_tv->getRootDomain()[i]->isReduction()) {
      continue;
    }
    ++consumer_axis;
  }

  TORCH_INTERNAL_ASSERT(
      consumer_axis >= 0 &&
          consumer_axis < (int)gather_op->windowShape().size(),
      "Invalid consumer axis",
      consumer_axis,
      ", producer_axis: ",
      producer_root_axis);

  // If the window extent is one, no specific offsetting
  // is necessary
  if (gather_op->windowShape()[consumer_axis]->isOneInt()) {
    return producer_index;
  }

  // Basically, the goal is to build an expression of producer_index +
  // window_index, so we first need to locate the index expression
  // that corresponds to the window axis of this producer axis.

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Locate the root IterDomain of the reference that corresponds to the gather
  // axis
  const auto window_root_axis = gather_op->gatherAxis(consumer_axis);
  auto concrete_window_id = gpu_lower->caIndexMap().getConcreteMappedID(
      consumer_tv->getRootDomain().at(window_root_axis));
  auto ref_concrete_map_it = ref_concrete_map.find(concrete_window_id);
  TORCH_INTERNAL_ASSERT(ref_concrete_map_it != ref_concrete_map.end());
  IterDomain* reference_root_of_gather_axis = ref_concrete_map_it->second;

  // Now that reference_root_of_gather_axis is the IterDomain for the
  // window axis, take its corresponding index from the index map
  auto window_idx =
      ref_index_map.at(gpu_lower->lowerValue(reference_root_of_gather_axis)
                           ->as<kir::IterDomain>());

  // Positive (or negative) padding at offset zero means the indexing
  // shifted to the negative (or positive) direction.
  auto pad_width = gather_op->padWidth()[consumer_axis][0];

  // producer_index - padding + window_index
  auto offset_producer_index = ir_builder.addExpr(
      ir_builder.subExpr(
          producer_index, ir_builder.create<kir::Int>(pad_width)),
      window_idx);

  return offset_producer_index;
}

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

  const bool outer_zero = isZero(outer_id);
  const bool inner_zero = isZero(inner_id);

  // We want to mark as zero merged in if we're working with shared or local
  // memory, and the dimension we're working with is not part of the allocation,
  // as we have special propagation rules for that scenario.

  // Maybe clear in_id as it could have been mapped over from another
  // IndexCompute. Uncertain if this is needed but seems to be safe.
  bool zero_merged_in = hasZeroMerged(in_id) || hasZeroMerged(inner_id) ||
      hasZeroMerged(outer_id);

  // If both are zero, the split input is also zero
  if (inner_zero && outer_zero) {
    zero_.emplace(in_id);
  }

  if (zero_merged_in) {
    zero_merged_in_.emplace(in_id);
  }

  if (isZero(in_id)) {
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
    // The extent of a root axis should be only updated when its
    // allocation is partial, i.e., zero_merged_in is true. See issue
    // #1016 and the FusionIssue1016 test.
    if (split->in()->definition() != nullptr || zero_merged_in) {
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

  auto zero = ir_builder.zeroVal();

  if (isZero(out_id)) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = zero;
    extent_map_[outer_id] = zero;
    extent_map_[inner_id] = zero;
    zero_.emplace(outer_id);
    zero_.emplace(inner_id);
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

    index_map_[gpu_lower
                   ->lowerValue(*(input_ids.end() - 1))
                   // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
                   ->as<kir::IterDomain>()] = out_ind;
    return;
  }

  kir::Val* inner_extent = getExtent(inner_id);

  // When the reference has halo extent for inner_id, that extent needs to
  // be used to un-merge
  if (reference_halo_extent_map_.find(inner_id) !=
      reference_halo_extent_map_.end()) {
    inner_extent = reference_halo_extent_map_[inner_id];
  }

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
    std::unordered_set<kir::IterDomain*> preferred_paths,
    std::unordered_map<kir::IterDomain*, kir::Val*> reference_halo_extent_map)
    : td_(_td),
      index_map_(std::move(initial_index_map)),
      extent_map_(std::move(extent_map)),
      zero_merged_in_(std::move(zero_merged_in)),
      preferred_paths_(std::move(preferred_paths)),
      reference_halo_extent_map_(std::move(reference_halo_extent_map)) {
  FUSER_PERF_SCOPE("GpuLower::Lower::IndexCompute::IndexCompute");

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

  // Initialize the zero_ set with domains that do not contibute to
  // the resulting index. Any domain that is mapped to Int(0), except
  // for vectorized ones, is included in this set.
  const auto gpu_lower = GpuLower::current();
  for (auto dom : td_->domain()) {
    auto kir_dom = gpu_lower->lowerValue(dom)->as<kir::IterDomain>();
    auto it = index_map_.find(kir_dom);
    if (it == index_map_.end()) {
      continue;
    }
    auto idx = it->second;
    if (idx->isZeroInt() && !isParallelTypeVectorize(dom->getParallelType())) {
      zero_.emplace(kir_dom);
    }
  }
}

void IndexCompute::run() {
  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  traverseFrom(td_->fusion(), domain_vals, false);
}

kir::Val* IndexCompute::getExtent(kir::IterDomain* id) {
  if (isParallelTypeThread(id->parallelType())) {
    auto parallel_dim =
        GpuLower::current()->parallelDimensionMap().get(id->parallelType());
    TORCH_INTERNAL_ASSERT(parallel_dim != nullptr);
    return parallel_dim;
  } else if (extent_map_.find(id) != extent_map_.end()) {
    return extent_map_.at(id);
  } else {
    return id->extent();
  }
}

bool IndexCompute::hasZeroMerged(kir::IterDomain* id) const {
  return zero_merged_in_.find(id) != zero_merged_in_.end() || isZero(id);
}

bool IndexCompute::isZero(kir::IterDomain* id) const {
  return zero_.find(id) != zero_.end();
}

IndexCompute IndexCompute::updateIndexCompute(
    const TensorDomain* new_td,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    const std::vector<bool>& root_contiguity,
    const std::unordered_map<kir::IterDomain*, kir::Val*>&
        reference_halo_extent_map) {
  FUSER_PERF_SCOPE("GpuLower::Lower::updateIndexCompute");

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
      root_contiguity,
      {},
      reference_halo_extent_map);
  updated_index_compute.run();

  return updated_index_compute;
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

// Returns halo-extended extent if id has halo. Otherwise, just
// returns id->extent.
kir::Val* getHaloExtentOfRootAxis(
    IterDomain* id,
    kir::Val* normal_extent = nullptr) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  if (normal_extent == nullptr) {
    normal_extent = gpu_lower->lowerValue(id->extent());
  }

  const auto& halo = gpu_lower->haloInfo().getRootAxisInfo(id);
  if (halo.hasHalo()) {
    auto halo_extent = ir_builder.addExpr(normal_extent, halo.width());
    return halo_extent;
  } else {
    return normal_extent;
  }
}

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
          id_to_swizzle_j_kir->extent());
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

std::vector<kir::Val*> Index::getGlobalProducerStridedIndices(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getGlobalProducerIndex");
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
      TransformReplay::replayPasC(producer_tv, consumer_tv, -1, pairwiseMap)
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
      root_ref_to_producer);

  const auto& ref_2_producer = replay_producer_as_ref.getReplay();

  // Forward vectorized IDs to index into producer correctly
  // We want p_id to be vectorized like consumer just for the indexing, then we
  // need to switch it back later. Store previous state here when changing. We
  // need to do this as replaying producer as consumer can use replay best
  // effort which means some domains may be the originals.
  std::vector<std::pair<IterDomain*, ParallelType>> p_id_backup;
  for (auto entry : ref_2_producer) {
    auto ref_id = entry.first;
    auto p_id = entry.second;
    if (ref_id->getParallelType() == ParallelType::Vectorize) {
      p_id_backup.emplace_back(std::make_pair(p_id, p_id->getParallelType()));
      p_id->parallelize(ParallelType::Vectorize);
    } else if (ref_id->getParallelType() == ParallelType::MisalignedVectorize) {
      p_id->parallelize(ParallelType::MisalignedVectorize);
    }
  }

  const auto reference_halo_extent_map = getReferenceHaloExtentMap(
      reference, consumer_tv, ref_2_producer, ref_compute.extentMap());

  // Index into producer using reference indexing
  auto producer_indexing = ref_compute.updateIndexCompute(
      producer_tv->domain(),
      ref_2_producer,
      producer_tv->domain()->contiguity(),
      reference_halo_extent_map);

  // Revert p_ids
  for (auto entry : p_id_backup) {
    entry.first->parallelize(entry.second);
  }

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();

  // TODO: Abstract stride logic to reuse with consumer indexing
  auto zero = ir_builder.create<kir::Int>(0);
  std::vector<kir::Val*> strides(root_dom.size(), nullptr);
  {
    int stride_i = 0;
    for (size_t i = 0; i < root_dom.size(); i++) {
      if (root_dom[i]->isReduction() ||
          root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
        strides[i] = zero;
        continue;
      }
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << stride_i++ << "]";
      strides[i] = ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int);
    }
  }

  kir::Val* cur_contig_stride = ir_builder.create<kir::Int>(1);
  // if we have rfactor we can't simplify the indexing like this, we would need
  // to fix contiguity size to be rfactor size not root size
  if (root_dom.size() == producer_tv->domain()->contiguity().size()) {
    for (size_t i = 0; i < root_dom.size(); i++) {
      auto dim = root_dom.size() - i - 1;
      if (root_dom[dim]->isReduction()) {
        continue;
      }
      if (root_dom[dim]->getIterType() == IterType::BroadcastWithoutStride) {
        continue;
      }

      kir::Val* root_ind = nullptr;
      auto kir_root_dom =
          gpu_lower->lowerValue(root_dom[dim])->as<kir::IterDomain>();
      if (producer_indexing.indexMap().find(kir_root_dom) !=
          producer_indexing.indexMap().end()) {
        root_ind = producer_indexing.indexMap().at(kir_root_dom);
      } else if (
          root_dom[dim]->getIterType() == IterType::BroadcastWithStride) {
        root_ind = zero;
      }

      TORCH_INTERNAL_ASSERT(
          root_ind != nullptr,
          "Couldn't find root mapping for TV",
          producer_tv->name(),
          " dim: ",
          i,
          " id: ",
          root_dom[dim]);

      if (producer_tv->domain()->contiguity()[dim]) {
        // If contig, used the stored stride which may be the previous
        // dimensions stride * previous dimensions size
        strides[dim] = cur_contig_stride;
        // Prepare for the next dimension which may also be contiguous, multiply
        // by extent of this dimension
        auto root_dim_extent = getHaloExtentOfRootAxis(root_dom[dim]);
        cur_contig_stride =
            ir_builder.mulExpr(cur_contig_stride, root_dim_extent);
      } else {
        // If non contiguous dimension, keep local stride information, set cur
        // stride to local stride * local raw extent
        auto root_dim_extent = getHaloExtentOfRootAxis(root_dom[dim]);
        cur_contig_stride = ir_builder.mulExpr(strides[dim], root_dim_extent);
      }
    }
  }

  auto vectorize_shift =
      loops.empty() ? nullptr : loops.back()->vectorize_shift();

  // Global striding
  std::vector<kir::Val*> strided_inds(root_dom.size(), ir_builder.zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    // If the domain is derived from a trivial reduction, no indexing
    // to create.
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride ||
        root_dom[i]->getIterType() == IterType::BroadcastWithStride ||
        gpu_lower->trivialReductionInfo().isDerived(root_dom[i])) {
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

    root_ind = getProducerIndexWithHalo(producer_tv, i, root_ind, consumer_tv);

    root_ind = getProducerIndexWithGather(
        i,
        root_ind,
        producer_tv,
        consumer_tv,
        ref_compute.indexMap(),
        reference_id_map);

    if (root_ind->isZeroInt()) {
      continue;
    } else {
      auto strided_ind = ir_builder.mulExpr(root_ind, strides[i]);
      if (i == root_dom.size() - 1 && vectorize_shift != nullptr) {
        strided_inds[i] = ir_builder.addExpr(strided_ind, vectorize_shift);
      } else {
        strided_inds[i] = strided_ind;
      }
    }
  }

  return strided_inds;
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
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (!within_alloc) {
      if ((loop->iter_domain()->isThreadDim() && is_shared) ||
          (loop->iter_domain()->isThread() && is_global)) {
        idx = loop->index();
      } else {
        idx = zero;
      }
    } else if (
        (loop->iter_domain()->isBlockDim() && is_shared) ||
        (loop->iter_domain()->isThread() && is_local) || loop->vectorize()) {
      idx = zero;
    } else {
      idx = loop->index();
    }

    loop_to_ind_map[loop] = idx;

    if (!within_alloc && loop == alloc_loop) {
      within_alloc = true;
    }
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return loop_to_ind_map;
}

} // namespace

// Producer index for either shared or local memory
std::vector<kir::Val*> Index::getNonGlobalProducerStridedIndices(
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
  auto pairwise_map = PairwiseRootDomainMap(producer_tv, consumer_tv);
  auto producer_replayed_as_consumer =
      TransformReplay::replayPasC(producer_tv, consumer_tv, -1, pairwise_map)
          .first;

  ir_utils::TVDomainGuard domain_guard(
      producer_tv, producer_replayed_as_consumer);

  //  We want to play producer as consumer instead of the other way around since
  //  consumer may have some broadcasted axes producer doesn't have merged into
  //  loops producer may use. If we did consumer as producer we wouldn't have
  //  this information in the mapping.
  auto replay_PasC =
      BestEffortReplay::replayPasC(producer_tv, consumer_tv, -1, pairwise_map);

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
      root_ref_to_producer);

  const auto& ref_2_producer = replay_producer_as_ref.getReplay();

  // Forward vectorized IDs to index into producer correctly
  // We want p_id to be vectorized like consumer just for the indexing, then we
  // need to switch it back later. Store previous state here when changing. We
  // need to do this as replaying producer as consumer can use replay best
  // effort which means some domains may be the originals.
  std::vector<std::pair<IterDomain*, ParallelType>> p_id_backup;
  for (auto entry : ref_2_producer) {
    auto ref_id = entry.first;
    auto p_id = entry.second;
    if (ref_id->getParallelType() == ParallelType::Vectorize) {
      p_id_backup.emplace_back(std::make_pair(p_id, p_id->getParallelType()));
      p_id->parallelize(ParallelType::Vectorize);
    } else if (ref_id->getParallelType() == ParallelType::MisalignedVectorize) {
      p_id->parallelize(ParallelType::MisalignedVectorize);
    }
  }

  // Index into producer using reference indexing

  const auto reference_halo_extent_map = getReferenceHaloExtentMap(
      reference, consumer_tv, ref_2_producer, ref_compute.extentMap());

  auto producer_indexing = ref_compute.updateIndexCompute(
      producer_tv->domain(),
      ref_2_producer,
      producer_tv->domain()->contiguity(),
      reference_halo_extent_map);

  // Revert p_ids
  for (auto entry : p_id_backup) {
    entry.first->parallelize(entry.second);
  }

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

  // Figure out which root axes we don't need to index
  std::unordered_set<IterDomain*> skip_indexing;

  for (auto root_id : root_dom) {
    // Already taken care of because we can detect no indexing required
    if (root_id->isBroadcast() || root_id->isReduction() ||
        gpu_lower->trivialReductionInfo().isDerived(root_id)) {
      skip_indexing.insert(root_id);
      continue;
    }

    // Already an entry for this root domain, continue
    if (index_map.find(gpu_lower->lowerValue(root_id)->as<kir::IterDomain>()) !=
        index_map.end()) {
      continue;
    }

    // Maps to consumers trivial reduction, don't index
    if (p2c_map.find(root_id) != p2c_map.end() &&
        gpu_lower->trivialReductionInfo().isDerived(p2c_map.at(root_id))) {
      skip_indexing.emplace(root_id);
    }
  }

  std::vector<kir::Val*> strided_inds(root_dom.size(), ir_builder.zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    if (skip_indexing.count(root_dom[i])) {
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

    auto root_ind_i = index_map.at(kir_root_dom_i);

    root_ind_i =
        getProducerIndexWithHalo(producer_tv, i, root_ind_i, consumer_tv);

    root_ind_i = getProducerIndexWithGather(
        i,
        root_ind_i,
        producer_tv,
        consumer_tv,
        ref_compute.indexMap(),
        reference_id_map);

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    kir::Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (skip_indexing.count(root_dom[j])) {
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

      root_ext_j = getHaloExtentOfRootAxis(root_dom[j], root_ext_j);

      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = ir_builder.mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds[i] = ir_builder.mulExpr(root_ind_i, stride);
    } else {
      strided_inds[i] = root_ind_i;
    }
  }

  return strided_inds;
}

std::vector<kir::Val*> Index::getGlobalConsumerStridedIndices(
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getGlobalConsumerIndex");
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
      root_ref_to_consumer);

  const auto& ref_2_consumer = replay_consumer_as_ref.getReplay();

  // Index into the reference tensor. Reference indexing will handle vectorized
  // dims where index should be set to 0
  auto ref_compute = getReferenceIndexing(loops, reference_domain);

  // Index into consumer using reference indexing

  const auto reference_halo_extent_map = getReferenceHaloExtentMap(
      reference, consumer_tv, ref_2_consumer, ref_compute.extentMap());

  auto consumer_indexing = ref_compute.updateIndexCompute(
      consumer_tv->domain(),
      ref_2_consumer,
      consumer_tv->domain()->contiguity(),
      reference_halo_extent_map);

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  // TODO: Abstract stride logic to reuse with producer indexing
  auto zero = ir_builder.zeroVal();
  std::vector<kir::Val*> strides(root_dom.size(), zero);
  {
    int stride_i = 0;
    for (size_t i = 0; i < root_dom.size(); i++) {
      if (root_dom[i]->isReduction() ||
          root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
        strides[i] = zero;
        continue;
      }
      std::stringstream ss;
      ss << "T" << consumer_tv->name() << ".stride[" << stride_i++ << "]";
      strides[i] = ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int);
    }
  }

  kir::Val* cur_contig_stride = ir_builder.oneVal();
  // if we have rfactor we can't simplify the indexing like this, we would need
  // to fix contiguity size to be rfactor size not root size
  if (root_dom.size() == consumer_tv->domain()->contiguity().size()) {
    for (size_t i = 0; i < root_dom.size(); i++) {
      auto dim = root_dom.size() - i - 1;
      if (root_dom[dim]->isReduction()) {
        continue;
      }
      if (root_dom[dim]->getIterType() == IterType::BroadcastWithoutStride) {
        continue;
      }

      kir::Val* root_ind = nullptr;
      auto kir_root_dom =
          gpu_lower->lowerValue(root_dom[dim])->as<kir::IterDomain>();
      if (consumer_indexing.indexMap().find(kir_root_dom) !=
          consumer_indexing.indexMap().end()) {
        root_ind = consumer_indexing.indexMap().at(kir_root_dom);
      } else if (
          root_dom[dim]->getIterType() == IterType::BroadcastWithStride) {
        root_ind = zero;
      }

      TORCH_INTERNAL_ASSERT(
          root_ind != nullptr,
          "Couldn't find root mapping for TV",
          consumer_tv->name(),
          " dim: ",
          i,
          " id: ",
          root_dom[dim]);

      if (consumer_tv->domain()->contiguity()[dim]) {
        // If contig, used the stored stride which may be the previous
        // dimensions stride * previous dimensions size
        strides[dim] = cur_contig_stride;
        // Prepare for the next dimension which may also be contiguous, multiply
        // by extent of this dimension
        auto root_dim_extent = getHaloExtentOfRootAxis(root_dom[dim]);
        cur_contig_stride =
            ir_builder.mulExpr(cur_contig_stride, root_dim_extent);
      } else {
        // If non contiguous dimension, keep local stride information, set cur
        // stride to local stride * local raw extent
        cur_contig_stride = ir_builder.mulExpr(
            strides[dim], getHaloExtentOfRootAxis(root_dom[dim]));
      }
    }
  }

  auto vectorize_shift =
      loops.empty() ? nullptr : loops.back()->vectorize_shift();

  // Global striding
  std::vector<kir::Val*> strided_inds(root_dom.size(), ir_builder.zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    // See a comment in indexing to root domains in getGlobalProducerIndex.
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride ||
        root_dom[i]->getIterType() == IterType::BroadcastWithStride ||
        gpu_lower->trivialReductionInfo().isDerived(root_dom[i])) {
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

    auto root_ind = consumer_indexing.indexMap().at(kir_root_dom_i);

    if (root_ind->isZeroInt()) {
      continue;
    } else {
      auto strided_ind = ir_builder.mulExpr(root_ind, strides[i]);
      if (i == root_dom.size() - 1 && vectorize_shift != nullptr) {
        strided_inds[i] = ir_builder.addExpr(strided_ind, vectorize_shift);
      } else {
        strided_inds[i] = strided_ind;
      }
    }
  }

  return strided_inds;
}

// Consumer index for either shared or local memory
std::vector<kir::Val*> Index::getNonGlobalConsumerStridedIndices(
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
      root_ref_to_consumer);

  const auto& ref_2_consumer = replay_consumer_as_ref.getReplay();

  const auto reference_halo_extent_map = getReferenceHaloExtentMap(
      reference, consumer_tv, ref_2_consumer, ref_compute.extentMap());

  // Index into consumer using reference indexing
  auto consumer_indexing = ref_compute.updateIndexCompute(
      consumer_tv->domain(),
      ref_2_consumer,
      consumer_tv->domain()->contiguity(),
      reference_halo_extent_map);

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
  std::vector<kir::Val*> strided_inds(root_dom.size(), ir_builder.zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast() ||
        gpu_lower->trivialReductionInfo().isDerived(root_dom[i])) {
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
          gpu_lower->trivialReductionInfo().isDerived(root_dom[j])) {
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

      root_ext_j = getHaloExtentOfRootAxis(root_dom[j], root_ext_j);

      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = ir_builder.mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds[i] = ir_builder.mulExpr(root_ind_i, stride);
    } else {
      strided_inds[i] = root_ind_i;
    }
  }

  return strided_inds;
}

std::vector<kir::Val*> Index::getProducerStridedIndices(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getProducerStridedIndices");
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  if (producer->domain()->noReductions().size() == 0) {
    return std::vector<kir::Val*>(
        producer->getMaybeRFactorDomain().size(), ir_builder.zeroVal());
  }

  std::vector<kir::Val*> strided_indices;
  if (producer->getMemoryType() == MemoryType::Global) {
    strided_indices =
        getGlobalProducerStridedIndices(producer, consumer, loops);
  } else {
    strided_indices =
        getNonGlobalProducerStridedIndices(producer, consumer, loops);
  }

  TORCH_INTERNAL_ASSERT(
      strided_indices.size() == producer->getMaybeRFactorDomain().size());

  return strided_indices;
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto strided_indices = getProducerStridedIndices(producer, consumer, loops);
  return ir_builder.create<kir::TensorIndex>(producer, strided_indices);
}

std::vector<kir::Val*> Index::getConsumerStridedIndices(
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getConsumerStridedIndices");
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  if (consumer->domain()->noReductions().size() == 0) {
    return std::vector<kir::Val*>(
        consumer->getMaybeRFactorDomain().size(), ir_builder.zeroVal());
  }

  std::vector<kir::Val*> strided_indices;
  if (consumer->getMemoryType() == MemoryType::Global) {
    strided_indices = getGlobalConsumerStridedIndices(consumer, loops);
  } else {
    strided_indices = getNonGlobalConsumerStridedIndices(consumer, loops);
  }

  TORCH_INTERNAL_ASSERT(
      strided_indices.size() == consumer->getMaybeRFactorDomain().size());

  return strided_indices;
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto strided_indices = getConsumerStridedIndices(consumer, loops);
  return ir_builder.create<kir::TensorIndex>(consumer, strided_indices);
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
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getConsumerRootPredIndices");

  auto consumer_tv = kir_consumer_tv->fuserTv();

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Get a reference tensor replayed as existing loop structure
  ReferenceTensor reference = IndexReferenceReplay::getReference(loops);
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
      root_ref_to_consumer);

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

      if (within_unswitch) {
        if (loop->iter_domain()->isThread()) {
          loop_to_ind_map[loop] = loop->start();
        } else {
          loop_to_ind_map[loop] = ir_builder.subExpr(loop->stop(), one);
        }
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

  const auto reference_halo_extent_map = getReferenceHaloExtentMap(
      reference, consumer_tv, ref_2_consumer, ref_compute.extentMap());

  // Index into consumer using reference indexing
  auto consumer_indexing = ref_compute.updateIndexCompute(
      consumer_tv->domain(),
      ref_2_consumer,
      root_contiguity,
      reference_halo_extent_map);

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

  for (const auto i : c10::irange(root_domain.size())) {
    if (root_domain[i]->isBroadcast() ||
        gpu_lower->trivialReductionInfo().isDerived(root_domain[i])) {
      continue;
    }
    const auto it = consumer_indexing.indexMap().find(root_domain[i]);
    if (it != consumer_indexing.indexMap().end()) {
      root_inds[i] = it->second;
    }
  }

  return {root_inds, buffer_init};
}

namespace {
struct PredicateContigInfo {
 public:
  // Iteration domain that is only comprised of merge transformations
  IterDomain* contig_id;
  // The set of root iteration domains that make up the contig_id
  std::unordered_set<IterDomain*> root_ids;
};

// Find iteration domains in the history of reference comprised only of
// merge operations. Only return iteration domains that are subsequently fed
// into a split, or are in the provided domain. In other words, we don't want to
// return every IterDomain that's contiguous, just the one closest to the
// leaves. Predicates are not associated with physical memory so we can treat
// all of them as contiguous merges.
std::vector<PredicateContigInfo> getPredicateContigIds(
    std::vector<IterDomain*> reference_domain) {
  auto root_vals = IterVisitor::getInputsTo(
      {reference_domain.begin(), reference_domain.end()});
  auto root_ids = ir_utils::filterByType<IterDomain>(root_vals);

  // Mark all roots as being originally "contiguous"
  std::vector<IterDomain*> contiguous_ids(root_ids.begin(), root_ids.end());

  // Dereference root_vals.begin below, so make sure there's at least one entry
  if (root_vals.empty()) {
    return std::vector<PredicateContigInfo>();
  }

  // Run through iteration domain history
  auto exprs = ExprSort::getExprs(
      (*root_vals.begin())->fusion(),
      {reference_domain.begin(), reference_domain.end()});

  for (auto expr : exprs) {
    // If not a merge, output is not contiguous
    if (expr->isA<Merge>()) {
      auto merge = expr->as<Merge>();
      auto inner_contig_it = std::find(
          contiguous_ids.begin(), contiguous_ids.end(), merge->inner());
      auto outer_contig_it = std::find(
          contiguous_ids.begin(), contiguous_ids.end(), merge->outer());

      if (inner_contig_it != contiguous_ids.end() &&
          outer_contig_it != contiguous_ids.end()) {
        // If inner and outer are contiguous, out must be contiguous. Remove
        // inner and outer, and add out.
        contiguous_ids.erase(outer_contig_it);
        contiguous_ids.erase(std::find(
            contiguous_ids.begin(), contiguous_ids.end(), merge->inner()));
        contiguous_ids.emplace_back(merge->out());
      }
    }
  }

  std::vector<PredicateContigInfo> contig_id_infos;

  // Create entries and return them
  for (auto contig_id : contiguous_ids) {
    auto contig_root_vals = IterVisitor::getInputsTo({contig_id});
    auto contig_root_ids = ir_utils::filterByType<IterDomain>(contig_root_vals);
    PredicateContigInfo contig_id_info;
    contig_id_info.contig_id = contig_id;
    contig_id_info.root_ids = std::unordered_set<IterDomain*>(
        contig_root_ids.begin(), contig_root_ids.end());
    contig_id_infos.push_back(contig_id_info);
  }
  return contig_id_infos;
}

} // namespace

// Returns predicates and the concrete (by loop map) root domains they cover
std::pair<std::vector<kir::Bool*>, std::vector<std::unordered_set<IterDomain*>>>
Index::getReferenceRootPredicates(
    const kir::TensorView* kir_consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    bool unswitch) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getReferenceRootPredicates");

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Get a reference tensor replayed as existing loop structure
  ReferenceTensor reference = IndexReferenceReplay::getReference(loops);
  auto reference_domain = reference.domain;
  auto reference_id_map = reference.concrete_to_id;

  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map;

  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  // If unswitch don't directly use indices from for loop, use for loop extent
  // minus 1
  if (unswitch) {
    TORCH_INTERNAL_ASSERT(
        loops.size() <= reference_domain->nDims(),
        "Invalid reference generated.");
    bool within_unswitch = false;
    const auto one = ir_builder.create<kir::Int>(1);
    for (size_t loop_i = 0; loop_i < loops.size(); loop_i++) {
      auto loop = loops[loop_i];
      auto ref_id = reference_domain->axis(loop_i);
      if (loop->iter_domain()->parallelType() == ParallelType::Unroll ||
          loop->iter_domain()->parallelType() == ParallelType::Unswitch ||
          loop->iter_domain()->parallelType() == ParallelType::Vectorize) {
        within_unswitch = true;
      }

      if (within_unswitch) {
        // Rely on the reference to check broadcasting. The for loop could be
        // broadcasted on a constant value from an unroll split. Since reference
        // may convert this to an iter domain, that for loop could be valid to
        // generate predication from.
        if (ref_id->isBroadcast()) {
          // Ignore indexing into broadcasted dimensions.
          continue;
        } else if (loop->iter_domain()->isThread()) {
          loop_to_ind_map[loop] = loop->start();
        } else {
          loop_to_ind_map[loop] = ir_builder.subExpr(loop->stop(), one);
        }
      }
    }
  }

  // Add magic zero to a loop pretty far inside in indexing
  kir::IterDomain* magic_zero_loop = nullptr;
  std::unordered_map<kir::IterDomain*, kir::Val*> ref_id_to_ind_map;
  // Due to rfactor/initialization reference_domain may be bigger than loop nest
  // structure
  TORCH_INTERNAL_ASSERT(loops.size() <= reference_domain->nDims());
  for (size_t loop_i = 0; loop_i < loops.size(); loop_i++) {
    auto loop = loops[loop_i];
    auto ind = loop_to_ind_map[loops[loop_i]];
    auto ref_axis = reference_domain->axis(loop_i);
    auto kir_ref_axis = gpu_lower->lowerValue(ref_axis)->as<kir::IterDomain>();

    if (Index::protectWithMagicZero(loop, ref_axis, ind)) {
      magic_zero_loop = kir_ref_axis;
    }

    ref_id_to_ind_map[kir_ref_axis] = loop_to_ind_map[loop];
  }

  if (ref_id_to_ind_map.count(magic_zero_loop)) {
    ref_id_to_ind_map[magic_zero_loop] = ir_builder.addExpr(
        ref_id_to_ind_map[magic_zero_loop], ir_builder.magicZeroVal());
  }

  auto consumer_tv = kir_consumer_tv->fuserTv();

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
      root_ref_to_consumer);

  const auto& ref_2_consumer = replay_consumer_as_ref.getReplay();

  // Halo information is not currently used as lower_shift will take care of the
  // predicate generation and is still using the older function:
  // getConsumerRootPredIndices

  // Generate halo information for reference.
  updateHaloInfoForReference(reference, consumer_tv);

  std::unordered_map<kir::IterDomain*, kir::Val*> reference_halo_extent_map;

  const auto& halo_info = gpu_lower->haloInfo();

  // Generate map from reference iter domains to halo extents
  for (auto entry : ref_2_consumer) {
    auto ref_id = entry.first;
    auto extent = halo_info.getExtent(ref_id);
    if (extent != nullptr) {
      reference_halo_extent_map[gpu_lower->lowerValue(ref_id)
                                    ->as<kir::IterDomain>()] = extent;
    }
  }

  // Index into the reference tensor
  auto ref_indexing = getReferenceIndexing(
      loops,
      reference_domain,
      ref_id_to_ind_map,
      {},
      reference_halo_extent_map);

  // If we are initializing a reduction buffer and the tensor has a
  // rfactor root, the predicate should be based on the rfactor root.
  const auto root_domain = reference_domain->getRootDomain();

  // Get the contiguous ids we need to generate predicates for
  auto contig_id_infos = getPredicateContigIds(reference_domain->domain());

  // Roots in contiguous processing is based on reference roots, want to convert
  // these to concrete roots, flip reference's concrete_to_id map as reference
  // ids are not part of compute at maps.
  decltype(reference_id_map) ref_id_to_concrete;
  std::transform(
      reference_id_map.begin(),
      reference_id_map.end(),
      std::inserter(ref_id_to_concrete, ref_id_to_concrete.begin()),
      [](auto entry) { return std::make_pair(entry.second, entry.first); });

  // Track which roots have been handled by the generated predicates
  std::vector<std::unordered_set<IterDomain*>> handeled_roots;

  std::vector<kir::Bool*> predicates;

  for (auto contig_id_entry : contig_id_infos) {
    auto contig_id = contig_id_entry.contig_id;
    // No predicates needed for braodcasted indices.
    if (contig_id->isBroadcast() ||
        gpu_lower->trivialReductionInfo().isDerived(contig_id)) {
      continue;
    }

    auto root_ids = contig_id_entry.root_ids;
    auto kir_contig_id =
        gpu_lower->lowerValue(contig_id)->as<kir::IterDomain>();

    const auto it = ref_indexing.indexMap().find(kir_contig_id);

    // First condition below is due to broadcasts in consumers of consumer that
    // are not in consumer there can be unresolved indexing in the reference
    // tensor. This can happen when we have something like: TV3[i1o*i2, i1i] and
    // TV1[i2] where tv3 and tv1 share their outer dimension. i1 will be part of
    // reference tensors root domain, but when indexing into TV1 there aren't
    // enough indices to resolve it.
    //
    // The condition also happens with Misaligned predicates, where
    // inner-most vectorized loops are not included in the loops
    // parameter. Predicates involving vectorized loops are separately
    // generated in lower_misaligned_vectorization.
    //
    // Second condition is simply to avoid predication on broadcasting axes as
    // it's not required.
    if (it == ref_indexing.indexMap().end() || it->second->isZeroInt()) {
      continue;
    }

    // Use the iteration domains extent unless there's a halo extent
    auto extent = kir_contig_id->extent();

    auto halo_extent_it = reference_halo_extent_map.find(kir_contig_id);
    if (halo_extent_it != reference_halo_extent_map.end()) {
      extent = halo_extent_it->second;
    }

    // If the index definition is "simple" and the extent is "simple" then our
    // for loop goes exactly across the iteration domain extent so no predicate
    // needed.
    if (it->second->definition() == nullptr &&
        extent->definition() == nullptr) {
      continue;
    }

    predicates.push_back(
        ir_builder.ltExpr(it->second, extent)->as<kir::Bool>());

    // Transform roots from reference to concrete roots (based on loop compute
    // at map)
    std::unordered_set<IterDomain*> concrete_root_ids;
    std::transform(
        contig_id_entry.root_ids.begin(),
        contig_id_entry.root_ids.end(),
        std::inserter(concrete_root_ids, concrete_root_ids.begin()),
        [&ref_id_to_concrete](IterDomain* root_id) {
          return ref_id_to_concrete.at(root_id);
        });
    handeled_roots.push_back(concrete_root_ids);
  }

  return {predicates, handeled_roots};
}

bool Index::protectWithMagicZero(
    kir::ForLoop* loop,
    IterDomain* reference_domain,
    kir::Val* ind) {
  bool ref_dom_simple =
      (reference_domain == nullptr ? true
                                   : reference_domain->definition() != nullptr);
  bool ind_simple =
      (ind == nullptr ? true
                      : ind->definition() != nullptr && !ind->isZeroInt());
  return loop->isUnrollable() && (!ref_dom_simple || !ind_simple);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
