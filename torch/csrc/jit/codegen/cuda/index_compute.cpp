#include <torch/csrc/jit/codegen/cuda/index_compute.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/index_reference_replay.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>
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

  // Split outputs are not contiguous, don't need to do anything.
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

  // Check through the history of ids whose inputs map to root_domain with
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
      // If a root domain has halo, can't use merged domain even if
      // both inputs are contiguous. HaloInfo is also initialized for
      // rfactor root domains, which should just return "zero"
      // RootAxisInfo. This should be safe as no rfactor tensor should
      // need halo.
      if (root_contiguity_[i] &&
          !gpu_lower->haloInfo().getRootAxisInfo(root_domain_[i]).hasHalo()) {
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

  auto reference_domain = reference.domain;

  // First, propagate the halo information of the consumer root domain
  // to the reference root domain.
  for (auto consumer_root_id : consumer_tv->getRootDomain()) {
    auto consumer_index_concrete_id =
        gpu_lower->caIndexMap().getConcreteMappedID(consumer_root_id);
    auto reference_it =
        reference.concrete_to_id.find(consumer_index_concrete_id);
    if (reference_it == reference.concrete_to_id.end()) {
      // This happens when consumer_root_id is a broadcast or an
      // initialization of a reduction buffer. In those cases, since
      // the domain is not going to be predicated, it's not necessary
      // to propagate halo information to the reference tensor.
      continue;
    }
    auto reference_id = reference_it->second;
    halo_info.setRootAxisInfo(
        reference_id, halo_info.getRootAxisInfo(consumer_root_id));
  }

  // Now that the reference root has halo information copied from
  // the cosumer, propagate it down to non-root domains.
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
    const std::unordered_map<IterDomain*, IterDomain*>& index_map_from_ref) {
  const auto gpu_lower = GpuLower::current();

  const auto& halo_info = gpu_lower->haloInfo();

  std::unordered_map<kir::IterDomain*, kir::Val*> reference_halo_extent_map;

  // Propagate halo extents of the reference to the consumer or
  // producer tensor
  for (auto kv : index_map_from_ref) {
    auto ref_id = gpu_lower->lowerValue(kv.first)->as<kir::IterDomain>();
    auto producer_or_consumer_id =
        gpu_lower->lowerValue(kv.second)->as<kir::IterDomain>();
    auto extent = halo_info.getExtent(ref_id);
    if (extent != nullptr) {
      reference_halo_extent_map[producer_or_consumer_id] = extent;
    }
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

//! Create a producer offset based off a consumer index
//!
//! \param consumer_root_axis Position of corresponding consumer axis
//! \param consumer_tv Consumer TensorView
//! \param concrete_to_ref_map Mappings from concrete to reference domains
//! \param ref_index_map Mappings from reference domains to indices
kir::Val* getProducerOffsetWithGather(
    size_t consumer_root_axis,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, IterDomain*>& concrete_to_ref_map,
    const std::unordered_map<kir::IterDomain*, kir::Val*>& ref_index_map) {
  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  const auto gather_expr = dynamic_cast<GatherOp*>(consumer_tv->definition());

  if (gather_expr == nullptr) {
    return ir_builder.zeroVal();
  }

  // If the window extent is one, no specific offsetting
  // is necessary
  if (consumer_root_axis >= gather_expr->windowShape().size() ||
      gather_expr->windowShape()[consumer_root_axis]->isOneInt()) {
    return ir_builder.zeroVal();
  }

  // Basically, the goal is to build an expression of producer_index +
  // window_index, so we first need to locate the index expression
  // that corresponds to the window axis of this producer axis.

  // Locate the root IterDomain of the reference that corresponds to the gather
  // axis
  const auto window_axis = gather_expr->gatherAxis(consumer_root_axis);
  auto window_id = consumer_tv->getRootDomain().at(window_axis);
  auto concrete_window_id =
      gpu_lower->caIndexMap().getConcreteMappedID(window_id);
  auto concrete_2_ref_it = concrete_to_ref_map.find(concrete_window_id);
  TORCH_INTERNAL_ASSERT(concrete_2_ref_it != concrete_to_ref_map.end());
  IterDomain* reference_root_of_gather_axis = concrete_2_ref_it->second;

  // Now that reference_root_of_gather_axis is the IterDomain for the
  // window axis, take its corresponding index from the index map
  auto window_idx =
      ref_index_map.at(gpu_lower->lowerValue(reference_root_of_gather_axis)
                           ->as<kir::IterDomain>());

  // Positive (or negative) padding at offset zero means the indexing
  // shifted to the negative (or positive) direction.
  auto pad_width = gather_expr->padWidth()[consumer_root_axis][0];

  // producer offset: window_index - padding
  auto producer_offset =
      ir_builder.subExpr(window_idx, ir_builder.create<kir::Int>(pad_width));
  return producer_offset;
  ;
}

//! Offset a producer index of a gather expression
//!
//! Given an index of a producer root axis, build a new index
//! expression that accesses a window position that the current loop
//! structure refers to. Use getGatherProducerOffset to create an
//! offset Val.
kir::Val* getProducerIndexWithGather(
    kir::Val* producer_index,
    size_t producer_root_axis,
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, IterDomain*>& concrete_to_ref_map,
    const std::unordered_map<kir::IterDomain*, kir::Val*>& ref_index_map) {
  auto gather_op = dynamic_cast<const GatherOp*>(consumer_tv->definition());

  // Just return the producer index as is if this is not a gather
  if (gather_op == nullptr) {
    return producer_index;
  }

  // Consumer axis that corresponds to the producer axis
  int consumer_axis = -1;
  for (const auto i : c10::irange(producer_root_axis + 1)) {
    if (producer_tv->getMaybeRFactorDomain()[i]->isReduction() ||
        producer_tv->getMaybeRFactorDomain()[i]->isStride()) {
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

  kir::SimplifyingIrBuilder ir_builder(GpuLower::current()->kernel());
  auto offset = getProducerOffsetWithGather(
      consumer_axis, consumer_tv, concrete_to_ref_map, ref_index_map);
  return ir_builder.addExpr(producer_index, offset);
}

// Adjusts a global consumer index when its root domain is partially
// split. Note that non-global consumer indices don't need any
// adjustment.
kir::Val* getGlobalConsumerOffsetWithPartialSplit(kir::IterDomain* root_id) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto offset = gpu_lower->partialSplitMap().getStartOffset(root_id);
  if (offset == nullptr) {
    return ir_builder.zeroVal();
  } else {
    return offset;
  }
}

// Adjusts a global producer index when its root domain and
// corresponding consumer root domain have non-matching split
// offsets. Specifically, since producer_index is calcualted based on
// the consumer, if the consumer has a non-zero offset,
// it needs to be added to the index. Also, when the producer itself
// also has a non-zero split offset, that needs to be subtracted from
// the index.
kir::Val* getProducerIndexWithPartialSplit(
    kir::Val* producer_index,
    IterDomain* producer_root_id,
    const TensorView* producer_tv,
    const TensorView* consumer_tv) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto p2c =
      PairwiseRootDomainMap(producer_tv, consumer_tv)
          .mapProducerToConsumer(producer_tv->domain(), consumer_tv->domain());

  auto it = p2c.find(producer_root_id);
  if (it == p2c.end()) {
    return producer_index;
  }

  auto consumer_root_id = it->second;

  auto consumer_offset =
      gpu_lower->partialSplitMap().getStartOffset(consumer_root_id);
  auto consumer_offset_kir = consumer_offset == nullptr
      ? ir_builder.zeroVal()
      : gpu_lower->lowerValue(consumer_offset);

  auto producer_offset =
      gpu_lower->partialSplitMap().getStartOffset(producer_root_id);
  auto producer_offset_kir = producer_offset == nullptr
      ? ir_builder.zeroVal()
      : gpu_lower->lowerValue(producer_offset);

  // If the producer is on global memory, it's always allocated
  // without trimming the out-of-bounds region, so the consumer offset
  // should be added to the index.
  if (producer_tv->getMemoryType() == MemoryType::Global) {
    if (consumer_offset_kir->isZeroInt()) {
      return producer_index;
    } else {
      return ir_builder.addExpr(producer_index, consumer_offset_kir);
    }
  }

  // Non-global case. Difference of the split offsets must be
  // accounted.

  auto diff = ir_builder.subExpr(consumer_offset_kir, producer_offset_kir);
  kir::ExpressionEvaluator ee;
  auto diff_eval = ee.evaluate(diff);
  // We currently only allow constant offsetting
  TORCH_INTERNAL_ASSERT(diff_eval.has_value(), "Invalid partial split");

  if (diff_eval.value() == 0) {
    return producer_index;
  }

  return ir_builder.addExpr(
      producer_index, ir_builder.create<kir::Int>(diff_eval.value()));
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
  if (outer_it == index_map_.end() || inner_it == index_map_.end()) {
    return;
  }

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
    zero_domains_.emplace(in_id);
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
    // The extent should be updated only when its allocation is
    // partial, i.e., zero_merged_in is true. See PR #1270.
    if (zero_merged_in) {
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
    zero_domains_.emplace(outer_id);
    zero_domains_.emplace(inner_id);
    return;
  }

  if (!hasZeroMerged(out_id) && contig_ids.find(out_id) != contig_ids.end()) {
    // Contiguous indexing path
    auto input_ids = ir_utils::iterDomainInputsOfOrderedAs(
        {merge->out()}, td_->getMaybeRFactorDomain());

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
        zero_domains_.emplace(inner_id);
      } else {
        // Prop through inner
        index_map_[inner_id] = out_ind;
        extent_map_[inner_id] = getExtent(out_id);
        index_map_[outer_id] = zero;
        extent_map_[outer_id] = zero;
        zero_domains_.emplace(outer_id);
      }
    } else if (inner_id->isBroadcast() && !outer_id->isBroadcast()) {
      // Inner is broadcast and outer isn't, prop through outer
      index_map_[outer_id] = out_ind;
      extent_map_[outer_id] = getExtent(out_id);
      index_map_[inner_id] = zero;
      extent_map_[inner_id] = zero;
      zero_domains_.emplace(inner_id);
    } else {
      // Default to propagating through inner
      index_map_[inner_id] = out_ind;
      extent_map_[inner_id] = getExtent(out_id);
      index_map_[outer_id] = zero;
      extent_map_[outer_id] = zero;
      zero_domains_.emplace(outer_id);
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
    std::unordered_set<kir::IterDomain*> zero_domains,
    std::unordered_set<kir::IterDomain*> zero_merged_in,
    const std::vector<bool>& root_contiguity,
    std::unordered_set<kir::IterDomain*> preferred_paths,
    std::unordered_map<kir::IterDomain*, kir::Val*> reference_halo_extent_map)
    : td_(_td),
      index_map_(std::move(initial_index_map)),
      extent_map_(std::move(extent_map)),
      zero_domains_(std::move(zero_domains)),
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
        td_->domain(), td_->getMaybeRFactorDomain(), root_contiguity);
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
  // Pick from extent_map_ if available. Previously parallel
  // dimensions were ued (e.g., blockDim.x), however, it would result
  // in out-of-bounds errors when the extent of IterDomain is smaller
  // than the threading dimension.
  if (extent_map_.find(id) != extent_map_.end()) {
    return extent_map_.at(id);
  } else {
    return id->extent();
  }
}

bool IndexCompute::hasZeroMerged(kir::IterDomain* id) const {
  return zero_merged_in_.find(id) != zero_merged_in_.end() || isZero(id);
}

bool IndexCompute::isZero(kir::IterDomain* id) const {
  return zero_domains_.find(id) != zero_domains_.end();
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
  std::unordered_set<kir::IterDomain*> updated_zero_domains;
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

    if (zero_domains_.find(prev_id) != zero_domains_.end()) {
      updated_zero_domains.emplace(new_id);
    }

    if (zero_merged_in_.find(prev_id) != zero_merged_in_.end()) {
      updated_zero_merged_in.emplace(new_id);
    }
  }

  IndexCompute updated_index_compute(
      new_td,
      updated_index_map,
      updated_extent_map,
      updated_zero_domains,
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
    std::unordered_set<kir::IterDomain*> zero_domains,
    std::unordered_set<kir::IterDomain*> zero_merged_in)
    : IndexCompute(
          tv->domain(),
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_domains),
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

namespace {

// Used for local and shared index mapping. Returns a map from loops
// to loop indices as well as a set of loops that do not contribute to
// indexing.
std::pair<
    std::unordered_map<kir::ForLoop*, kir::Val*>,
    std::unordered_set<kir::ForLoop*>>
indexMapFromTV(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::pair<kir::ForLoop*, int64_t>& alloc_point,
    bool as_consumer) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto alloc_loop = alloc_point.first;

  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  const bool is_global = tv->getMemoryType() == MemoryType::Global;
  const bool is_shared = tv->getMemoryType() == MemoryType::Shared;
  const bool is_local = tv->getMemoryType() == MemoryType::Local;

  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map;

  // When indexed as a producer, the parallel types of the the
  // producer domains may not be the same as those of the loops, but
  // that's still valid parallelization. However, in that case, using
  // the parallel types of the loops to decide replacement of indices
  // with zero isn't valid. That's only valid when there's a matching
  // IterDomain in the producer tensor that has the same parallel
  // type.
  auto find_matching_parallel_domain = [tv](kir::IterDomain* id) -> bool {
    const auto gpu_lower = GpuLower::current();
    auto it = std::find_if(
        tv->domain()->domain().begin(),
        tv->domain()->domain().end(),
        [&](IterDomain* tv_id) {
          auto kir_tv_id = gpu_lower->lowerValue(tv_id)->as<kir::IterDomain>();
          // Matching is done using the index and loop maps. See
          // validateParallelize as well.
          return gpu_lower->caIndexMap().areMapped(id, kir_tv_id) ||
              (gpu_lower->caLoopMap().areMapped(id, kir_tv_id) &&
               ir_utils::derivedFromRootCAAxes(tv, tv_id));
        });
    if (it == tv->domain()->domain().end()) {
      return false;
    }

    auto corresponding_domain = *it;
    return corresponding_domain->getParallelType() == id->parallelType();
  };

  // Track domains that do not contibute to the resulting
  // index. Previously, index->isZeroInt() was used to detect such
  // domains, but that's not a reliable method as we may set an
  // initial index to zero for unswitch.
  std::unordered_set<kir::ForLoop*> zero_loops;

  for (auto loop : loops) {
    kir::Val* idx = nullptr;
    const auto same_parallel_type =
        as_consumer || find_matching_parallel_domain(loop->iter_domain());
    // See also LoopNestGenerator::pushAlloc.
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (!within_alloc) {
      if ((loop->iter_domain()->isThreadDim() && is_shared) ||
          (loop->iter_domain()->isThread() && is_global)) {
        idx = loop->index();
      } else {
        idx = ir_builder.zeroVal();
        zero_loops.insert(loop);
      }
    } else if (
        // For shared-memory tensors, when a domain is parallelized by
        // BID, the index can be replaced with zero as long as the
        // tensor has a matching domain that has the same parallel
        // type. Matching can be omitted when indexed as a consumer
        // since it is always the case. When indexed as a producer, to
        // replace it with zero, the same parallel type of BID must be
        // used by the producer tensor. Thus, since this is a shared
        // memory tensor, when a producer domain is parallelized by
        // BID, there must be a matching consumer domain with the same
        // parallel type, which must be the IterDomain of the
        // loop.
        (loop->iter_domain()->isBlockDim() && is_shared &&
         same_parallel_type) ||
        // Similarly for local memory tensors, zero replacement can be
        // only done when there's a matching domain with the same
        // parallel type
        (loop->iter_domain()->isThread() && is_local && same_parallel_type) ||
        loop->vectorize()) {
      idx = ir_builder.zeroVal();
      if (!loop->vectorize()) {
        zero_loops.insert(loop);
      }
    } else {
      idx = loop->index();
    }

    loop_to_ind_map[loop] = idx;

    if (!within_alloc && loop == alloc_loop) {
      within_alloc = true;
    }
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return {loop_to_ind_map, zero_loops};
}

//! Set "pragma unroll" required for loops that indexing of Local
//! tensors depends on.
//!
//! \param tv Indexed tensor
//! \param alloc_loop Allocation loop of tv
//! \param loops The current loop structure
//! \param id_map Producer-to-consumer map in case of indexing as producer
void ensureStaticIndexing(
    const TensorView* tv,
    kir::ForLoop* alloc_loop,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map = {}) {
  if (tv->getMemoryType() != MemoryType::Local) {
    return;
  }

  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  const auto gpu_lower = GpuLower::current();

  for (auto loop : loops) {
    if (!within_alloc) {
      if (loop == alloc_loop) {
        within_alloc = true;
      }
      continue;
    }
    kir::IterDomain* loop_id = loop->iter_domain();
    if (loop->vectorize() || loop_id->isThread()) {
      continue;
    }
    // Look for a domain that is mapped with the loop. If mapped in
    // the loop map, the loop index should be used for indexing of the
    // tensor, except for broadcast and reduction domains.
    auto it = std::find_if(
        tv->domain()->domain().begin(),
        tv->domain()->domain().end(),
        [loop_id, gpu_lower, &id_map](IterDomain* id) {
          if (id->isBroadcast() || id->isReduction() || id->isStride()) {
            return false;
          }
          auto id_replacement = id_map.find(id);
          if (id_replacement != id_map.end()) {
            id = id_replacement->second;
          }
          auto kir_id = gpu_lower->lowerValue(id)->as<kir::IterDomain>();
          return gpu_lower->caLoopMap().areMapped(loop_id, kir_id);
        });
    if (it != tv->domain()->domain().end()) {
      loop->requireUnroll();
    }
  }
}

// Map everything we can from reference to provided tv using the provided
// compute at map. If root_only is true, only root domains are included.
// We can't simply try to use the provided tv root domains and
// map those to the reference as the provided tv may have root domains that
// don't exist in reference. This can happen when the provided tv is from before
// a view, but all the loops are generated from TVs generated after the view
// operation.
std::unordered_map<IterDomain*, IterDomain*> indexMapReferenceTo(
    const TensorView* tv,
    const ComputeAtMap& ca_map,
    const std::unordered_map<IterDomain*, IterDomain*>&
        reference_concrete_to_id_map,
    bool root_only = false) {
  std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_producer;

  auto gen_map = [&](const auto& pids) {
    for (auto p_id : pids) {
      auto concrete_id = ca_map.getConcreteMappedID(p_id);
      auto ref_id_it = reference_concrete_to_id_map.find(concrete_id);
      if (ref_id_it != reference_concrete_to_id_map.end()) {
        index_map_ref_to_producer[ref_id_it->second] = p_id;
      }
    }
  };

  if (root_only) {
    gen_map(tv->getRootDomain());
  } else {
    auto all_pid_vals = DependencyCheck::getAllValsBetween(
        {tv->getRootDomain().begin(), tv->getRootDomain().end()},
        {tv->domain()->domain().begin(), tv->domain()->domain().end()});
    auto all_pids = ir_utils::filterByType<IterDomain>(all_pid_vals);
    gen_map(all_pids);
  }

  return index_map_ref_to_producer;
}

} // namespace

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

  // Replay producer to look like consumer so we can index on producer since
  // our loop nests look like consumer
  auto pairwise_map = PairwiseRootDomainMap(producer_tv, consumer_tv);
  auto producerAsC =
      TransformReplay::replayPasC(producer_tv, consumer_tv, -1, pairwise_map)
          .first;

  // Make the producer_tv look like consumer while performing indexing math
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // Map everything we can from reference to producer using compute at index
  // map. Use consumer as a proxy between producer and the generated reference.
  std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_producer;
  {
    // This replay has to be consistent with compute at index map.
    BestEffortReplay replay_producer_as_consumer(
        producer_tv->domain()->domain(),
        consumer_tv->domain()->domain(),
        pairwise_map.mapConsumerToProducer(
            consumer_tv->domain(), producer_tv->domain()));

    const auto& c2p_map = replay_producer_as_consumer.getReplay();

    std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_consumer =
        indexMapReferenceTo(
            consumer_tv, gpu_lower->caIndexMap(), reference_id_map);

    for (auto entry : index_map_ref_to_consumer) {
      auto r_id = entry.first;
      auto c_id = entry.second;
      auto c2p_it = c2p_map.find(c_id);
      if (c2p_it != c2p_map.end()) {
        auto p_id = c2p_it->second;
        index_map_ref_to_producer[r_id] = p_id;
      }
    }
  }

  // Index into the reference tensor. Reference indexing will handle vectorized
  // dims where index should be set to 0
  auto ref_compute = getReferenceIndexing(loops, reference_domain);

  // Forward vectorized IDs to index into producer correctly
  // We want p_id to be vectorized like consumer just for the indexing, then we
  // need to switch it back later. Store previous state here when changing. We
  // need to do this as replaying producer as consumer can use replay best
  // effort which means some domains may be producer's original domains.
  std::vector<std::pair<IterDomain*, ParallelType>> p_id_backup;
  for (auto entry : index_map_ref_to_producer) {
    auto ref_id = entry.first;
    auto p_id = entry.second;
    if (ref_id->getParallelType() == ParallelType::Vectorize) {
      p_id_backup.emplace_back(std::make_pair(p_id, p_id->getParallelType()));
      p_id->parallelize(ParallelType::Vectorize);
    } else if (ref_id->getParallelType() == ParallelType::MisalignedVectorize) {
      p_id->parallelize(ParallelType::MisalignedVectorize);
    }
  }

  // Adds halo info mappings for the reference
  updateHaloInfoForReference(reference, consumer_tv);

  const auto reference_halo_extent_map =
      getReferenceHaloExtentMap(reference, index_map_ref_to_producer);

  // Index into producer using reference indexing
  auto producer_indexing = ref_compute.updateIndexCompute(
      producer_tv->domain(),
      index_map_ref_to_producer,
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
    for (const auto i : c10::irange(root_dom.size())) {
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

  TORCH_INTERNAL_ASSERT(
      root_dom.size() == producer_tv->domain()->contiguity().size());
  kir::Val* cur_contig_stride = ir_builder.create<kir::Int>(1);
  for (const auto i : c10::irange(root_dom.size())) {
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
    } else if (root_dom[dim]->getIterType() == IterType::BroadcastWithStride) {
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
        root_ind,
        i,
        producer_tv,
        consumer_tv,
        reference_id_map,
        ref_compute.indexMap());

    root_ind = getProducerIndexWithPartialSplit(
        root_ind, root_dom[i], producer_tv, consumer_tv);

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

  // This map has forwarded broadcast axes, it should only be used to compute
  // the allocation position of the producer, and to figure out which producer
  // indices are mapped to consumer trivial reductions.
  std::unordered_map<IterDomain*, IterDomain*> p2c_alloc_map;
  {
    //  We want to play producer as consumer instead of the other way around
    //  since consumer may have some broadcasted axes producer doesn't have
    //  merged into loops producer may use. If we did consumer as producer we
    //  wouldn't have this information in the mapping.
    auto replay_PasC = BestEffortReplay::replayPasC(
        producer_tv, consumer_tv, -1, pairwise_map);

    auto c2p_map = replay_PasC.getReplay();

    // Grab consumer domain entries and reverse replay map. TODO: Maybe
    // TransformReplay::replayPasC could return this map
    for (auto id : consumer_tv->domain()->domain()) {
      auto c2p_it = c2p_map.find(id);
      if (c2p_it != c2p_map.end()) {
        auto c_id = c2p_it->first;
        auto p_id = c2p_it->second;
        p2c_alloc_map[p_id] = c_id;
      }
    }
  }

  // Find allocation point of producer relative to loop nests. P2C map is
  // required because producer was replayed as consumer, so we can't use the
  // regular compute at maps to line up its iter domains with the for loops.
  auto alloc_point =
      loop_utils::getAllocPoint(producer_tv, loops, p2c_alloc_map, true);
  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map;
  std::unordered_set<kir::ForLoop*> zero_loops;
  std::tie(loop_to_ind_map, zero_loops) =
      indexMapFromTV(producer_tv, loops, alloc_point, false);

  ensureStaticIndexing(producer_tv, alloc_point.first, loops, p2c_alloc_map);

  // Map loop nests to indicies, zeroing out those not used due to locality of
  // memory
  std::unordered_map<kir::IterDomain*, kir::Val*> ref_id_to_ind_map;
  // Track which domains are not used
  std::unordered_set<kir::IterDomain*> ref_zero_domains;

  // Due to rfactor/initialization reference_domain may be bigger than loop nest
  // structure, ignore IterDomains that aren't present in the loop nest when
  // indexing reference.
  TORCH_INTERNAL_ASSERT(loops.size() <= reference_domain->nDims());
  for (const auto loop_i : c10::irange(loops.size())) {
    auto ref_axis = gpu_lower->lowerValue(reference_domain->axis(loop_i))
                        ->as<kir::IterDomain>();
    ref_id_to_ind_map[ref_axis] = loop_to_ind_map[loops[loop_i]];
    if (zero_loops.count(loops[loop_i]) > 0) {
      ref_zero_domains.insert(ref_axis);
    }
  }

  // Map everything we can from reference to producer using compute at index
  // map. All producer id's don't exist in the compute at map. The rfactor axes
  // all may be, but since I haven't proven that to be the case, going to do a
  // more conservative approach, which is to use the consumer as a proxy between
  // producer to reference.
  std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_producer;
  {
    // This replay has to be consistent with compute at index map.
    BestEffortReplay replay_producer_as_consumer(
        producer_tv->domain()->domain(),
        consumer_tv->domain()->domain(),
        pairwise_map.mapConsumerToProducer(
            consumer_tv->domain(), producer_tv->domain()));

    const auto& c2p_map = replay_producer_as_consumer.getReplay();

    std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_consumer =
        indexMapReferenceTo(
            consumer_tv, gpu_lower->caIndexMap(), reference_id_map);

    for (auto entry : index_map_ref_to_consumer) {
      auto r_id = entry.first;
      auto c_id = entry.second;
      auto c2p_it = c2p_map.find(c_id);
      if (c2p_it != c2p_map.end()) {
        auto p_id = c2p_it->second;
        index_map_ref_to_producer[r_id] = p_id;
      }
    }
  }

  // Grab roots that map into producer and save them into the preferred roots
  // set for references indexing
  std::unordered_set<IterDomain*> preferred_roots;
  for (auto entry : index_map_ref_to_producer) {
    if (entry.second->isBroadcast() || entry.second->isReduction() ||
        entry.second->isStride()) {
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
      loops,
      reference_domain,
      ref_id_to_ind_map,
      ref_zero_domains,
      preferred_paths);

  // Forward vectorized IDs to index into producer correctly
  // We want p_id to be vectorized like consumer just for the indexing, then we
  // need to switch it back later. Store previous state here when changing. We
  // need to do this as replaying producer as consumer can use replay best
  // effort which means some domains may be the originals.
  std::vector<std::pair<IterDomain*, ParallelType>> p_id_backup;
  for (auto entry : index_map_ref_to_producer) {
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

  // Adds halo info mappings for the reference
  updateHaloInfoForReference(reference, consumer_tv);

  const auto reference_halo_extent_map =
      getReferenceHaloExtentMap(reference, index_map_ref_to_producer);

  auto producer_indexing = ref_compute.updateIndexCompute(
      producer_tv->domain(),
      index_map_ref_to_producer,
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
      producer_indexing.zeroDomains(),
      producer_indexing.zeroMergedIn());

  index_swizzle.run();

  const auto& index_map = index_swizzle.indexMap();
  const auto& extent_map = producer_indexing.extentMap();
  const auto& zero_domain_map = producer_indexing.zeroDomains();
  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();

  // Figure out which root axes we don't need to index
  std::unordered_set<IterDomain*> skip_indexing;

  for (auto root_id : root_dom) {
    // Already taken care of because we can detect no indexing required
    if (root_id->isBroadcast() || root_id->isReduction() ||
        gpu_lower->trivialReductionInfo().isDerived(root_id) ||
        root_id->isStride()) {
      skip_indexing.insert(root_id);
      continue;
    }

    // Already an entry for this root domain, continue
    if (index_map.find(gpu_lower->lowerValue(root_id)->as<kir::IterDomain>()) !=
        index_map.end()) {
      continue;
    }

    // Maps to consumers trivial reduction, don't index
    if (p2c_alloc_map.find(root_id) != p2c_alloc_map.end() &&
        gpu_lower->trivialReductionInfo().isDerived(
            p2c_alloc_map.at(root_id))) {
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
        root_ind_i,
        i,
        producer_tv,
        consumer_tv,
        reference_id_map,
        ref_compute.indexMap());

    root_ind_i = getProducerIndexWithPartialSplit(
        root_ind_i, root_dom[i], producer_tv, consumer_tv);

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    kir::Val* stride = nullptr;
    for (const auto j : c10::irange(i + 1, root_dom.size())) {
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

      auto root_ext_j = extent_map.find(kir_root_dom_j) == extent_map.end()
          ? kir_root_dom_j->extent()
          : extent_map.at(kir_root_dom_j);

      root_ext_j = getHaloExtentOfRootAxis(root_dom[j], root_ext_j);

      if (zero_domain_map.count(kir_root_dom_j) == 0) {
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
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  // Get a reference tensor replayed as existing loop structure
  auto reference = IndexReferenceReplay::getReference(loops);
  auto reference_domain = reference.domain;
  auto reference_id_map = reference.concrete_to_id;

  // Map everything we can from reference to consumer using compute at index
  // map.
  std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_consumer =
      indexMapReferenceTo(
          consumer_tv, gpu_lower->caIndexMap(), reference_id_map);

  // Index into the reference tensor. Reference indexing will handle vectorized
  // dims where index should be set to 0
  auto ref_compute = getReferenceIndexing(loops, reference_domain);

  // Index into consumer using reference indexing

  // Adds halo info mappings for the reference
  updateHaloInfoForReference(reference, consumer_tv);

  const auto reference_halo_extent_map =
      getReferenceHaloExtentMap(reference, index_map_ref_to_consumer);

  auto consumer_indexing = ref_compute.updateIndexCompute(
      consumer_tv->domain(),
      index_map_ref_to_consumer,
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
    for (const auto i : c10::irange(root_dom.size())) {
      if (root_dom[i]->isReduction() ||
          root_dom[i]->getIterType() == IterType::BroadcastWithoutStride ||
          root_dom[i]->isStride()) {
        strides[i] = zero;
        continue;
      }
      std::stringstream ss;
      ss << "T" << consumer_tv->name() << ".stride[" << stride_i++ << "]";
      strides[i] = ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int);
    }
  }

  TORCH_INTERNAL_ASSERT(
      root_dom.size() == consumer_tv->domain()->contiguity().size());
  kir::Val* cur_contig_stride = ir_builder.oneVal();
  for (const auto i : c10::irange(root_dom.size())) {
    auto dim = root_dom.size() - i - 1;
    if (root_dom[dim]->isReduction() || root_dom[dim]->isStride()) {
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
    } else if (root_dom[dim]->getIterType() == IterType::BroadcastWithStride) {
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

  auto vectorize_shift =
      loops.empty() ? nullptr : loops.back()->vectorize_shift();

  // Global striding
  std::vector<kir::Val*> strided_inds(root_dom.size(), ir_builder.zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    // See a comment in indexing to root domains in getGlobalProducerIndex.
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride ||
        root_dom[i]->getIterType() == IterType::BroadcastWithStride ||
        gpu_lower->trivialReductionInfo().isDerived(root_dom[i]) ||
        root_dom[i]->isStride()) {
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

    root_ind = ir_builder.addExpr(
        root_ind, getGlobalConsumerOffsetWithPartialSplit(kir_root_dom_i));

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
  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map;
  std::unordered_set<kir::ForLoop*> zero_loops;
  std::tie(loop_to_ind_map, zero_loops) =
      indexMapFromTV(consumer_tv, loops, alloc_point, true);

  ensureStaticIndexing(consumer_tv, alloc_point.first, loops);

  // Map loop nests to indicies, zeroing out those not used due to locality of
  // memory
  std::unordered_map<kir::IterDomain*, kir::Val*> ref_id_to_ind_map;
  std::unordered_set<kir::IterDomain*> ref_zero_domains;

  // Due to rfactor/initialization reference_domain may be bigger than loop nest
  // structure, ignore IterDomains that aren't present in the loop nest when
  // indexing reference.
  TORCH_INTERNAL_ASSERT(loops.size() <= reference_domain->nDims());
  for (const auto loop_i : c10::irange(loops.size())) {
    auto ref_axis = gpu_lower->lowerValue(reference_domain->axis(loop_i))
                        ->as<kir::IterDomain>();
    ref_id_to_ind_map[ref_axis] = loop_to_ind_map[loops[loop_i]];
    if (zero_loops.count(loops[loop_i]) > 0) {
      ref_zero_domains.insert(ref_axis);
    }
  }

  // Map everything we can from reference to consumer using compute at index
  // map.
  std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_consumer =
      indexMapReferenceTo(
          consumer_tv, gpu_lower->caIndexMap(), reference_id_map);

  // Grab roots that map into consumer and save them into the preferred roots
  // set for references indexing
  std::unordered_set<IterDomain*> preferred_roots;
  for (auto entry : index_map_ref_to_consumer) {
    if (entry.second->isBroadcast() || entry.second->isReduction() ||
        entry.second->isStride()) {
      continue;
    }
    preferred_roots.emplace(entry.first);
  }

  // Make sure propagation of indexing while mixing with 0 indicies we propagate
  // in a way that consumer will be able to see what's going on.
  auto preferred_paths = buildPreferredPaths(reference_domain, preferred_roots);

  // Index into the reference tensor
  auto ref_compute = getReferenceIndexing(
      loops,
      reference_domain,
      ref_id_to_ind_map,
      ref_zero_domains,
      preferred_paths);

  // Adds halo info mappings for the reference
  updateHaloInfoForReference(reference, consumer_tv);

  const auto reference_halo_extent_map =
      getReferenceHaloExtentMap(reference, index_map_ref_to_consumer);

  // Index into consumer using reference indexing
  auto consumer_indexing = ref_compute.updateIndexCompute(
      consumer_tv->domain(),
      index_map_ref_to_consumer,
      consumer_tv->domain()->contiguity(),
      reference_halo_extent_map);

  IndexSwizzle index_swizzle(
      consumer_tv,
      consumer_indexing.indexMap(),
      consumer_indexing.extentMap(),
      consumer_indexing.zeroDomains(),
      consumer_indexing.zeroMergedIn());

  index_swizzle.run();

  const auto& index_map = index_swizzle.indexMap();
  const auto& extent_map = consumer_indexing.extentMap();
  const auto& zero_domain_map = consumer_indexing.zeroDomains();

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();
  std::vector<kir::Val*> strided_inds(root_dom.size(), ir_builder.zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast() ||
        gpu_lower->trivialReductionInfo().isDerived(root_dom[i]) ||
        root_dom[i]->isStride()) {
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
    for (const auto j : c10::irange(i + 1, root_dom.size())) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction() ||
          gpu_lower->trivialReductionInfo().isDerived(root_dom[j]) ||
          root_dom[j]->isStride()) {
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

      auto root_ext_j = extent_map.find(kir_root_dom_j) == extent_map.end()
          ? kir_root_dom_j->extent()
          : extent_map.at(kir_root_dom_j);

      root_ext_j = getHaloExtentOfRootAxis(root_dom[j], root_ext_j);

      if (zero_domain_map.count(kir_root_dom_j) == 0) {
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

namespace {

struct PredicateDomainInfo {
 public:
  // Iteration domain to predicate
  IterDomain* id = nullptr;
  // The set of iteration domains that make up the id. If this is for
  // a non-divisible split, the set only contains the id itself. This
  // set is used to remove redundant predicates when gathering
  // unswitch predicates.
  std::unordered_set<IterDomain*> covered_ids;
  // True if this predicate is for a non-divisible split
  bool is_non_divisible_split = false;
};

// Find iteration domains in the history of reference comprised only of
// merge operations. Only return iteration domains that are subsequently fed
// into a split, or are in the provided domain. In other words, we don't want to
// return every IterDomain that's contiguous, just the one closest to the
// leaves. Predicates are not associated with physical memory so we can treat
// all of them as contiguous merges.
std::vector<PredicateDomainInfo> getPredicateContigIds(
    const ReferenceTensor& reference,
    TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, IterDomain*>& ref_2_consumer) {
  const auto gpu_lower = GpuLower::current();

  std::vector<IterDomain*> reference_predicated_root_domain;
  for (const auto consumer_root : consumer_tv->getRootDomain()) {
    if (consumer_root->isBroadcast()) {
      continue;
    }
    auto consumer_root_concrete =
        gpu_lower->caIndexMap().getConcreteMappedID(consumer_root);
    auto it = reference.concrete_to_id.find(consumer_root_concrete);
    // When initializing a reduction buffer, the reduction axis
    // doesn't have a loop, so the reference tensor doesn't have a
    // mapped domain. The reduction axis can be safely ignored.
    if (it == reference.concrete_to_id.end()) {
      continue;
    }
    auto reference_root = it->second;
    reference_predicated_root_domain.emplace_back(reference_root);
  }

  std::vector<IterDomain*> contiguous_ids = reference_predicated_root_domain;

  if (contiguous_ids.empty()) {
    return std::vector<PredicateDomainInfo>();
  }

  // If root IDs are partial, i.e., start is non-zero and stop is not
  // equal to extent, predication can't be done with merged domains as
  // start and stop information is only available with root
  // domains. Similarly, merged domains don't have enough information
  // about halo to do correct predication, so they must be excluded.
  std::unordered_set<IterDomain*> excluded_ids;

  for (auto reference_predicated_id : reference_predicated_root_domain) {
    if (GpuLower::current()
            ->haloInfo()
            .getRootAxisInfo(reference_predicated_id)
            .hasHalo()) {
      continue;
    }
    auto it = ref_2_consumer.find(reference_predicated_id);
    if (it == ref_2_consumer.end()) {
      continue;
    }
    auto consumer_root_id = it->second;
    if (consumer_root_id->maybePartial()) {
      excluded_ids.insert(reference_predicated_id);
      continue;
    }
    // Shifted or gathered axes need to be predicated at the root domain
    auto shift_expr = dynamic_cast<ShiftOp*>(consumer_tv->definition());
    auto gather_expr = dynamic_cast<GatherOp*>(consumer_tv->definition());
    if (shift_expr == nullptr && gather_expr == nullptr) {
      continue;
    }
    auto consumer_root_pos = consumer_tv->domain()->rootPosOf(consumer_root_id);
    if ((shift_expr && shift_expr->offset(consumer_root_pos) != 0) ||
        (gather_expr && consumer_root_pos < gather_expr->windowShape().size() &&
         !gather_expr->windowShape().at(consumer_root_pos)->isOneInt())) {
      excluded_ids.insert(reference_predicated_id);
    }
  }

  // Run through iteration domain history
  auto exprs = ExprSort::getExprs(
      consumer_tv->fusion(),
      {reference.domain->domain().begin(), reference.domain->domain().end()});

  for (auto expr : exprs) {
    // If not a merge, output is not contiguous
    if (expr->isA<Merge>()) {
      auto merge = expr->as<Merge>();
      auto inner_contig_it = std::find(
          contiguous_ids.begin(), contiguous_ids.end(), merge->inner());
      auto outer_contig_it = std::find(
          contiguous_ids.begin(), contiguous_ids.end(), merge->outer());

      if (excluded_ids.count(merge->inner()) > 0 ||
          excluded_ids.count(merge->outer()) > 0) {
        continue;
      }

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

  std::vector<PredicateDomainInfo> contig_id_infos;

  // Create entries and return them
  for (auto contig_id : contiguous_ids) {
    // Pick inputs from the starting domains, i.e.,
    // reference_predicated_root_domain.
    auto contig_root_vals = IterVisitor::getInputsTo(
        {contig_id},
        {reference_predicated_root_domain.begin(),
         reference_predicated_root_domain.end()});
    auto contig_root_ids = ir_utils::filterByType<IterDomain>(contig_root_vals);
    PredicateDomainInfo contig_id_info;
    contig_id_info.id = contig_id;
    contig_id_info.covered_ids = std::unordered_set<IterDomain*>(
        contig_root_ids.begin(), contig_root_ids.end());
    contig_id_infos.push_back(contig_id_info);
  }
  return contig_id_infos;
}

IterDomain* getMappedReferenceDomain(
    IterDomain* id,
    const ReferenceTensor& reference) {
  // Partially overlaps with getPredicateContigIds()
  const auto gpu_lower = GpuLower::current();
  auto concrete_id = gpu_lower->caIndexMap().getConcreteMappedID(id);
  auto it = reference.concrete_to_id.find(concrete_id);
  if (it == reference.concrete_to_id.end()) {
    return nullptr;
  }
  return it->second;
}

std::vector<PredicateDomainInfo> getNonDivisibleReferenceDomainsToPredicate(
    TensorView* consumer_tv,
    const ReferenceTensor& reference) {
  const auto& non_divisible_split_info =
      GpuLower::current()->nonDivisibleSplitInfo();

  std::vector<PredicateDomainInfo> pred_info_vec;

  auto it = non_divisible_split_info.splitsToPredicate().find(consumer_tv);
  if (it == non_divisible_split_info.splitsToPredicate().end()) {
    return {};
  }

  const auto& splits_to_predicate = it->second;

  for (auto split : splits_to_predicate) {
    auto ref_id = getMappedReferenceDomain(split->in(), reference);
    if (ref_id == nullptr) {
      continue;
    }
    PredicateDomainInfo info{ref_id, {ref_id}, true};
    pred_info_vec.emplace_back(info);
  }

  return pred_info_vec;
}

bool needsPadding(TensorView* tv) {
  auto shift_expr = dynamic_cast<ShiftOp*>(tv->definition());
  auto gather_expr = dynamic_cast<GatherOp*>(tv->definition());

  // Padding is only necessary for padded shift and
  // gather
  return (shift_expr != nullptr && shift_expr->pad()) || gather_expr != nullptr;
}

// Get an additional offset of a stop index when building a predicate
// for unswitch. Initial stop indices generated at getPredicateReferenceIndexing
// do not take halo into account, and the adjustment for halo is done as an
// additional offset to the final index value so that unswitch predicates can be
// compared with each other by just looking at the additional offsets.
//
// consumer_root_id: the domain for which a stop predicate is being built.
kir::Val* getUnswitchStopOffset(
    IterDomain* consumer_root_id,
    TensorView* consumer_tv) {
  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  AxisHaloInfo halo_info =
      gpu_lower->haloInfo().getRootAxisInfo(consumer_root_id);

  // If the consumer root domain to predicate does not have halo, no
  // adjustment is required.
  if (!halo_info.hasHalo()) {
    return ir_builder.zeroVal();
  }

  // Find if this contig_id is used in the unswitched domains
  auto unswitch_it = std::find_if(
      consumer_tv->domain()->domain().begin(),
      consumer_tv->domain()->domain().end(),
      [](IterDomain* id) {
        return id->getParallelType() == ParallelType::Unswitch ||
            id->getParallelType() == ParallelType::Unroll ||
            id->getParallelType() == ParallelType::Vectorize;
      });

  // If any of the unswitched leaf domains inherits the halo from the
  // root domain, the halo width needs to be added to the stop offset
  if (std::any_of(
          unswitch_it,
          consumer_tv->domain()->domain().end(),
          [&gpu_lower, &consumer_root_id](auto leaf_id) {
            return gpu_lower->haloInfo().isHaloInherited(
                consumer_root_id, leaf_id);
          })) {
    return halo_info.width();
  } else {
    return ir_builder.zeroVal();
  }
}

// Get offsets for the start and stop predicates. Similar to the
// gather case, but it's a little simpler as it does not (yet)
// dynamic shifting.
void adjustStartAndStopOffsetsForShift(
    std::vector<kir::Val*>& start_offsets,
    std::vector<kir::Val*>& stop_offsets,
    TensorView* consumer_tv,
    IterDomain* consumer_id,
    bool padding_predicate) {
  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  TORCH_INTERNAL_ASSERT(consumer_id != nullptr);

  auto shift_expr = dynamic_cast<ShiftOp*>(consumer_tv->definition());

  // Adjustment is not necessary if not shift.
  // Even so, padding predicate does not need any adjustment.
  if (shift_expr == nullptr || padding_predicate) {
    return;
  }

  const auto root_axis_pos = consumer_tv->domain()->rootPosOf(consumer_id);

  // Assume this adjustment is done first, so start and stop offsets
  // just contain zeroVal.
  TORCH_INTERNAL_ASSERT(
      start_offsets.size() == 1 && start_offsets[0]->isZeroInt() &&
      stop_offsets.size() == 1 && stop_offsets[0]->isZeroInt());
  start_offsets.clear();
  stop_offsets.clear();

  // The consumer offset is zero.
  auto consumer_offset = 0;
  // The producer offset is based off the consumer offset.
  auto producer_offset = 0;

  // When the shift operation is not padded, the start and stop positions of the
  // consumer axis, i.e., consumer_id->start and
  // consumer_id->stop_ofset, are adjusted accordingly, which includes
  // the effect of the shift offset, so using the consumer offset is
  // sufficient as the only predicate is sufficient.

  if (shift_expr->pad()) {
    // Positive shift offset means shifting the input tensor to the
    // positive direction, so the producer offset becomes negative.
    auto shift_offset = shift_expr->offset(root_axis_pos);
    producer_offset = -shift_offset;
  }

  // Since shift doesn't allow dynamic offsets, we can statically
  // choose more restrictive offsets between the producer and consumer
  // offsets. The start predicate uses greater-than, so using the
  // smaller offset is sufficient. Similarly, for the stop predicate,
  // using the larger offset is sufficient.
  auto start_offset = std::min(consumer_offset, producer_offset);
  auto stop_offset = std::max(consumer_offset, producer_offset);

  start_offsets.push_back(ir_builder.create<kir::Int>(start_offset));
  stop_offsets.push_back(ir_builder.create<kir::Int>(stop_offset));
}

// Get offsets for the start and stop predicates. There can be two
// offsets because the shift offset is determined by a loop index.
void adjustStartAndStopOffsetsForGather(
    std::vector<kir::Val*>& start_offsets,
    std::vector<kir::Val*>& stop_offsets,
    TensorView* consumer_tv,
    IterDomain* consumer_id,
    const ReferenceTensor& reference,
    const std::unordered_map<kir::IterDomain*, kir::Val*>& ref_start_index_map,
    const std::unordered_map<kir::IterDomain*, kir::Val*>& ref_stop_index_map,
    bool padding_predicate) {
  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  TORCH_INTERNAL_ASSERT(consumer_id != nullptr);

  // Adjustment is not necessary if not gather. Even so, padding
  // predicate does not need any adjustment.
  if (!consumer_tv->definition()->isA<GatherOp>() || padding_predicate) {
    return;
  }

  const auto root_axis_pos = consumer_tv->domain()->rootPosOf(consumer_id);

  // Assume this adjustment is done first, so start and stop offsets
  // just contain zeroVal.
  TORCH_INTERNAL_ASSERT(
      start_offsets.size() == 1 && start_offsets[0]->isZeroInt() &&
      stop_offsets.size() == 1 && stop_offsets[0]->isZeroInt());
  start_offsets.clear();
  stop_offsets.clear();

  auto producer_start_offset = getProducerOffsetWithGather(
      root_axis_pos,
      consumer_tv,
      reference.concrete_to_id,
      ref_start_index_map);

  auto producer_stop_offset = getProducerOffsetWithGather(
      root_axis_pos, consumer_tv, reference.concrete_to_id, ref_stop_index_map);

  // The producer and consumer accesses must be predicated as it is
  // not statically determined which is more restrictive.

  // Consumer offsets are just zero.
  start_offsets.push_back(ir_builder.zeroVal());
  stop_offsets.push_back(ir_builder.zeroVal());

  // Adds producer offsets if they are not zero.
  if (!producer_start_offset->isZeroInt()) {
    start_offsets.push_back(producer_start_offset);
  }

  if (!producer_stop_offset->isZeroInt()) {
    stop_offsets.push_back(producer_stop_offset);
  }
}

// Get the start and stop limit offsets that define the valid range to
// compute. In the simplest case, they are just 0 and
// IterDomain::extent. However, IterDomain may have non-zero start and
// stop that's different from extent. Also, when IterDomain has halo,
// the actual offsets of the logical start and stop positions are
// shifted.
std::pair<kir::Val*, kir::Val*> getStartAndStopLimitOffsets(
    IterDomain* consumer_id,
    bool padding_predicate,
    bool non_divisible_pred) {
  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  TORCH_INTERNAL_ASSERT(consumer_id != nullptr);

  kir::Val* start_limit = gpu_lower->lowerValue(consumer_id->start());
  kir::Val* stop_limit =
      ir_builder.negExpr(gpu_lower->lowerValue(consumer_id->stopOffset()));

  if (!non_divisible_pred) {
    AxisHaloInfo halo_info = gpu_lower->haloInfo().getRootAxisInfo(consumer_id);

    // Below, "left" and "right" halo mean halo at offset zero and
    // axis extent, respectively.
    //
    // The consumer axis looks like this:
    //
    // [0, left halo)[start_limit, stop_limit)[0, right halo)
    //
    if (!padding_predicate) {
      start_limit = ir_builder.addExpr(start_limit, halo_info.width(0));
      stop_limit = ir_builder.addExpr(stop_limit, halo_info.width(0));
    } else {
      // In case of the padding predicate, the whole range, including both left
      // and right halo regions, is computed.
      stop_limit = ir_builder.addExpr(stop_limit, halo_info.width());
    }
  } else {
    // For non-divisible predicates, the index must be predicated such
    // that it is less than the extent of the predicated ID +
    // halo. Note that getRootAxisInfo doesn't work since consumer_id
    // isn't a root domain.
    if (gpu_lower->haloInfo().hasHaloWidth(consumer_id)) {
      auto halo = gpu_lower->haloInfo().getHaloWidth(consumer_id);
      stop_limit = ir_builder.addExpr(stop_limit, halo);
    }
  }

  return {start_limit, stop_limit};
}

// Return an index map for a predicate reference tensor. Two different
// maps are used when generating predicates for unswitched expressions
// as start and stop conditions need to use different loop-to-index
// mappings.
std::unordered_map<kir::IterDomain*, kir::Val*> getPredicateReferenceIndexing(
    const std::vector<kir::ForLoop*>& loops,
    const ReferenceTensor& reference,
    kir::ForLoop* unswitch_or_vec_loop,
    bool start) {
  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  auto reference_domain = reference.domain;

  std::unordered_map<kir::ForLoop*, kir::Val*> loop_to_ind_map;

  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  // If unswitch don't directly use indices from for loop, use zero
  // and for loop extent minus 1
  if (unswitch_or_vec_loop != nullptr) {
    // Vectorized predicates are different from unswitch. Unswitch predicates
    // all loops within the unswitch (the outer most unswitch) are generated
    // with loop->extent-1 as the index. With vectorized predicates, only the
    // vectorized loop should be like this.

    bool vectorized_pred =
        unswitch_or_vec_loop->iter_domain()->parallelType() ==
        ParallelType::Vectorize;

    TORCH_INTERNAL_ASSERT(
        loops.size() <= reference_domain->nDims(),
        "Invalid reference generated.");

    bool within_unswitch = false;
    const auto one = ir_builder.oneVal();

    for (const auto loop_i : c10::irange(loops.size())) {
      auto loop = loops[loop_i];
      auto loop_id = loop->iter_domain();
      auto loop_pt = loop_id->parallelType();
      auto ref_id = reference_domain->axis(loop_i);

      if (loop == unswitch_or_vec_loop) {
        within_unswitch = true;
      }

      if (within_unswitch) {
        // Rely on the reference to check broadcasting. The for loop could be
        // broadcasted on a constant value from an unroll split. Since reference
        // may convert this to an iter domain, that for loop could be valid to
        // generate predication from.

        // Note that loop->stop() is not used below. Instead,
        // loop->iter_domain()->extent() is used, which is uniform
        // across the mapped domains irrespective of halo. Predicates are
        // compared with each to pick the most restrictive ones. The
        // comparison is done by only using the offset, which is the
        // term added to the index. So, the index term must be the
        // same among all predicates, otherwise the comparison would
        // be invalid. The effect by halo is added to the offset
        // term. See getUnswitchStopOffset.

        if (ref_id->isBroadcast()) {
          // Ignore indexing into broadcasted dimensions.
          continue;
        } else if (loop_id->isThread()) {
          // When parallelized, if the loop stop is the same as the
          // extent of the associated IterDomain, i.e., no extra
          // iterations for halo, predicating with the threading index
          // is sufficient for both the start and stop
          // predicates. That isn't the case if the loop has halo, and
          // in the case either the minimum and maximum values of the
          // iteration domain needs to be used.
          //
          // Note: Better performance was obtained if using
          // threadIdx in unswitch predicates was avoided. More
          // specifically, in the Hdiff stencil example, instead of
          // predicating with threadIdx.x for both the start and stop
          // predicates, using zero and (blockDim.x - 1) for the start
          // and stop predicates, respectively, resulted in less
          // register pressure. The alternative codegen can be done by
          // adding this to the first if condition:
          // loop_id->isBlockDim(). This would not be a concern if the
          // else part could be omitted, so canOmitElseClause should
          // be used as well.
          if (loop->stop() == loop_id->extent()) {
            loop_to_ind_map[loop] = loop->start();
          } else if (start) {
            loop_to_ind_map[loop] = ir_builder.zeroVal();
          } else {
            // Note that the parallel dimension is used rather than
            // loop-stop(). See the above comment.
            loop_to_ind_map[loop] = ir_builder.subExpr(
                gpu_lower->parallelDimensionMap().get(loop_pt),
                ir_builder.create<kir::Int>(1));
          }
        } else if (start) {
          loop_to_ind_map[loop] = ir_builder.zeroVal();
        } else {
          // Similar to the above, loop_id()->extent() is
          // used here instead of loop->stop(). See the above comment.
          loop_to_ind_map[loop] = ir_builder.subExpr(loop_id->extent(), one);
        }
      }

      // If a vectorized predicate, bail after the vectorized loop was found.
      // Don't continue unswitching loops.
      if (vectorized_pred && within_unswitch) {
        break;
      }
    }
  }

  // Add magic zero to a loop pretty far inside in indexing
  kir::IterDomain* magic_zero_loop = nullptr;
  std::unordered_map<kir::IterDomain*, kir::Val*> ref_id_to_ind_map;
  // Due to rfactor/initialization reference_domain may be bigger than loop nest
  // structure
  TORCH_INTERNAL_ASSERT(loops.size() <= reference_domain->nDims());
  for (const auto loop_i : c10::irange(loops.size())) {
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
    auto& ind = ref_id_to_ind_map[magic_zero_loop];
    if (!ind->isConstScalar()) {
      ind = ir_builder.addExpr(ind, ir_builder.magicZeroVal());
    }
  }

  std::unordered_map<IterDomain*, IterDomain*> ref_self_map;
  auto all_vals = DependencyCheck::getAllValsBetween(
      {reference_domain->getRootDomain().begin(),
       reference_domain->getRootDomain().end()},
      {reference_domain->domain().begin(), reference_domain->domain().end()});
  auto all_ids = ir_utils::filterByType<IterDomain>(all_vals);
  std::for_each(all_ids.begin(), all_ids.end(), [&ref_self_map](auto id) {
    ref_self_map.insert({id, id});
  });

  std::unordered_map<kir::IterDomain*, kir::Val*> reference_halo_extent_map =
      getReferenceHaloExtentMap(reference, ref_self_map);

  // Index into the reference tensor
  auto index_compute = getReferenceIndexing(
      loops,
      reference_domain,
      ref_id_to_ind_map,
      {},
      {},
      reference_halo_extent_map);

  return index_compute.indexMap();
}

// Get the offsets for the start and stop predicates. The offsets
// are to be added to the index.
std::pair<std::vector<kir::Val*>, std::vector<kir::Val*>> getStartAndStopOffsets(
    IterDomain* consumer_id,
    TensorView* consumer_tv,
    const ReferenceTensor& reference,
    const std::unordered_map<kir::IterDomain*, kir::Val*>& ref_start_index_map,
    const std::unordered_map<kir::IterDomain*, kir::Val*>& ref_stop_index_map,
    bool padding_predicate,
    bool unswitch,
    bool non_divisible_pred) {
  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  // By default, the offsets for the start and stop predicates are
  // just zero.
  std::vector<kir::Val*> start_offsets{ir_builder.zeroVal()};
  std::vector<kir::Val*> stop_offsets{ir_builder.zeroVal()};

  if (consumer_id == nullptr) {
    return {start_offsets, stop_offsets};
  }

  auto consumer_def = consumer_tv->definition();

  // These adjustments are not required when predicating non-divisible splits
  if (!non_divisible_pred) {
    if (consumer_def->isA<ShiftOp>()) {
      adjustStartAndStopOffsetsForShift(
          start_offsets,
          stop_offsets,
          consumer_tv,
          consumer_id,
          padding_predicate);
    } else if (consumer_def->isA<GatherOp>()) {
      adjustStartAndStopOffsetsForGather(
          start_offsets,
          stop_offsets,
          consumer_tv,
          consumer_id,
          reference,
          ref_start_index_map,
          ref_stop_index_map,
          padding_predicate);
    }

    // Adjustment for partial split
    auto partial_split_offset = getGlobalConsumerOffsetWithPartialSplit(
        gpu_lower->lowerValue(consumer_id)->as<kir::IterDomain>());
    for (auto& start_offset : start_offsets) {
      start_offset = ir_builder.addExpr(start_offset, partial_split_offset);
    }
    for (auto& stop_offset : stop_offsets) {
      stop_offset = ir_builder.addExpr(stop_offset, partial_split_offset);
    }

    // If generating a predicate for unswitch, adjust the stop offset to
    // accommodate the addition of halo to the loop stop. See the
    // comment in getPredicateReferenceIndexing as well.
    if (unswitch) {
      TORCH_INTERNAL_ASSERT(
          !padding_predicate, "Unswitch should not use the padding predicate");
      auto stop_unswitch_offset =
          getUnswitchStopOffset(consumer_id, consumer_tv);
      for (auto& stop_offset : stop_offsets) {
        stop_offset = ir_builder.addExpr(stop_offset, stop_unswitch_offset);
      }
    }
  }

  // Get the boundaries of two ends
  auto limits = getStartAndStopLimitOffsets(
      consumer_id, padding_predicate, non_divisible_pred);

  // At this point, we have everything to create both start and stop
  // predicates as:
  //
  //  index + start_offset >= start_limit
  //  index + stop_offset  < extent + stop_limit
  //
  // In order to enable consolidating unswitch predicates, organize
  // the predicates as:
  //
  //  index + (start_offset - start_limit) >= 0
  //  index + (stop_offset - stop_limit)  < extent

  for (auto& start_offset : start_offsets) {
    start_offset = ir_builder.subExpr(start_offset, limits.first);
  }
  for (auto& stop_offset : stop_offsets) {
    stop_offset = ir_builder.subExpr(stop_offset, limits.second);
  }

  return {start_offsets, stop_offsets};
}

bool canOmitStartPredicate(kir::Val* start_offset) {
  // Start predicate can be omitted when start_offset >= 0.
  auto offset_val = start_offset->as<kir::Int>()->value();
  return offset_val.has_value() && offset_val.value() >= 0;
}

bool canOmitStopPredicate(
    kir::Val* stop_index,
    kir::Val* stop_offset,
    kir::IterDomain* kir_contig_id) {
  bool index_simple = stop_index->definition() == nullptr;
  // The definition may be just adding the magic zero, which can be
  // effectively considered "simple"
  if (!index_simple && isProtectedWithMagicZero(stop_index)) {
    // Make sure the lhs of stop_index is simple.
    auto lhs = stop_index->definition()->as<kir::BinaryOp>()->lhs();
    if (lhs->definition() == nullptr) {
      index_simple = true;
    }
  }

  // Omit only when both the index and extent are "simple".
  if (!(index_simple && kir_contig_id->extent()->definition() == nullptr)) {
    return false;
  }

  const auto gpu_lower = GpuLower::current();

  // Stop predicate: stop_index + stop_offset < extent, where
  // stop_index ranges from 0 to (extent + halo), so this can be
  // omitted if extent + halo + stop_offset < extent, i.e., halo +
  // stop_offset <= 0.

  auto stop_offset_val = stop_offset->as<kir::Int>()->value();

  auto halo_ext =
      gpu_lower->haloInfo().getRootAxisInfo(kir_contig_id).width()->value();

  // If they are not compile-time constant, can't prove the
  // condition.
  if (!stop_offset_val.has_value() || !halo_ext.has_value()) {
    return false;
  }

  if (halo_ext.value() + stop_offset_val.value() > 0) {
    return false;
  }

  // When the domain is parallelized, the parallel dimension must be
  // exact. Otherwise, there would be extra threads/blocks that need
  // to be predicated out.
  if (isParallelTypeThread(kir_contig_id->parallelType())) {
    if (!gpu_lower->parallelDimensionMap().isExact(
            kir_contig_id->parallelType())) {
      return false;
    }
    // If the domain has halo, the loop is expanded by the halo
    // extent, so we can't prove the loop extent is the same as the
    // parallel dimension.
    if (!(halo_ext.has_value() && halo_ext.value() == 0)) {
      return false;
    }
  }

  return true;
}

} // namespace

// Returns predicates and the concrete (by loop map) root domains they cover
std::pair<std::vector<RootPredicateInfo>, ReferenceTensor> Index::
    getReferenceRootPredicates(
        const kir::TensorView* kir_consumer_tv,
        const std::vector<kir::ForLoop*>& loops,
        kir::ForLoop* unswitch_or_vec_loop,
        bool shift_padding) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getReferenceRootPredicates");

  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  // Nothing needs to be done when padding is not required.
  if (shift_padding && !needsPadding(kir_consumer_tv->fuserTv())) {
    return {{RootPredicateInfo::getFalseInfo()}, ReferenceTensor{}};
  }

  auto consumer_tv = kir_consumer_tv->fuserTv();

  // Get a reference tensor replayed as existing loop structure
  ReferenceTensor reference = IndexReferenceReplay::getReference(loops);

  // Generate halo information for reference.
  updateHaloInfoForReference(reference, consumer_tv);

  // Both start and stop positions may need to be predicated. Indexing
  // differs when generating predicates for unswitch.
  // NOTE: If we could find-and-replace KIR nodes, we could just
  // generate one index map, clone it and replace the loop-to-index
  // mappings of unswitched loops for the start predicate.
  const auto ref_stop_index_map = getPredicateReferenceIndexing(
      loops, reference, unswitch_or_vec_loop, false);
  // If not unswitch, share the same indexing map as the stop index map
  const auto& ref_start_index_map = unswitch_or_vec_loop != nullptr
      ? getPredicateReferenceIndexing(
            loops, reference, unswitch_or_vec_loop, true)
      : ref_stop_index_map;

  auto ref_2_consumer = indexMapReferenceTo(
      consumer_tv, gpu_lower->caIndexMap(), reference.concrete_to_id);

  // Get the contiguous ids we need to generate predicates for
  auto contig_id_infos =
      getPredicateContigIds(reference, consumer_tv, ref_2_consumer);

  auto non_divisible_splits =
      getNonDivisibleReferenceDomainsToPredicate(consumer_tv, reference);
  contig_id_infos.insert(
      contig_id_infos.end(),
      non_divisible_splits.begin(),
      non_divisible_splits.end());

  std::vector<RootPredicateInfo> pred_info_vec;

  for (auto contig_id_entry : contig_id_infos) {
    auto contig_id = contig_id_entry.id;
    // No predicates needed for braodcasted indices.
    if (contig_id->isBroadcast() ||
        gpu_lower->trivialReductionInfo().isDerived(contig_id)) {
      continue;
    }

    auto root_ids = contig_id_entry.covered_ids;
    auto kir_contig_id =
        gpu_lower->lowerValue(contig_id)->as<kir::IterDomain>();

    const auto ref_stop_indexing_it = ref_stop_index_map.find(kir_contig_id);

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
    // It can also happens with rfactored reductions. The reference
    // tensor may include rfactored domains, so the contig id may be
    // a root domain of the reference, not a rfactor root. Since
    // there is no loop for rfactor domains, there's no indexing
    // mapping for root domains. This seems safe as it can only happen
    // with rfactor and rfactored tensors do not need predicates.
    //
    // Second condition is simply to avoid predication on broadcasting axes as
    // it's not required.
    if (ref_stop_indexing_it == ref_stop_index_map.end() ||
        ref_stop_indexing_it->second->isZeroInt()) {
      continue;
    }

    // Find a corresponding consumer root id if exists. Used to
    // support shift. If a contig_id is a merged non-root domain, nothing
    // is required to do for shift as shift-related domains are
    // excluded from contig domains.
    IterDomain* consumer_id = nullptr;
    if (contig_id->definition() == nullptr ||
        contig_id_entry.is_non_divisible_split) {
      auto it = ref_2_consumer.find(contig_id);
      if (it != ref_2_consumer.end()) {
        consumer_id = it->second;
      } else {
        continue;
      }
    }

    RootPredicateInfo info;

    // Compute offsets for start and stop predicate. For non-shift,
    // non-gather ops, there's only stop predicate as indices never be
    // negative. However, for shift and gather, the index may need to
    // be predicated so that it is >= zero.
    //
    // Furthermore, in case of gather, both producer and consumer
    // positions may need to be predicated, so there can be multiple
    // offset values.
    //
    // The final predicates will look like:
    // (index + start_offset) >= 0 && (index + stop_offset) < extent.

    std::tie(info.start_offsets_, info.stop_offsets_) = getStartAndStopOffsets(
        consumer_id,
        consumer_tv,
        reference,
        ref_start_index_map,
        ref_stop_index_map,
        shift_padding,
        unswitch_or_vec_loop != nullptr,
        contig_id_entry.is_non_divisible_split);

    auto stop_index = ref_stop_indexing_it->second;
    auto start_index = ref_start_index_map.at(kir_contig_id);

    // Build predicates for start positions as:
    //   start_index + start_offset >= 0
    for (auto start_offset : info.start_offsets_) {
      if (canOmitStartPredicate(start_offset)) {
        info.start_predicates_.push_back(ir_builder.trueVal());
        continue;
      }
      auto offsetted_start_index =
          ir_builder.addExpr(start_index, start_offset);
      auto pred =
          ir_builder.geExpr(offsetted_start_index, ir_builder.zeroVal())
              ->as<kir::Bool>();
      info.start_predicates_.push_back(pred);
    }

    // Build predicates for stop positions as:
    //   stop_index + stop_offset < IterDomain::extent
    for (auto stop_offset : info.stop_offsets_) {
      if (canOmitStopPredicate(stop_index, stop_offset, kir_contig_id)) {
        info.stop_predicates_.push_back(ir_builder.trueVal());
        continue;
      }
      auto offsetted_stop_index = ir_builder.addExpr(stop_index, stop_offset);
      auto pred =
          ir_builder.ltExpr(offsetted_stop_index, kir_contig_id->extent())
              ->as<kir::Bool>();
      info.stop_predicates_.push_back(pred);
    }

    // Transform ids from reference to concrete and consumer domains
    // (based on loop compute at map)
    for (auto ref_id : contig_id_entry.covered_ids) {
      info.root_ids_.insert(reference.id_to_concrete.at(ref_id));
      info.consumer_ids_.insert(ref_2_consumer.at(ref_id));
    }
    pred_info_vec.emplace_back(info);
  }

  return {pred_info_vec, reference};
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
  return loop->isUnrolled() && (!ref_dom_simple || !ind_simple);
}

RootPredicateInfo RootPredicateInfo::getFalseInfo() {
  const auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

  RootPredicateInfo info;
  info.start_predicates_.push_back(ir_builder.falseVal());
  info.stop_predicates_.push_back(ir_builder.falseVal());
  // These are just placeholder. When the predicate is false, the
  // offset should not be used.
  info.start_offsets_.push_back(nullptr);
  info.stop_offsets_.push_back(nullptr);

  return info;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
