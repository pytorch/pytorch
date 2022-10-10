#include <torch/csrc/jit/codegen/cuda/index_compute.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/contiguity.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_double_buffer.h>
#include <torch/csrc/jit/codegen/cuda/lower_index_compute.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_validation.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Offset of an index of a producer axis with respect to its
//! corresponding consumer index
int getProducerHaloOffset(
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

  auto offset = p_pad - c_pad;

  // If the consumer is a result of shifting the producer, adjust the
  // producer index per the offsets argument of the shift op.
  if (auto shift_op = dynamic_cast<const ShiftOp*>(consumer_tv->definition())) {
    offset -= shift_op->offset(producer_axis);
  }

  return offset;
}

//! Offset producer index when necessary
Val* getProducerIndexWithHalo(
    const TensorView* producer_tv,
    size_t producer_axis,
    Val* producer_index,
    const TensorView* consumer_tv) {
  const auto offset =
      getProducerHaloOffset(producer_tv, producer_axis, consumer_tv);

  if (offset == 0) {
    return producer_index;
  }

  producer_index = SimplifyingIrBuilder::addExpr(producer_index, offset);

  return producer_index;
}

//! Create a producer offset based off a consumer index
//!
//! \param consumer_root_axis Position of corresponding consumer axis
//! \param consumer_tv Consumer TensorView
//! \param index_map Mappings from consumer or reference to indices
//! \param use_reference_map True when index_map maps reference domains
//! \param concrete_to_ref_map Mappings from concrete to reference domains
Val* getProducerOffsetWithGather(
    size_t consumer_root_axis,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& index_map,
    bool use_reference_map = false,
    const std::unordered_map<IterDomain*, IterDomain*>& concrete_to_ref_map =
        {}) {
  const auto gpu_lower = GpuLower::current();

  const auto gather_expr = dynamic_cast<GatherOp*>(consumer_tv->definition());

  if (gather_expr == nullptr) {
    return gpu_lower->kernel()->zeroVal();
  }

  // If the window extent is one, no specific offsetting
  // is necessary
  if (consumer_root_axis >= gather_expr->windowShape().size() ||
      gather_expr->windowShape()[consumer_root_axis] == 1) {
    return gpu_lower->kernel()->zeroVal();
  }

  // Basically, the goal is to build an expression of producer_index +
  // window_index, so we first need to locate the index expression
  // that corresponds to the window axis of this producer axis.

  const auto window_axis = gather_expr->gatherAxis(consumer_root_axis);
  auto window_id = consumer_tv->getRootDomain().at(window_axis);

  // When index_map maps a reference tensor, find the corresponding
  // reference ID of window_id.
  if (use_reference_map) {
    auto concrete_window_id = gpu_lower->caMap()->getConcreteMappedID(
        window_id, IdMappingMode::EXACT);
    auto concrete_2_ref_it = concrete_to_ref_map.find(concrete_window_id);
    TORCH_INTERNAL_ASSERT(concrete_2_ref_it != concrete_to_ref_map.end());
    window_id = concrete_2_ref_it->second;
  }

  auto window_idx = index_map.at(window_id);

  // Positive padding at offset zero means the indexing shifted to the
  // negative direction.
  auto pad_width = gather_expr->padWidth()[consumer_root_axis][0];

  // producer offset: window_index - padding
  auto producer_offset = SimplifyingIrBuilder::subExpr(
      window_idx, SimplifyingIrBuilder::create<Int>(pad_width));
  return producer_offset;
}

//! Create a producer offset based off a consumer index
//!
//! \param consumer_root_axis Position of corresponding consumer axis
//! \param consumer_tv Consumer TensorView
//! \param index_map Mappings from consumer or reference to indices
//! \param use_reference_map True when index_map maps reference domains
//! \param concrete_to_ref_map Mappings from concrete to reference domains
Val* getConcreteProducerOffsetWithGather(
    size_t consumer_root_axis,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& index_map,
    bool use_concrete_map = false) {
  const auto gpu_lower = GpuLower::current();

  const auto gather_expr = dynamic_cast<GatherOp*>(consumer_tv->definition());

  if (gather_expr == nullptr) {
    return gpu_lower->kernel()->zeroVal();
  }

  // If the window extent is one, no specific offsetting
  // is necessary
  if (consumer_root_axis >= gather_expr->windowShape().size() ||
      gather_expr->windowShape()[consumer_root_axis] == 1) {
    return gpu_lower->kernel()->zeroVal();
  }

  // Basically, the goal is to build an expression of producer_index +
  // window_index, so we first need to locate the index expression
  // that corresponds to the window axis of this producer axis.

  const auto window_axis = gather_expr->gatherAxis(consumer_root_axis);
  auto window_id = consumer_tv->getRootDomain().at(window_axis);

  Val* window_idx = nullptr;

  if (use_concrete_map) {
    window_idx = index_map.at(ir_utils::caMapExactConcreteId(window_id));
  } else {
    window_idx = index_map.at(window_id);
  }

  // Positive padding at offset zero means the indexing shifted to the
  // negative direction.
  auto pad_width = gather_expr->padWidth()[consumer_root_axis][0];

  // producer offset: window_index - padding
  auto producer_offset = SimplifyingIrBuilder::subExpr(
      window_idx, SimplifyingIrBuilder::create<Int>(pad_width));
  return producer_offset;
}

//! Offset a producer index of a gather expression
//!
//! Given an index of a producer root axis, build a new index
//! expression that accesses a window position that the current loop
//! structure refers to. Use getGatherProducerOffset to create an
//! offset Val.
Val* getProducerIndexWithGather(
    Val* producer_index,
    size_t producer_root_axis,
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& concrete_index_map) {
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

  auto offset = getConcreteProducerOffsetWithGather(
      consumer_axis, consumer_tv, concrete_index_map, true);
  return SimplifyingIrBuilder::addExpr(producer_index, offset);
}

// Adjusts a global consumer index when its root domain is partially
// split. Note that non-global consumer indices don't need any
// adjustment.
Val* getGlobalConsumerOffsetWithPartialSplit(IterDomain* root_id) {
  auto offset = GpuLower::current()->partialSplitMap().getStartOffset(root_id);
  if (offset == nullptr) {
    return GpuLower::current()->kernel()->zeroVal();
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
Val* getProducerIndexWithPartialSplit(
    Val* producer_index,
    IterDomain* producer_root_id,
    const TensorView* producer_tv,
    const TensorView* consumer_tv) {
  const auto gpu_lower = GpuLower::current();

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
  consumer_offset = consumer_offset == nullptr ? gpu_lower->kernel()->zeroVal()
                                               : consumer_offset;

  auto producer_offset =
      gpu_lower->partialSplitMap().getStartOffset(producer_root_id);
  producer_offset = producer_offset == nullptr ? gpu_lower->kernel()->zeroVal()
                                               : producer_offset;

  // If the producer is on global memory, it's always allocated
  // without trimming the out-of-bounds region, so the consumer offset
  // should be added to the index.
  if (producer_tv->getMemoryType() == MemoryType::Global) {
    if (consumer_offset->isZeroInt()) {
      return producer_index;
    } else {
      return SimplifyingIrBuilder::addExpr(producer_index, consumer_offset);
    }
  }

  // Non-global case. Difference of the split offsets must be
  // accounted.

  auto diff = SimplifyingIrBuilder::subExpr(consumer_offset, producer_offset);
  kir::ExpressionEvaluator ee;
  auto diff_eval = ee.evaluate(diff);
  // We currently only allow constant offsetting
  TORCH_INTERNAL_ASSERT(diff_eval.has_value(), "Invalid partial split");

  if (diff_eval.value() == 0) {
    return producer_index;
  }

  return SimplifyingIrBuilder::addExpr(
      producer_index,
      SimplifyingIrBuilder::create<Int>(diff_eval->as<int64_t>()));
}

} // namespace

void IndexCompute::handle(Split* split) {
  auto in_id = maybeGetExactMapConcreteID(split->in()->as<IterDomain>());
  auto outer_id = maybeGetExactMapConcreteID(split->outer()->as<IterDomain>());
  auto inner_id = maybeGetExactMapConcreteID(split->inner()->as<IterDomain>());

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
    index_map_[in_id] = GpuLower::current()->kernel()->zeroVal();
    extent_map_[in_id] = GpuLower::current()->kernel()->zeroVal();
  } else if (zero_merged_in && outer_zero) {
    index_map_[in_id] = inner_ind;
    extent_map_[in_id] = getExtent(inner_id);
  } else if (zero_merged_in && inner_zero) {
    index_map_[in_id] = outer_ind;
    extent_map_[in_id] = getExtent(outer_id);
  } else {
    index_map_[in_id] = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(outer_ind, getExtent(inner_id)),
        inner_ind);
    // The extent should be updated only when its allocation is
    // partial, i.e., zero_merged_in is true. See PR #1270.
    if (zero_merged_in) {
      extent_map_[in_id] = SimplifyingIrBuilder::mulExpr(
          getExtent(outer_id), getExtent(inner_id));
    }
  }
}

void IndexCompute::handle(Merge* merge) {
  auto out_id = maybeGetExactMapConcreteID(merge->out());
  auto outer_id = maybeGetExactMapConcreteID(merge->outer());
  auto inner_id = maybeGetExactMapConcreteID(merge->inner());

  auto out_it = index_map_.find(out_id);
  if (out_it == index_map_.end()) {
    return;
  }
  auto out_ind = out_it->second;

  auto zero = GpuLower::current()->kernel()->zeroVal();

  if (isZero(out_id)) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = zero;
    // TODO: Why do we set extent_map_ to zero? This has to be protected by zero
    // merged in, but seems logical to me the extent would still be one.
    extent_map_[outer_id] = zero;
    extent_map_[inner_id] = zero;
    zero_domains_.emplace(outer_id);
    zero_domains_.emplace(inner_id);
    return;
  }

  if (!hasZeroMerged(out_id) && contig_ids_.find(out_id) != contig_ids_.end()) {
    // Contiguous indexing path
    auto input_ids = ir_utils::iterDomainInputsOfOrderedAs(
        {merge->out()}, td_->getMaybeRFactorDomain());

    // Shouldn't hit this, but don't want to segfault if somehow we do.
    TORCH_INTERNAL_ASSERT(!input_ids.empty());

    // Try to find the last non broadcast entry to put the index in if it's a
    // contiguous merge. This isn't strictly necessary but there's implicit
    // assumptions in the indexing logic that assume broadcasted root domains
    // can be ignored. This logic is just to try and match that logic.
    // Initialize everything to zero.
    for (auto root_id : input_ids) {
      index_map_[root_id] = zero;
    }

    // If all are broadcast we can just send the index to the last entry.
    if (std::all_of(input_ids.begin(), input_ids.end(), [](IterDomain* id) {
          // I don't think reductions can be in here, but strictly matching the
          // logic in the indexing functions like
          // getNonGlobalConsumerStridedIndices
          return id->isBroadcast() || id->isReduction() || id->isStride();
        })) {
      index_map_[*(input_ids.end() - 1)] = out_ind;
    } else {
      for (auto id_it = input_ids.rbegin(); id_it != input_ids.rend();
           id_it++) {
        auto id = *id_it;
        if (id->isBroadcast() || id->isReduction() || id->isStride()) {
          continue;
        } else {
          index_map_[id] = out_ind;
          break;
        }
      }
    }

    return;
  }

  Val* inner_extent = getExtent(inner_id);

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
    if (hasZeroMerged(out_id)) {
      zero_merged_in_.insert(outer_id);
    }
  } else if (outer_id->isBroadcast() && outer_extent->isOneInt()) {
    // Propagate away from broadcast dims
    index_map_[outer_id] = zero;
    index_map_[inner_id] = out_ind;

    extent_map_[inner_id] = getExtent(out_id);
    if (hasZeroMerged(out_id)) {
      zero_merged_in_.insert(inner_id);
    }
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
    index_map_[outer_id] = SimplifyingIrBuilder::divExpr(out_ind, inner_extent);
    index_map_[inner_id] = SimplifyingIrBuilder::modExpr(out_ind, inner_extent);
  }
}

void IndexCompute::handle(Swizzle2D* swizzle_2d) {
  auto out_x_id = maybeGetExactMapConcreteID(swizzle_2d->outX());
  auto out_y_id = maybeGetExactMapConcreteID(swizzle_2d->outY());
  auto in_x_id = maybeGetExactMapConcreteID(swizzle_2d->inX());
  auto in_y_id = maybeGetExactMapConcreteID(swizzle_2d->inY());

  auto out_x_it = index_map_.find(out_x_id);
  auto out_y_it = index_map_.find(out_y_id);

  if (out_x_it == index_map_.end() || out_y_it == index_map_.end()) {
    return;
  }

  const auto out_x_ind = out_x_it->second;
  const auto out_y_ind = out_y_it->second;

  if (swizzle_mode_ == SwizzleMode::NoSwizzle ||
      swizzle_mode_ != swizzle_2d->swizzleMode()) {
    // Handle inactive swizzles by just passing through index
    //  and extend information.

    TORCH_INTERNAL_ASSERT(
        index_map_.count(in_x_id) == index_map_.count(in_y_id),
        "input index should be either both defined or both undefined");
    if (index_map_.count(in_x_id)) {
      // Only propagate original index through if
      //  the input index hasn't been computed.
      // TODO:
      //  This part should be cleaner once we remove the
      // second index traversal pass.
      return;
    }
    index_map_[in_x_id] = out_x_ind;
    index_map_[in_y_id] = out_y_ind;
    extent_map_[in_y_id] = getExtent(out_y_id);
    extent_map_[in_x_id] = getExtent(out_x_id);
  } else {
    // Generate integer swizzle math if the
    //  swizzle is activated. See also
    //  [Note on swizzle mode].

    auto out_pair = IrBuilder::swizzle2DIntExpr(
        out_x_ind,
        out_y_ind,
        getExtent(out_x_id),
        getExtent(out_y_id),
        swizzle_2d->swizzleType());

    index_map_[in_x_id] =
        IrBuilder::pairSelectExpr(out_pair, kir::PairSelect::Selection::X);
    index_map_[in_y_id] =
        IrBuilder::pairSelectExpr(out_pair, kir::PairSelect::Selection::Y);
  }
}

void IndexCompute::handle(Expr* e) {
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
    case (ExprType::Swizzle2D):
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid expr type found in transform traversal.");
  }
  BackwardVisitor::handle(e);
}

IndexCompute::IndexCompute(
    const TensorDomain* _td,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_map<IterDomain*, Val*> reference_halo_extent_map)
    : IndexCompute(
          _td,
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_domains),
          std::move(zero_merged_in),
          ContigIDs(
              _td->domain(),
              _td->getMaybeRFactorDomain(),
              std::vector<bool>(_td->getMaybeRFactorDomain().size(), false),
              {}),
          std::move(preferred_paths),
          std::move(reference_halo_extent_map)) {}

IndexCompute::IndexCompute(
    const TensorDomain* _td,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in,
    const ContigIDs& contig_finder,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_map<IterDomain*, Val*> reference_halo_extent_map)
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
  contig_ids_ = contig_finder.contigIDs();
  root_to_indexed_id_ = contig_finder.rootToIndexedID();
  const auto& within_contig = contig_finder.withinContigIDs();
  for (auto contig_id : contig_ids_) {
    if (index_map_.find(contig_id) != index_map_.end()) {
      TORCH_INTERNAL_ASSERT(
          within_contig.find(contig_id) != within_contig.end());
      for (auto id : within_contig.at(contig_id)) {
        index_map_.erase(id);
      }
    }
  }
}

IndexCompute::IndexCompute(
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> preferred_paths,
    std::unordered_map<IterDomain*, Val*> reference_halo_extent_map)
    : index_map_(std::move(initial_index_map)),
      zero_domains_(std::move(zero_domains)),
      preferred_paths_(std::move(preferred_paths)),
      reference_halo_extent_map_(std::move(reference_halo_extent_map)) {
  FUSER_PERF_SCOPE("GpuLower::Lower::IndexCompute::IndexCompute");
  concrete_id_pass_ = true;
  swizzle_mode_ = SwizzleMode::Loop;
}

void IndexCompute::run(const LoopIndexing& loop_indexing) {
  TORCH_INTERNAL_ASSERT(
      concrete_id_pass_, "concrete pass only for this option");
  // Apply loop swizzles if there are any that outputs to
  //  the loop domains.
  // Currently only support loop swizzles that directly output
  //  to concrete loop domains and these are validated in
  //  validate swizzle pass.
  // TODO:
  //  will gradually enable replaying and mapping of loop
  // swizzles in the IR infrastructure and once that's piped
  // through this part of logic will be removed.
  std::unordered_set<Expr*> visited;
  for (auto loop_id : loop_indexing.loopDomains()) {
    auto loop_id_def = loop_id->definition();
    if (loop_id_def != nullptr && loop_id_def->isA<Swizzle2D>()) {
      if (visited.insert(loop_id_def).second) {
        handle(loop_id_def);
      }
    }
  }

  // Resolve the index vals that could be resolved with only
  //  the loops that consumer_tv doesn't share with any of its
  //  consumers, i.e. the not-inlined loops that define consumer_tv
  //  values.
  collectIndexIntoPermissiveMap(loop_indexing);

  // Run through the loop indexing expressions and generate
  //  the indexing integer math for the concrete ids.
  for (auto expr : loop_indexing.getBackwardExprList()) {
    // Resolve missing values from permissive map.
    updateIndexMapFromPermissiveMap(expr);

    handle(expr);
  }
}

void IndexCompute::collectIndexIntoPermissiveMap(
    const LoopIndexing& loop_indexing) {
  // Visit the expressions that only produces un-inlined iterdomains,
  //  in reverse topological order.
  for (auto expr : loop_indexing.getBackwardOutOfLineExprList()) {
    // Compute indexing vals for the expression inputs.
    //
    // This stage should run before any indexing computation so it could be
    //  made sure that all index values computed at this stage are
    //  the ones that can be resolved only with the not-inlined
    //  iterdomains.
    //
    auto id_outputs = ir_utils::filterByType<IterDomain>(expr->outputs());
    if (std::all_of(
            id_outputs.begin(), id_outputs.end(), [this](IterDomain* id) {
              return index_map_.count(ir_utils::caMapExactConcreteId(id));
            })) {
      // Visit this expression:
      // LoopIndexingAnalysis::traverseFromDomainVals made sure that each
      //  concrete index is bound exactly once so computing these expressions
      //  early should still be consistent.
      handle(expr);

      auto id_inputs = ir_utils::filterByType<IterDomain>(expr->inputs());
      for (auto id : id_inputs) {
        // Collect backward pass results from this expression if they are
        //  made available in by this expression.
        auto idx_it = index_map_.find(ir_utils::caMapExactConcreteId(id));

        if (idx_it != index_map_.end()) {
          permissive_index_map_
              [GpuLower::current()->caMap()->getConcreteMappedID(
                  id, IdMappingMode::PERMISSIVE)] = idx_it->second;
        }
      }
    }
  }
}

void IndexCompute::updateIndexMapFromPermissiveMap(const Expr* id_expr) {
  auto id_outputs = ir_utils::filterByType<IterDomain>(id_expr->outputs());
  for (auto id : id_outputs) {
    auto concrete_id = ir_utils::caMapExactConcreteId(id);
    // Only try to copy index val from permissive map when
    //  the index is missing.
    if (!index_map_.count(concrete_id)) {
      auto permissive_id = GpuLower::current()->caMap()->getConcreteMappedID(
          id, IdMappingMode::PERMISSIVE);
      // Write the permissive index val into index_map_ if the
      //  missing value is found here.
      auto permissive_it = permissive_index_map_.find(permissive_id);
      if (permissive_it != permissive_index_map_.end()) {
        index_map_[concrete_id] = permissive_it->second;
      }
    }
  }
}

void IndexCompute::run() {
  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  traverseFrom(td_->fusion(), domain_vals, false);
}

IterDomain* IndexCompute::maybeGetExactMapConcreteID(IterDomain* id) {
  if (concrete_id_pass_) {
    return GpuLower::current()->caMap()->getConcreteMappedID(
        id, IdMappingMode::EXACT);
  }
  return id;
}

Val* IndexCompute::getExtent(IterDomain* id) const {
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

bool IndexCompute::hasZeroMerged(IterDomain* id) const {
  return zero_merged_in_.find(id) != zero_merged_in_.end() || isZero(id);
}

bool IndexCompute::isZero(IterDomain* id) const {
  return zero_domains_.find(id) != zero_domains_.end();
}

IndexCompute IndexCompute::updateIndexCompute(
    const TensorDomain* new_td,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    const ContigIDs& contig_finder,
    const std::unordered_map<IterDomain*, Val*>& reference_halo_extent_map)
    const {
  FUSER_PERF_SCOPE("GpuLower::Lower::updateIndexCompute");

  std::unordered_map<IterDomain*, Val*> updated_index_map;
  std::unordered_map<IterDomain*, Val*> updated_extent_map;
  std::unordered_set<IterDomain*> updated_zero_domains;
  std::unordered_set<IterDomain*> updated_zero_merged_in;

  for (auto id_entry : id_map) {
    IterDomain* prev_id = id_entry.first;
    IterDomain* new_id = id_entry.second;

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
      contig_finder,
      {},
      reference_halo_extent_map);

  if (concrete_id_pass_) {
    // This should be the same behavior as with a reference tensor
    //   created, since originally halo was pulled through exact
    //   ca mapping and in the concrete_id_pass case, the id_map
    //   also represents exact ca mapping.
    // TODO: might need to re-visit pathological cases when we may
    //  need to traverse and propagate halo info again in here.
    for (auto id_entry : id_map) {
      IterDomain* prev_id = id_entry.first;
      IterDomain* new_id = id_entry.second;
      auto halo_extent_it = reference_halo_extent_map_.find(prev_id);
      if (halo_extent_it != reference_halo_extent_map_.end()) {
        updated_index_compute.reference_halo_extent_map_[new_id] =
            halo_extent_it->second;
      }
    }
  }

  updated_index_compute.run();

  return updated_index_compute;
}

namespace {
// Map indices down to the leaf domains for applying swizzle
class UpdateLeafIndices : public IterVisitor {
 public:
  UpdateLeafIndices(
      const TensorDomain* td,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> extent_map)
      : td_(td),
        index_map_(std::move(initial_index_map)),
        extent_map_(std::move(extent_map)) {
    const std::vector<Val*> domain_vals(
        td_->domain().begin(), td_->domain().end());

    traverseFrom(td_->fusion(), domain_vals, false);
  }

  const std::unordered_map<IterDomain*, Val*>& indexMap() const {
    return index_map_;
  }

  const std::unordered_map<IterDomain*, Val*>& extentMap() const {
    return extent_map_;
  }

 private:
  using IterVisitor::handle;

  void handle(Split* split) override {
    auto in_id = split->in();
    auto outer_id = split->outer();
    auto inner_id = split->inner();

    // Nothing need to be done when mappings for the output axes
    // already exist.
    if (index_map_.find(outer_id) != index_map_.end()) {
      TORCH_INTERNAL_ASSERT(
          index_map_.find(inner_id) != index_map_.end(),
          "Outer exists but inner not found");
      return;
    }

    if (!index_map_.count(in_id)) {
      // Reduction axes on producer side could be visited on forward
      //  propagation pass and current implementation does not yet
      //  support reduciton on swizzled iterdomains, so un-indexed
      //  reduction iterdomains are just ignored for now.
      TORCH_INTERNAL_ASSERT(
          in_id->isReduction(), "Undefined index for ", in_id->toString());
      return;
    }

    auto factor = split->factor();
    index_map_[inner_id] =
        SimplifyingIrBuilder::modExpr(index_map_[in_id], factor);
    extent_map_[inner_id] = factor;
    index_map_[outer_id] =
        SimplifyingIrBuilder::divExpr(index_map_[in_id], factor);
    extent_map_[outer_id] =
        SimplifyingIrBuilder::ceilDivExpr(getExtent(in_id), factor);
  }

  void handle(Merge* merge) override {
    auto out_id = merge->out();
    auto outer_id = merge->outer();
    auto inner_id = merge->inner();

    if (!index_map_.count(outer_id) || !index_map_.count(inner_id)) {
      // Reduction axes on producer side could be visited on forward
      //  propagation pass and current implementation does not yet
      //  support reduciton on swizzled iterdomains, so un-indexed
      //  reduction iterdomains are just ignored for now.
      TORCH_INTERNAL_ASSERT(
          outer_id->isReduction() && inner_id->isReduction(),
          "Undefined index for ",
          outer_id->toString(),
          " and ",
          inner_id->toString());
      return;
    }

    // Nothing need to be done when mappings for the output axes
    // already exist.
    if (index_map_.find(out_id) != index_map_.end()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        index_map_.find(outer_id) != index_map_.end(), "Outer ID not found");
    TORCH_INTERNAL_ASSERT(
        index_map_.find(inner_id) != index_map_.end(), "Inner ID not found");

    index_map_[out_id] = SimplifyingIrBuilder::addExpr(
        index_map_[inner_id],
        SimplifyingIrBuilder::mulExpr(
            index_map_[outer_id], getExtent(inner_id)));

    extent_map_[out_id] =
        SimplifyingIrBuilder::mulExpr(getExtent(outer_id), getExtent(inner_id));
  }

  void handle(Swizzle2D* swizzle_2d) override {
    auto in_x = swizzle_2d->inX();
    auto in_y = swizzle_2d->inY();
    auto out_x = swizzle_2d->outX();
    auto out_y = swizzle_2d->outY();

    // Forward propagation pass still just forward
    //  through the indices and the actual swizzle
    //  will be applied on the backward pass in
    //  IndexSwizzle class implementation.
    index_map_[out_x] = index_map_.at(in_x);
    extent_map_[out_x] = getExtent(in_x);
    index_map_[out_y] = index_map_.at(in_y);
    extent_map_[out_y] = getExtent(in_y);
  }

  // return extent_map_[id] if exists, else return id->extent()
  Val* getExtent(IterDomain* id) {
    if (extent_map_.find(id) != extent_map_.end()) {
      return extent_map_.at(id);
    } else {
      return id->extent();
    }
  }

 private:
  const TensorDomain* td_;
  std::unordered_map<IterDomain*, Val*> index_map_;
  std::unordered_map<IterDomain*, Val*> extent_map_;
};

// Returns halo-extended extent if id has halo. Otherwise, just
// returns id->extent.
Val* getHaloExtentOfRootAxis(IterDomain* id, Val* normal_extent = nullptr) {
  if (normal_extent == nullptr) {
    normal_extent = id->extent();
  }

  const auto& halo = GpuLower::current()->haloInfo().getRootAxisInfo(id);
  if (halo.hasHalo()) {
    auto halo_extent = SimplifyingIrBuilder::addExpr(
        normal_extent, SimplifyingIrBuilder::create<Int>(halo.width()));
    return halo_extent;
  } else {
    return normal_extent;
  }
}

} // namespace

IndexSwizzle::IndexSwizzle(
    const TensorView* tv,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in)
    : IndexCompute(
          tv->domain(),
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_domains),
          std::move(zero_merged_in)),
      tv_(tv),
      swizzle_type_(tv->swizzleType()),
      ids_to_swizzle_(tv->axesToSwizzle()) {}

IndexSwizzle::IndexSwizzle(
    const TensorView* tv,
    const TensorDomain* domain,
    std::unordered_map<IterDomain*, Val*> initial_index_map,
    std::unordered_map<IterDomain*, Val*> extent_map,
    std::unordered_set<IterDomain*> zero_domains,
    std::unordered_set<IterDomain*> zero_merged_in)
    : IndexCompute(
          domain,
          std::move(initial_index_map),
          std::move(extent_map),
          std::move(zero_domains),
          std::move(zero_merged_in)),
      tv_(tv),
      swizzle_type_(tv->swizzleType()),
      ids_to_swizzle_(tv->axesToSwizzle()) {}

void IndexSwizzle::run() {
  TORCH_INTERNAL_ASSERT(
      swizzle_type_ == SwizzleType::NoSwizzle ||
          swizzle_type_ == SwizzleType::Transpose,
      "Invalid swizzle type");
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

    if (indexMap().find(id_to_swizzle_i) != indexMap().end() &&
        indexMap().find(id_to_swizzle_j) != indexMap().end()) {
      auto idx_to_swizzle_i = indexMap().at(id_to_swizzle_i);
      auto idx_to_swizzle_j = indexMap().at(id_to_swizzle_j);

      auto swizzled_idx = SimplifyingIrBuilder::modExpr(
          SimplifyingIrBuilder::addExpr(idx_to_swizzle_i, idx_to_swizzle_j),
          id_to_swizzle_j->extent());
      index_map_[id_to_swizzle_j] = swizzled_idx;
      swizzled_ids_.insert(id_to_swizzle_j);
      IndexCompute::run();
    }
  } else if (tv_->hasSwizzleOp()) {
    // Propagate backward for the annotated swizzle path.
    // TODO:
    //  eventually will unify the two swizzling implementation
    //  code path in a follow up. Currently just focusing on
    //  getting the necessary implementation of the swizzle
    //  operator ready.
    //
    // At this intermediate state, the legacy swizzle implementation
    //  takes precedence, i.e. whenever swizzle_type_ is not NoSwizzle,
    //  the new swizzle op pass is disabled.
    UpdateLeafIndices update_leaves(td_, indexMap(), extentMap());
    index_map_ = update_leaves.indexMap();
    extent_map_ = update_leaves.extentMap();
    IndexCompute::swizzle_mode_ = SwizzleMode::Data;
    IndexCompute::run();
  }
}

void IndexSwizzle::handle(Expr* e) {
  auto out_ids = ir_utils::filterByType<IterDomain>(e->outputs());
  bool needs_update =
      std::any_of(
          out_ids.begin(),
          out_ids.end(),
          [this](IterDomain* id) {
            return swizzled_ids_.find(id) != swizzled_ids_.end();
          }) ||
      (e->isA<Swizzle2D>() &&
       e->as<Swizzle2D>()->swizzleType() != Swizzle2DType::NoSwizzle &&
       e->as<Swizzle2D>()->swizzleMode() == SwizzleMode::Data);
  if (!needs_update) {
    return;
  }

  IndexCompute::handle(e);
  for (auto input : ir_utils::filterByType<IterDomain>(e->inputs())) {
    swizzled_ids_.insert(input);
  }
}

void IndexSwizzle::handle(Swizzle2D* swizzle_2d) {
  auto out_x_id = swizzle_2d->outX();
  auto out_y_id = swizzle_2d->outY();

  auto out_x_it = index_map_.find(out_x_id);
  auto out_y_it = index_map_.find(out_y_id);

  // TODO: unify the legacy path in all usage
  TORCH_INTERNAL_ASSERT(
      swizzle_type_ == SwizzleType::NoSwizzle,
      "Cannot mix usage of two swizzle implementations");

  TORCH_INTERNAL_ASSERT(
      out_x_it != index_map_.end() && out_y_it != index_map_.end(),
      "Swizzle output indices were not propagated through");

  IndexCompute::handle(swizzle_2d);
}

// Used for local and shared index mapping. Returns a map from loops
// to loop indices as well as a set of loops that do not contribute to
// indexing.
std::pair<
    std::unordered_map<kir::ForLoop*, Val*>,
    std::unordered_set<kir::ForLoop*>>
indexMapFromTV(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    kir::ForLoop* alloc_loop,
    bool as_consumer,
    kir::ForLoop* double_buffer_loop) {
  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  const bool is_global = tv->getMemoryType() == MemoryType::Global;
  const bool is_shared = tv->getMemoryType() == MemoryType::Shared;
  const bool is_local = tv->getMemoryType() == MemoryType::Local;

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;

  // Check if the current op has an implicit loop implemented
  //  within an mma instruction.
  bool within_mma_loops =
      std::any_of(loops.begin(), loops.end(), [](kir::ForLoop* fl) {
        return fl->iter_domain()->isMma();
      });

  // When indexed as a producer, the parallel types of the the
  // producer domains may not be the same as those of the loops, but
  // that's still valid parallelization. However, in that case, using
  // the parallel types of the loops to decide replacement of indices
  // with zero isn't valid. That's only valid when there's a matching
  // IterDomain in the producer tensor that has the same parallel
  // type.
  auto find_matching_parallel_domain = [tv](IterDomain* id) -> bool {
    const auto gpu_lower = GpuLower::current();
    auto it = std::find_if(
        tv->domain()->domain().begin(),
        tv->domain()->domain().end(),
        [&](IterDomain* tv_id) {
          // Matching is done using the index and loop maps. See
          // validateParallelize as well.
          return gpu_lower->caMap()->areMapped(
                     id, tv_id, IdMappingMode::EXACT) ||
              (GpuLower::current()->caMap()->areMapped(
                   id, tv_id, IdMappingMode::PERMISSIVE) &&
               ir_utils::derivedFromRootCAAxes(tv, tv_id));
        });
    if (it == tv->domain()->domain().end()) {
      return false;
    }

    auto corresponding_domain = *it;
    return corresponding_domain->getParallelType() == id->getParallelType();
  };

  // Track domains that do not contibute to the resulting
  // index. Previously, index->isZeroInt() was used to detect such
  // domains, but that's not a reliable method as we may set an
  // initial index to zero for unswitch.
  std::unordered_set<kir::ForLoop*> zero_loops;

  for (auto loop : loops) {
    Val* idx = nullptr;
    const auto same_parallel_type = as_consumer ||
        find_matching_parallel_domain(loop->iter_domain()) ||
        // Note && TODO:
        //  mma swizzled lane_id does not map naturally from producer
        //   to consumer but they should still be detected as same
        //   parallel type. In a follow up may want to extent
        //   find_matching_parallel_domain to cover this case.
        (within_mma_loops &&
         loop->iter_domain()->getParallelType() == ParallelType::TIDx);
    // See also LoopNestGenerator::pushAlloc.
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (!within_alloc) {
      if ((loop->iter_domain()->isThreadDim() && is_shared) ||
          (loop->iter_domain()->isThread() && is_global)) {
        idx = loop->index();
      } else {
        idx = GpuLower::current()->kernel()->zeroVal();
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
        // MMA operands are currently indexed in units of "fragments",
        //  so each mma tensor domain would be zero-ed and the tensor index
        //  calculated here would be the fragment index.
        // TODO: This is a quick WAR to enable iterating over a register array
        //  of MMA fragments, so we could generate unrolled mma loops.
        //  Eventually we still want IdGraph to be able to analyze the
        //  in-register layout of mma fragments for more unified indexing math
        //  as well as more flexibility in swizzling loops.
        (loop->iter_domain()->isMma() && !as_consumer)) {
      idx = GpuLower::current()->kernel()->zeroVal();
      zero_loops.insert(loop);
    } else {
      idx = loop->index();
    }

    // If the loop is trivial, the loop index can only be the loop
    // start value.
    if (idx == loop->index() && loop->isTrivial()) {
      idx = loop->start();
    }

    if (loop == double_buffer_loop) {
      auto stage_depth =
          GpuLower::current()->doubleBufferInfo().getStageDepthFor(
              loop->iter_domain());
      idx = SimplifyingIrBuilder::addExpr(
          idx, SimplifyingIrBuilder::create<Int>(stage_depth - 1));
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
    const std::unordered_map<IterDomain*, IterDomain*>& id_map) {
  if (tv->getMemoryType() != MemoryType::Local) {
    return;
  }

  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  for (auto loop : loops) {
    if (!within_alloc) {
      if (loop == alloc_loop) {
        within_alloc = true;
      }
      continue;
    }
    IterDomain* loop_id = loop->iter_domain();
    if (loop->vectorize() || loop_id->isThread()) {
      continue;
    }
    // Look for a domain that is mapped with the loop. If mapped in
    // the loop map, the loop index should be used for indexing of the
    // tensor, except for broadcast and reduction domains.
    auto it = std::find_if(
        tv->domain()->domain().begin(),
        tv->domain()->domain().end(),
        [loop_id, &id_map](IterDomain* id) {
          if (id->isBroadcast() || id->isReduction() || id->isStride()) {
            return false;
          }
          auto id_replacement = id_map.find(id);
          if (id_replacement != id_map.end()) {
            id = id_replacement->second;
          }
          return GpuLower::current()->caMap()->areMapped(
              loop_id, id, IdMappingMode::PERMISSIVE);
        });
    if (it != tv->domain()->domain().end()) {
      loop->requireUnroll();
    }
  }
}

namespace {

//! Returns an iterdomain that corresponds to the
//!  indexing sub-expression to hoist or a nullopt
//!  if the index should not be hoisted.
c10::optional<IterDomain*> getMaybeIndexedIdToHoist(
    IterDomain* root_id,
    const TensorView* tv,
    const IndexCompute& indexing,
    Val* index) {
  if (isOptionDisabled(DisableOption::IndexHoist) ||
      index->definition() == nullptr) {
    return c10::nullopt;
  }

  // The old swizzle interface, which should be deprecated, is not
  // supported.
  if (tv->swizzleType() != SwizzleType::NoSwizzle) {
    return c10::nullopt;
  }

  // New swizzle interface not yet supported
  if (tv->hasSwizzleOp()) {
    return c10::nullopt;
  }

  // Find the true indexed domain, which can be a merged contiguous domain.
  auto contig_id_it = indexing.rootToContigID().find(root_id);
  TORCH_INTERNAL_ASSERT(
      contig_id_it != indexing.rootToContigID().end(),
      "Consumer indexed ID not found: ",
      root_id->toString());
  auto indexed_id = contig_id_it->second;
  // Make sure this contig ID is indeed indexed
  TORCH_INTERNAL_ASSERT(
      indexing.indexMap().find(contig_id_it->second) !=
          indexing.indexMap().end(),
      "Invalid contig index: ",
      contig_id_it->second->toString());

  return indexed_id;
}

// Version of hoisting without using reference tensor,
//  should eventually deprecate the other one once reference
//  tensor is completely deprecated.
Val* hoistConsumerIndex(
    IterDomain* consumer_root_id,
    const TensorView* consumer_tv,
    const IndexCompute& consumer_indexing,
    std::vector<IterDomain*> loop_domains,
    const std::unordered_map<IterDomain*, Val*> initial_loop_index_map,
    const std::vector<kir::ForLoop*>& loops,
    Val* index) {
  auto maybe_hoisted_consumer_id = getMaybeIndexedIdToHoist(
      consumer_root_id, consumer_tv, consumer_indexing, index);

  if (!maybe_hoisted_consumer_id.has_value()) {
    return index;
  }

  // Insert the index into the common index map. A previously inserted
  // val can be returned.
  auto common_index = GpuLower::current()
                          ->commonIndexMap()
                          .insert(
                              maybe_hoisted_consumer_id.value(),
                              consumer_tv->domain(),
                              loop_domains,
                              initial_loop_index_map,
                              loops,
                              index)
                          .first;

  return common_index;
}

std::unordered_map<IterDomain*, IterDomain*> invertOneToOneMap(
    const std::unordered_map<IterDomain*, IterDomain*>& map) {
  std::unordered_map<IterDomain*, IterDomain*> inverted;
  for (const auto& kv : map) {
    bool inserted = inverted.emplace(kv.second, kv.first).second;
    TORCH_INTERNAL_ASSERT(
        inserted,
        "Multiple mappings to the same value detected: ",
        kv.second->toString());
  }
  return inverted;
}

Val* hoistProducerIndex(
    IterDomain* producer_root_id,
    const TensorView* producer_tv,
    const IndexCompute& producer_indexing,
    const TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_map,
    std::vector<IterDomain*> loop_domains,
    const std::unordered_map<IterDomain*, Val*> initial_loop_index_map,
    const std::vector<kir::ForLoop*>& loops,
    Val* index) {
  auto maybe_indexed_producer_id = getMaybeIndexedIdToHoist(
      producer_root_id, producer_tv, producer_indexing, index);

  if (!maybe_indexed_producer_id.has_value()) {
    return index;
  }

  // Use the corresponding consumer domain to find matching
  // for-loops. Note that there's no CA mapping with the producer
  // domains as the producer TensorDomain is a temporary replay
  // domain.
  auto indexed_consumer_id_it = p2c_map.find(maybe_indexed_producer_id.value());

  // There can be no corresponding consumer ID. For example, consider:
  //   consumer: [b1, i2, i3]
  //   producer: [i2, i3].
  // Suppose the consumer is transformed as:
  //   consumer: [(b1*i2)*i3]
  // Then the producer would be transformed when indexed:
  //   producer: [i2*i3]
  // Assuming i2 and i3 are contiguous, the producer indexing is done
  // with the mreged i2*i3 domain, but there's no domain in the
  // cosumer that maps with the producer indexed domain.
  // It seems non-trivial to support patterns like this. Skip for now.
  if (indexed_consumer_id_it == p2c_map.end()) {
    return index;
  }

  IterDomain* indexed_consumer_id = indexed_consumer_id_it->second;

  auto common_index = GpuLower::current()
                          ->commonIndexMap()
                          .insert(
                              indexed_consumer_id,
                              consumer_tv->domain(),
                              loop_domains,
                              initial_loop_index_map,
                              loops,
                              index)
                          .first;

  return common_index;
}

} // namespace

std::vector<Val*> Index::getGlobalProducerStridedIndices(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getGlobalProducerIndex");
  const auto gpu_lower = GpuLower::current();

  // Replay producer to look like consumer so we can index on producer since
  // our loop nests look like consumer
  auto pairwise_map = PairwiseRootDomainMap(producer_tv, consumer_tv);
  auto producerAsC =
      TransformReplay::replayPasC(producer_tv, consumer_tv, -1, pairwise_map)
          .first;

  // Make the producer_tv look like consumer while performing indexing math
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // Map sent to best effort replay needs to match the exact incantation for
  // compute_at_mode.cpp with MappingMode::Index
  auto c2p_root_map =
      PairwiseRootDomainMap(producer_tv, consumer_tv, true)
          .mapConsumerToProducer(consumer_tv->domain(), producer_tv->domain());

  // This replay has to be consistent with compute at index map.
  BestEffortReplay replay_producer_as_consumer(
      producer_tv->domain()->domain(),
      consumer_tv->domain()->domain(),
      c2p_root_map);

  const auto& c2p_map = replay_producer_as_consumer.getReplay();
  const auto p2c_map = invertOneToOneMap(c2p_map);

  // Forward vectorized IDs to index into producer correctly
  // We want p_id to be vectorized like consumer just for the indexing, then we
  // need to switch it back later. Store previous state here when changing. We
  // need to do this as replaying producer as consumer can use replay best
  // effort which means some domains may be producer's original domains.
  std::vector<std::pair<IterDomain*, ParallelType>> p_id_backup;
  for (auto entry : c2p_map) {
    auto ref_id = ir_utils::caMapExactConcreteId(entry.first);
    auto p_id = entry.second;
    if (ref_id->getParallelType() == ParallelType::Vectorize) {
      p_id_backup.emplace_back(std::make_pair(p_id, p_id->getParallelType()));
      p_id->parallelize(ParallelType::Vectorize);
    } else if (ref_id->getParallelType() == ParallelType::MisalignedVectorize) {
      p_id->parallelize(ParallelType::MisalignedVectorize);
    }
  }

  auto producer_indexing_from_idgraph =
      getTensorIndexFromIdGraph(loops, consumer_tv, producer_tv, true, c2p_map);

  auto producer_indexing = producer_indexing_from_idgraph.index;

  // Revert p_ids
  for (auto entry : p_id_backup) {
    entry.first->parallelize(entry.second);
  }

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();

  // TODO: Abstract stride logic to reuse with consumer indexing
  std::vector<Val*> strides(root_dom.size(), nullptr);
  {
    int stride_i = 0;
    for (const auto i : c10::irange(root_dom.size())) {
      if (root_dom[i]->isReduction()) {
        strides[i] = GpuLower::current()->kernel()->oneVal();
        continue;
      }
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << stride_i++ << "]";
      strides[i] =
          SimplifyingIrBuilder::create<NamedScalar>(ss.str(), DataType::Int);
    }
  }

  TORCH_INTERNAL_ASSERT(
      root_dom.size() == producer_tv->domain()->contiguity().size());
  Val* cur_contig_stride = GpuLower::current()->kernel()->oneVal();
  for (const auto i : c10::irange(root_dom.size())) {
    auto dim = root_dom.size() - i - 1;
    if (root_dom[dim]->isReduction()) {
      continue;
    }

    Val* root_ind = nullptr;
    if (producer_indexing.indexMap().find(root_dom[dim]) !=
        producer_indexing.indexMap().end()) {
      root_ind = producer_indexing.indexMap().at(root_dom[dim]);
    } else if (root_dom[dim]->isBroadcast()) {
      root_ind = GpuLower::current()->kernel()->zeroVal();
    }

    TORCH_INTERNAL_ASSERT(
        root_ind != nullptr,
        "Couldn't find root mapping for ",
        producer_tv->toString(),
        " dim: ",
        dim,
        " id: ",
        root_dom[dim]->toString());

    if (producer_tv->domain()->contiguity()[dim]) {
      // If contig, used the stored stride which may be the previous
      // dimensions stride * previous dimensions size
      strides[dim] = cur_contig_stride;
      // Prepare for the next dimension which may also be contiguous, multiply
      // by extent of this dimension
      auto root_dim_extent = getHaloExtentOfRootAxis(root_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(cur_contig_stride, root_dim_extent);
    } else {
      // If non contiguous dimension, keep local stride information, set cur
      // stride to local stride * local raw extent
      auto root_dim_extent = getHaloExtentOfRootAxis(root_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(strides[dim], root_dim_extent);
    }
  }

  auto vectorize_shift =
      loops.empty() ? nullptr : loops.back()->vectorize_shift();

  // Global striding
  std::vector<Val*> strided_inds(
      root_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    // If the domain is derived from a trivial reduction, no indexing
    // to create.
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast() ||
        gpu_lower->trivialReductionInfo().isDerived(root_dom[i])) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        producer_indexing.indexMap().find(root_dom[i]) !=
            producer_indexing.indexMap().end(),
        "Couldn't find root mapping for TV",
        producer_tv->name(),
        " dim: ",
        i,
        " id: ",
        root_dom[i]->toString());

    auto root_ind = producer_indexing.indexMap().at(root_dom[i]);

    // index hoist must be done before the adjustments for halo
    root_ind = hoistProducerIndex(
        root_dom[i],
        producer_tv,
        producer_indexing,
        consumer_tv,
        p2c_map,
        producer_indexing_from_idgraph.resolved_loop_domains,
        producer_indexing_from_idgraph.initial_concrete_index_map,
        loops,
        root_ind);

    root_ind = getProducerIndexWithHalo(producer_tv, i, root_ind, consumer_tv);

    root_ind = getProducerIndexWithGather(
        root_ind,
        i,
        producer_tv,
        consumer_tv,
        producer_indexing_from_idgraph.concrete_index.indexMap());

    root_ind = getProducerIndexWithPartialSplit(
        root_ind, root_dom[i], producer_tv, consumer_tv);

    if (root_ind->isZeroInt()) {
      continue;
    } else {
      auto strided_ind = SimplifyingIrBuilder::mulExpr(root_ind, strides[i]);
      if (i == root_dom.size() - 1 && vectorize_shift != nullptr) {
        strided_inds[i] =
            SimplifyingIrBuilder::addExpr(strided_ind, vectorize_shift);
      } else {
        strided_inds[i] = strided_ind;
      }
    }
  }

  return strided_inds;
}

namespace {

// Maps all producer domains to consumer with broadcast
// forwarding. Used to find the allocation position.
std::unordered_map<IterDomain*, IterDomain*> mapAllProducerDomainsToConsumer(
    TensorView* producer_tv,
    const TensorView* consumer_tv) {
  // This map has forwarded broadcast axes, it should only be used to compute
  // the allocation position of the producer, and to figure out which producer
  // indices are mapped to consumer trivial reductions.
  std::unordered_map<IterDomain*, IterDomain*> p2c_alloc_map;

  //  We want to replay producer as consumer instead of the other way around
  //  since consumer may have some broadcasted axes producer doesn't have
  //  merged into loops producer may use. If we did consumer as producer we
  //  wouldn't have this information in the mapping.
  auto replay_PasC = BestEffortReplay::replayPasC(
      producer_tv,
      consumer_tv,
      -1,
      PairwiseRootDomainMap(producer_tv, consumer_tv));

  // Grab consumer domain entries and reverse replay map. TODO: Maybe
  // TransformReplay::replayPasC could return this map
  for (auto id : consumer_tv->domain()->domain()) {
    const auto& c2p_map = replay_PasC.getReplay();
    auto c2p_it = c2p_map.find(id);
    if (c2p_it != c2p_map.end()) {
      auto c_id = c2p_it->first;
      auto p_id = c2p_it->second;
      p2c_alloc_map[p_id] = c_id;
    }
  }

  return p2c_alloc_map;
}

} // namespace

// Producer index for either shared or local memory
std::vector<Val*> Index::getNonGlobalProducerStridedIndices(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  const auto gpu_lower = GpuLower::current();

  // Replay producer to look like consumer so we can index on producer since our
  // loop nests look like consumer
  auto pairwise_map = PairwiseRootDomainMap(producer_tv, consumer_tv);
  auto producer_replayed_as_consumer =
      TransformReplay::replayPasC(producer_tv, consumer_tv, -1, pairwise_map)
          .first;

  ir_utils::TVDomainGuard domain_guard(
      producer_tv, producer_replayed_as_consumer);
  const auto p2c_alloc_map =
      mapAllProducerDomainsToConsumer(producer_tv, consumer_tv);

  // Map everything we can from reference to producer using compute at index
  // map. All producer id's don't exist in the compute at map. The rfactor axes
  // all may be, but since I haven't proven that to be the case, going to do a
  // more conservative approach, which is to use the consumer as a proxy between
  // producer to reference.
  std::unordered_map<IterDomain*, IterDomain*> index_map_ref_to_producer;
  std::unordered_map<IterDomain*, IterDomain*> c2p_index_map;
  std::unordered_map<IterDomain*, IterDomain*> p2c_index_map;

  // Map sent to best effort replay needs to match the exact incantation for
  // compute_at_mode.cpp with MappingMode::Index
  auto c2p_root_map =
      PairwiseRootDomainMap(producer_tv, consumer_tv, true)
          .mapConsumerToProducer(consumer_tv->domain(), producer_tv->domain());

  // This replay has to be consistent with compute at index map.
  BestEffortReplay replay_producer_as_consumer(
      producer_tv->domain()->domain(),
      consumer_tv->domain()->domain(),
      c2p_root_map);

  c2p_index_map = replay_producer_as_consumer.getReplay();
  p2c_index_map = invertOneToOneMap(c2p_index_map);

  // Forward vectorized IDs to index into producer correctly
  // We want p_id to be vectorized like consumer just for the indexing, then we
  // need to switch it back later. Store previous state here when changing. We
  // need to do this as replaying producer as consumer can use replay best
  // effort which means some domains may be the originals.
  std::vector<std::pair<IterDomain*, ParallelType>> p_id_backup;
  for (auto entry : c2p_index_map) {
    auto ref_id = ir_utils::caMapExactConcreteId(entry.first);
    auto p_id = entry.second;
    if (ref_id->getParallelType() == ParallelType::Vectorize) {
      p_id_backup.emplace_back(std::make_pair(p_id, p_id->getParallelType()));
      p_id->parallelize(ParallelType::Vectorize);
    } else if (ref_id->getParallelType() == ParallelType::MisalignedVectorize) {
      p_id->parallelize(ParallelType::MisalignedVectorize);
    }
  }

  auto producer_indexing_from_idgraph = getTensorIndexFromIdGraph(
      loops, consumer_tv, producer_tv, false, c2p_index_map);

  auto producer_indexing = producer_indexing_from_idgraph.index;

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

  auto producer_swizzled_index = index_swizzle;

  if (producer_tv->hasSwizzleOp()) {
    // Special handling needed on the new swizzle
    //  op pass:
    //  each swizzle op is local to the tensor,
    //  so ReplayPasC will not include the swizzle
    //  ops on the producer iterdomain. So would
    //  need to traverse forward the producer domain
    //  before the replay to get the swizzle ops.
    IndexSwizzle producer_swizzle2d(
        producer_tv,
        domain_guard.prevDomain(),
        producer_indexing.indexMap(),
        producer_indexing.extentMap(),
        producer_indexing.zeroDomains(),
        producer_indexing.zeroMergedIn());
    producer_swizzle2d.run();
    producer_swizzled_index = producer_swizzle2d;
  }

  // TODO: merge the two swizzle compute logic once the new one is ready.
  //  will need to replace cyclic shift swizzle with xor since swizzle2d
  //  doesn't have cyclic shift.
  const auto& index_map = producer_swizzled_index.indexMap();

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
    if (index_map.find(root_id) != index_map.end()) {
      continue;
    }

    // Maps to consumers trivial reduction, don't index
    if (p2c_alloc_map.find(root_id) != p2c_alloc_map.end() &&
        gpu_lower->trivialReductionInfo().isDerived(
            p2c_alloc_map.at(root_id))) {
      skip_indexing.emplace(root_id);
    }
  }

  std::vector<Val*> strided_inds(
      root_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    if (skip_indexing.count(root_dom[i])) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        index_map.find(root_dom[i]) != index_map.end(),
        "Couldn't find root mapping for ",
        producer_tv->toString(),
        " dim: ",
        i,
        " id: ",
        root_dom[i]->toString());

    auto root_ind_i = index_map.at(root_dom[i]);

    // index hoist must be done before the adjustments for halo
    root_ind_i = hoistProducerIndex(
        root_dom[i],
        producer_tv,
        producer_indexing,
        consumer_tv,
        p2c_index_map,
        producer_indexing_from_idgraph.resolved_loop_domains,
        producer_indexing_from_idgraph.initial_concrete_index_map,
        loops,
        root_ind_i);

    root_ind_i =
        getProducerIndexWithHalo(producer_tv, i, root_ind_i, consumer_tv);

    root_ind_i = getProducerIndexWithGather(
        root_ind_i,
        i,
        producer_tv,
        consumer_tv,
        producer_indexing_from_idgraph.concrete_index.indexMap());

    root_ind_i = getProducerIndexWithPartialSplit(
        root_ind_i, root_dom[i], producer_tv, consumer_tv);

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (const auto j : c10::irange(i + 1, root_dom.size())) {
      if (skip_indexing.count(root_dom[j])) {
        continue;
      }

      TORCH_INTERNAL_ASSERT(
          index_map.find(root_dom[j]) != index_map.end(),
          "Couldn't find root mapping for ",
          producer_tv->name(),
          " dim: ",
          j,
          " id: ",
          root_dom[j]->toString());

      auto root_ext_j = extent_map.find(root_dom[j]) == extent_map.end()
          ? root_dom[j]->extent()
          : extent_map.at(root_dom[j]);

      root_ext_j = getHaloExtentOfRootAxis(root_dom[j], root_ext_j);

      if (zero_domain_map.count(root_dom[j]) == 0) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = SimplifyingIrBuilder::mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds[i] = SimplifyingIrBuilder::mulExpr(root_ind_i, stride);
    } else {
      strided_inds[i] = root_ind_i;
    }
  }

  if (producer_tv->isDoubleBuffered() || producer_tv->isCircularBuffered()) {
    auto db_loop = gpu_lower->doubleBufferInfo().getDoubleBufferLoop(
        producer_tv, loops, true);
    if (db_loop != nullptr) {
      auto stage_depth = gpu_lower->doubleBufferInfo().getStageDepthFor(
          db_loop->iter_domain());
      auto loop_index =
          db_loop->isTrivial() ? db_loop->start() : db_loop->index();
      auto db_switch_index = SimplifyingIrBuilder::modExpr(
          loop_index, SimplifyingIrBuilder::create<Int>(stage_depth));
      auto original_alloc_size =
          gpu_lower->doubleBufferInfo().getOriginalAllocSize(producer_tv);
      auto db_strided_index =
          SimplifyingIrBuilder::mulExpr(db_switch_index, original_alloc_size);
      strided_inds.push_back(db_strided_index);
    }
  }

  return strided_inds;
}

std::vector<Val*> Index::getLinearIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  // Use domain guard to ignore the contiguity of
  //  consumer tv.
  TensorDomain* consumer_tv_no_contiguity_domain = nullptr;
  auto contiguity_vector =
      std::vector<bool>(consumer_tv->getMaybeRFactorDomain().size(), true);
  if (consumer_tv->hasRFactor()) {
    consumer_tv_no_contiguity_domain = IrBuilder::create<TensorDomain>(
        consumer_tv->getRootDomain(),
        consumer_tv->getRFactorDomain(),
        consumer_tv->domain()->domain(),
        contiguity_vector);
  } else {
    consumer_tv_no_contiguity_domain = IrBuilder::create<TensorDomain>(
        consumer_tv->getRootDomain(),
        consumer_tv->domain()->domain(),
        contiguity_vector);
  }

  ir_utils::TVDomainGuard domain_guard(
      consumer_tv, consumer_tv_no_contiguity_domain);

  // TODO:
  //  More optimization on the underlying tensor layout
  //   will be done in a follow up.
  return getGlobalConsumerStridedIndices(consumer_tv, loops);
}

std::vector<Val*> Index::getGlobalConsumerStridedIndices(
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getGlobalConsumerIndex");

  auto gpu_lower = GpuLower::current();

  auto index_from_id_graph = getTensorIndexFromIdGraph(loops, consumer_tv);

  auto consumer_indexing = index_from_id_graph.index;

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  // TODO: Abstract stride logic to reuse with producer indexing
  std::vector<Val*> strides(
      root_dom.size(), GpuLower::current()->kernel()->oneVal());
  {
    int stride_i = 0;
    for (const auto i : c10::irange(root_dom.size())) {
      if (root_dom[i]->isReduction() || root_dom[i]->isStride()) {
        strides[i] = GpuLower::current()->kernel()->oneVal();
        continue;
      }
      std::stringstream ss;
      ss << "T" << consumer_tv->name() << ".stride[" << stride_i++ << "]";
      strides[i] =
          SimplifyingIrBuilder::create<NamedScalar>(ss.str(), DataType::Int);
    }
  }

  TORCH_INTERNAL_ASSERT(
      root_dom.size() == consumer_tv->domain()->contiguity().size());
  Val* cur_contig_stride = GpuLower::current()->kernel()->oneVal();
  for (const auto i : c10::irange(root_dom.size())) {
    auto dim = root_dom.size() - i - 1;
    if (root_dom[dim]->isReduction() || root_dom[dim]->isStride()) {
      continue;
    }

    Val* root_ind = nullptr;
    if (consumer_indexing.indexMap().find(root_dom[dim]) !=
        consumer_indexing.indexMap().end()) {
      root_ind = consumer_indexing.indexMap().at(root_dom[dim]);
    } else if (root_dom[dim]->isBroadcast()) {
      root_ind = GpuLower::current()->kernel()->zeroVal();
    }

    TORCH_INTERNAL_ASSERT(
        root_ind != nullptr,
        "Couldn't find root mapping for ",
        consumer_tv->toString(),
        " dim: ",
        dim,
        " id: ",
        root_dom[dim]->toString());

    if (consumer_tv->domain()->contiguity()[dim]) {
      // If contig, used the stored stride which may be the previous
      // dimensions stride * previous dimensions size
      strides[dim] = cur_contig_stride;
      // Prepare for the next dimension which may also be contiguous, multiply
      // by extent of this dimension
      auto root_dim_extent = getHaloExtentOfRootAxis(root_dom[dim]);
      cur_contig_stride =
          SimplifyingIrBuilder::mulExpr(cur_contig_stride, root_dim_extent);
    } else {
      // If non contiguous dimension, keep local stride information, set cur
      // stride to local stride * local raw extent
      cur_contig_stride = SimplifyingIrBuilder::mulExpr(
          strides[dim], getHaloExtentOfRootAxis(root_dom[dim]));
    }
  }

  auto vectorize_shift =
      loops.empty() ? nullptr : loops.back()->vectorize_shift();

  // Global striding
  std::vector<Val*> strided_inds(
      root_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    // See a comment in indexing to root domains in getGlobalProducerIndex.
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast() ||
        gpu_lower->trivialReductionInfo().isDerived(root_dom[i]) ||
        root_dom[i]->isStride()) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        consumer_indexing.indexMap().find(root_dom[i]) !=
            consumer_indexing.indexMap().end(),
        "Couldn't find root mapping for ",
        consumer_tv->toString(),
        " dim: ",
        i,
        " id: ",
        root_dom[i]->toString());

    auto root_ind = consumer_indexing.indexMap().at(root_dom[i]);

    // index hoist must be done before the adjustments for halo
    root_ind = hoistConsumerIndex(
        root_dom[i],
        consumer_tv,
        consumer_indexing,
        index_from_id_graph.resolved_loop_domains,
        index_from_id_graph.initial_concrete_index_map,
        loops,
        root_ind);

    root_ind = SimplifyingIrBuilder::addExpr(
        root_ind, getGlobalConsumerOffsetWithPartialSplit(root_dom[i]));

    if (root_ind->isZeroInt()) {
      continue;
    } else {
      auto strided_ind = SimplifyingIrBuilder::mulExpr(root_ind, strides[i]);
      if (i == root_dom.size() - 1 && vectorize_shift != nullptr) {
        strided_inds[i] =
            SimplifyingIrBuilder::addExpr(strided_ind, vectorize_shift);
      } else {
        strided_inds[i] = strided_ind;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      strided_inds.size() == consumer_tv->getMaybeRFactorDomain().size());

  return strided_inds;
}

// Consumer index for either shared or local memory
std::vector<Val*> Index::getNonGlobalConsumerStridedIndices(
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  const auto gpu_lower = GpuLower::current();

  auto consumer_indexing_from_idgraph = getTensorIndexFromIdGraph(
      loops,
      consumer_tv,
      // Producer tv
      nullptr,
      // Index global
      false);

  auto consumer_indexing = consumer_indexing_from_idgraph.index;

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
  std::vector<Val*> strided_inds(
      root_dom.size(), GpuLower::current()->kernel()->zeroVal());
  for (const auto i : c10::irange(root_dom.size())) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast() ||
        gpu_lower->trivialReductionInfo().isDerived(root_dom[i]) ||
        root_dom[i]->isStride()) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        index_map.find(root_dom[i]) != index_map.end(),
        "Couldn't find root mapping for ",
        consumer_tv->toString(),
        " dim: ",
        i,
        " id: ",
        root_dom[i]->toString());

    auto root_ind_i = index_map.at(root_dom[i]);
    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // index hoist must be done before the adjustments for halo
    root_ind_i = hoistConsumerIndex(
        root_dom[i],
        consumer_tv,
        consumer_indexing,
        consumer_indexing_from_idgraph.resolved_loop_domains,
        consumer_indexing_from_idgraph.initial_concrete_index_map,
        loops,
        root_ind_i);

    // Compute striding for this index.
    Val* stride = nullptr;
    for (const auto j : c10::irange(i + 1, root_dom.size())) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction() ||
          gpu_lower->trivialReductionInfo().isDerived(root_dom[j]) ||
          root_dom[j]->isStride()) {
        continue;
      }

      TORCH_INTERNAL_ASSERT(
          index_map.find(root_dom[j]) != index_map.end(),
          "Couldn't find root mapping for ",
          consumer_tv->toString(),
          " dim: ",
          j,
          " id: ",
          root_dom[j]->toString());

      auto root_ext_j = extent_map.find(root_dom[j]) == extent_map.end()
          ? root_dom[j]->extent()
          : extent_map.at(root_dom[j]);

      root_ext_j = getHaloExtentOfRootAxis(root_dom[j], root_ext_j);

      if (zero_domain_map.count(root_dom[j]) == 0) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = SimplifyingIrBuilder::mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds[i] = SimplifyingIrBuilder::mulExpr(root_ind_i, stride);
    } else {
      strided_inds[i] = root_ind_i;
    }
  }

  // This check was originally done in getConsumerStridedIndices, but
  // the number of strided index values depends on the loop where the
  // consumer tensor is located. If it's double buffered and not in
  // the prologue loop, strided_inds ends up having one more
  // index, so it's just much simpler to check here before adding the
  // additional index for double buffering.
  TORCH_INTERNAL_ASSERT(
      strided_inds.size() == consumer_tv->getMaybeRFactorDomain().size());

  if (consumer_tv->isDoubleBuffered() || consumer_tv->isCircularBuffered()) {
    auto db_loop =
        gpu_lower->doubleBufferInfo().getDoubleBufferLoop(consumer_tv, loops);
    auto stage_depth =
        gpu_lower->doubleBufferInfo().getStageDepthFor(db_loop->iter_domain());
    bool is_circular_buffer_loop = stage_depth > 2;
    bool is_prolog =
        db_loop->doubleBufferLoopStage() == DoubleBufferLoopStage::Prolog;

    Val* db_switch_index = nullptr;

    // In double buffered we don't materialize the prolog loop as there will
    //  be only one iteration. In circular buffer case we materialize the
    //  prolog loop as well covering the first N-1 iterations, N being the
    //  stage depth.
    if (!is_prolog || is_circular_buffer_loop) {
      if (is_prolog && is_circular_buffer_loop) {
        // The buffer switching logic is the same as original index
        //  in the case of circular buffer prolog.
        db_switch_index = db_loop->index();
      } else {
        // Switching index generated for main loop or epilog component.
        db_switch_index = SimplifyingIrBuilder::modExpr(
            SimplifyingIrBuilder::addExpr(
                db_loop->index(),
                SimplifyingIrBuilder::create<Int>(stage_depth - 1)),
            SimplifyingIrBuilder::create<Int>(stage_depth));
      }

      // Use the generated switching buffer index to access the buffer space.
      auto original_alloc_size =
          gpu_lower->doubleBufferInfo().getOriginalAllocSize(consumer_tv);
      auto db_strided_index =
          SimplifyingIrBuilder::mulExpr(db_switch_index, original_alloc_size);
      strided_inds.push_back(db_strided_index);
    }
  }

  return strided_inds;
}

std::vector<Val*> Index::getProducerStridedIndices(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getProducerStridedIndices");
  if (producer->domain()->noReductions().size() == 0) {
    return std::vector<Val*>(
        producer->getMaybeRFactorDomain().size(),
        GpuLower::current()->kernel()->zeroVal());
  }

  std::vector<Val*> strided_indices;
  if (producer->getMemoryType() == MemoryType::Global) {
    strided_indices =
        getGlobalProducerStridedIndices(producer, consumer, loops);
  } else {
    strided_indices =
        getNonGlobalProducerStridedIndices(producer, consumer, loops);
  }

  TORCH_INTERNAL_ASSERT(
      strided_indices.size() ==
      producer->getMaybeRFactorDomain().size() +
          (producer->isDoubleBuffered() || producer->isCircularBuffered() ? 1
                                                                          : 0));

  return strided_indices;
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  auto strided_indices = getProducerStridedIndices(producer, consumer, loops);
  return SimplifyingIrBuilder::create<kir::TensorIndex>(
      producer, strided_indices);
}

std::vector<Val*> Index::getConsumerStridedIndices(
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getConsumerStridedIndices");
  if (consumer->domain()->noReductions().size() == 0) {
    return std::vector<Val*>(
        consumer->getMaybeRFactorDomain().size(),
        GpuLower::current()->kernel()->zeroVal());
  }

  std::vector<Val*> strided_indices;
  if (consumer->getMemoryType() == MemoryType::Global) {
    strided_indices = getGlobalConsumerStridedIndices(consumer, loops);
  } else {
    strided_indices = getNonGlobalConsumerStridedIndices(consumer, loops);
  }

  return strided_indices;
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  auto strided_indices = getConsumerStridedIndices(consumer, loops);
  return SimplifyingIrBuilder::create<kir::TensorIndex>(
      consumer, strided_indices);
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

// Find iteration domains in the history of a consumer to predicate comprised
// only of merge operations. Only return iteration domains that are subsequently
// fed into a split, or are in the provided domain. In other words, we don't
// want to return every IterDomain that's contiguous, just the one closest to
// the leaves. Predicates are not associated with physical memory so we can
// treat all of them as contiguous merges.
//
// TODO: This seems to have a large overlap with ContigIDs. Consider
// refactoring.
std::vector<PredicateDomainInfo> getPredicateContigIds(
    TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& consumer_index_map) {
  const auto gpu_lower = GpuLower::current();

  const auto& consumer_root_domain = consumer_tv->getRootDomain();

  std::vector<IterDomain*> contiguous_ids = consumer_root_domain;

  if (contiguous_ids.empty()) {
    return std::vector<PredicateDomainInfo>();
  }

  // If root IDs are partial, i.e., start is non-zero and stop is not
  // equal to extent, predication can't be done with merged domains as
  // start and stop information is only available with root
  // domains. Similarly, merged domains don't have enough information
  // about halo to do correct predication, so they must be excluded.
  std::unordered_set<IterDomain*> excluded_ids;

  for (auto consumer_root_id : consumer_root_domain) {
    if (gpu_lower->haloInfo().getRootAxisInfo(consumer_root_id).hasHalo()) {
      excluded_ids.insert(consumer_root_id);
      continue;
    }
    if (consumer_root_id->maybePartial()) {
      excluded_ids.insert(consumer_root_id);
      continue;
    }
    // When consumer_root_id is a broadcast domain, do not allow contig
    // predication as the merged output is not mapped with the
    // reference unless the concrete domain is also a broadcast
    // domain.
    if (consumer_root_id->isBroadcast() &&
        !GpuLower::current()
             ->caMap()
             ->getConcreteMappedID(consumer_root_id, IdMappingMode::PERMISSIVE)
             ->isBroadcast()) {
      excluded_ids.insert(consumer_root_id);
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
         gather_expr->windowShape().at(consumer_root_pos) != 1)) {
      excluded_ids.insert(consumer_root_id);
    }
  }

  // Run through iteration domain history
  auto exprs = StmtSort::getExprs(
      consumer_tv->fusion(),
      {consumer_tv->domain()->domain().begin(),
       consumer_tv->domain()->domain().end()});

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

      // Do not try to predicate the merge output domain if the output
      // domain has not a predicate that is mapped from the reference.
      // See FusionContigPredicate_CUDA for a concrete example.
      if (consumer_index_map.find(merge->out()) == consumer_index_map.end()) {
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
        {consumer_root_domain.begin(), consumer_root_domain.end()});
    auto contig_root_ids = ir_utils::filterByType<IterDomain>(contig_root_vals);
    PredicateDomainInfo contig_id_info;
    contig_id_info.id = contig_id;
    contig_id_info.covered_ids = std::unordered_set<IterDomain*>(
        contig_root_ids.begin(), contig_root_ids.end());
    contig_id_infos.push_back(contig_id_info);
  }
  return contig_id_infos;
}

std::vector<PredicateDomainInfo> getNonDivisibleConsumerDomainsToPredicate(
    TensorView* consumer_tv) {
  const auto& non_divisible_split_info =
      GpuLower::current()->nonDivisibleSplitInfo();

  std::vector<PredicateDomainInfo> pred_info_vec;

  auto it = non_divisible_split_info.splitsToPredicate().find(consumer_tv);
  if (it == non_divisible_split_info.splitsToPredicate().end()) {
    return {};
  }

  const auto& splits_to_predicate = it->second;

  for (auto split : splits_to_predicate) {
    PredicateDomainInfo info{split->in(), {split->in()}, true};
    pred_info_vec.emplace_back(info);
  }

  return pred_info_vec;
}

bool needsPadding(TensorView* tv) {
  auto shift_expr = dynamic_cast<ShiftOp*>(tv->definition());
  auto gather_expr = dynamic_cast<GatherOp*>(tv->definition());

  return (shift_expr != nullptr && shift_expr->hasPadding()) ||
      (gather_expr != nullptr && gather_expr->hasPadding());
}

// Get an additional offset of a stop index when building a predicate
// for unswitch. Initial stop indices generated at
// getPredicateIndexingFromIdGraph do not take halo into account, and the
// adjustment for halo is done as an additional offset to the final index value
// so that unswitch predicates can be compared with each other by just looking
// at the additional offsets.
//
// consumer_root_id: the domain for which a stop predicate is being built.
int getUnswitchStopOffset(
    IterDomain* consumer_root_id,
    TensorView* consumer_tv) {
  const auto gpu_lower = GpuLower::current();

  AxisHaloInfo halo_info =
      gpu_lower->haloInfo().getRootAxisInfo(consumer_root_id);

  // If the consumer root domain to predicate does not have halo, no
  // adjustment is required.
  if (!halo_info.hasHalo()) {
    return 0;
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
    return 0;
  }
}

std::pair<Val*, Val*> getStartAndStopOffsetsForShift(
    TensorView* consumer_tv,
    IterDomain* consumer_id,
    bool padding_predicate) {
  TORCH_INTERNAL_ASSERT(consumer_id != nullptr);

  auto shift_expr = dynamic_cast<ShiftOp*>(consumer_tv->definition());

  // Adjustment is not necessary if not shift.
  // Even so, padding predicate does not need any adjustment.
  if (shift_expr == nullptr || padding_predicate) {
    return {
        GpuLower::current()->kernel()->zeroVal(),
        GpuLower::current()->kernel()->zeroVal()};
  }

  const auto root_axis_pos = consumer_tv->domain()->rootPosOf(consumer_id);

  // The first or last N elements, where N is the padding width,
  // correspond to the padding predicate.

  const auto shift_offset = shift_expr->offset(root_axis_pos);
  const auto pad_width = shift_expr->padWidth().at(root_axis_pos);

  int start_offset = 0;
  int stop_offset = 0;

  if (shift_offset > 0) {
    start_offset = -pad_width;
  } else if (shift_offset < 0) {
    stop_offset = pad_width;
  }

  return {
      SimplifyingIrBuilder::create<Int>(start_offset),
      SimplifyingIrBuilder::create<Int>(stop_offset)};
}

std::pair<Val*, Val*> getStartAndStopOffsetsForGather(
    TensorView* consumer_tv,
    IterDomain* consumer_id,
    const std::unordered_map<IterDomain*, Val*>& ref_start_index_map,
    const std::unordered_map<IterDomain*, Val*>& ref_stop_index_map,
    bool padding_predicate) {
  TORCH_INTERNAL_ASSERT(consumer_id != nullptr);

  // Adjustment is not necessary if not gather. Even so, padding
  // predicate does not need any adjustment.
  if (!consumer_tv->definition()->isA<GatherOp>() || padding_predicate) {
    return {
        GpuLower::current()->kernel()->zeroVal(),
        GpuLower::current()->kernel()->zeroVal()};
  }

  const auto root_axis_pos = consumer_tv->domain()->rootPosOf(consumer_id);

  auto producer_start_offset = getProducerOffsetWithGather(
      root_axis_pos, consumer_tv, ref_start_index_map);

  auto producer_stop_offset = getProducerOffsetWithGather(
      root_axis_pos, consumer_tv, ref_stop_index_map);

  auto consumer_start_offset = GpuLower::current()->kernel()->zeroVal();
  auto consumer_stop_offset = GpuLower::current()->kernel()->zeroVal();

  if (producer_start_offset->isZeroInt() && producer_stop_offset->isZeroInt()) {
    return {consumer_start_offset, consumer_stop_offset};
  }

  Val* start_offset = nullptr;
  Val* stop_offset = nullptr;

  // In the normal case, take the minimum of the start and the
  // maximum of the stop offsets. If there's no padding, the producer
  // offset must be always larger than the consumer
  // offset. So, the consumer and produce offsets can be always used
  // for the start and stop offsets, respectively.
  const auto pad_left =
      consumer_tv->definition()->as<GatherOp>()->padWidth()[root_axis_pos][0];
  const auto pad_right =
      consumer_tv->definition()->as<GatherOp>()->padWidth()[root_axis_pos][1];
  const auto window_size =
      consumer_tv->definition()->as<GatherOp>()->windowShape()[root_axis_pos];

  // consumer index: index
  // producer index: index + window_index - pad_left
  //
  // consumer extent: ext
  // producer extent: ext + window_size - 1 - pad_left - pad_right
  //
  // consumer stop pred: index < ext
  // producer stop pred: index + window_index - pad_left < ext + window_size - 1
  // - pad_left - pad_right
  //                  -> index + window_index - pad_left - (window_size - 1 -
  //                  pad_left - pad_right) < ext
  //                  -> index + window_index - (window_size - 1 - pad_right) <
  //                  ext
  //
  // consumer start pred: index >= 0
  // producer start pred: index + window_index - pad_left >= 0

  const auto producer_ext_adj = window_size - 1 - pad_left - pad_right;
  producer_stop_offset = SimplifyingIrBuilder::subExpr(
      producer_stop_offset,
      SimplifyingIrBuilder::create<Int>(producer_ext_adj));

  // As commented above, when pad_left is zero, the consumer predicate
  // is always more restrictive than the producer predicate.
  if (pad_left == 0) {
    start_offset = consumer_start_offset;
  } else {
    start_offset = SimplifyingIrBuilder::minExpr(
        consumer_start_offset, producer_start_offset);
  }

  // As commented above, when pad_right is zero, the consumer
  // predicate is always more restrictive than the producer
  // predicate.
  if (pad_right == 0) {
    stop_offset = consumer_stop_offset;
  } else {
    stop_offset = SimplifyingIrBuilder::maxExpr(
        consumer_stop_offset, producer_stop_offset);
  }

  TORCH_INTERNAL_ASSERT(start_offset != nullptr);
  TORCH_INTERNAL_ASSERT(stop_offset != nullptr);

  return {start_offset, stop_offset};
}

// Get the start and stop limit offsets that define the valid range to
// compute. In the simplest case, they are just 0 and
// IterDomain::extent. However, IterDomain may have non-zero start and
// stop that's different from extent. Also, when IterDomain has halo,
// the actual offsets of the logical start and stop positions are
// shifted.
std::pair<Val*, Val*> getStartAndStopLimitOffsets(
    IterDomain* consumer_id,
    bool padding_predicate,
    bool non_divisible_pred) {
  const auto gpu_lower = GpuLower::current();

  TORCH_INTERNAL_ASSERT(consumer_id != nullptr);

  Val* start_limit = consumer_id->start();
  Val* stop_limit = SimplifyingIrBuilder::negExpr(consumer_id->stopOffset());

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
      start_limit =
          SimplifyingIrBuilder::addExpr(start_limit, halo_info.width(0));
      stop_limit =
          SimplifyingIrBuilder::addExpr(stop_limit, halo_info.width(0));
    } else {
      // In case of the padding predicate, the whole range, including both left
      // and right halo regions, is computed.
      stop_limit = SimplifyingIrBuilder::addExpr(stop_limit, halo_info.width());
    }
  } else {
    // For non-divisible predicates, the index must be predicated such
    // that it is less than the extent of the predicated ID +
    // halo. Note that getRootAxisInfo doesn't work since consumer_id
    // isn't a root domain.
    if (gpu_lower->haloInfo().hasHaloWidth(consumer_id)) {
      auto halo = gpu_lower->haloInfo().getHaloWidth(consumer_id);
      stop_limit = SimplifyingIrBuilder::addExpr(stop_limit, halo);
    }
  }

  return {start_limit, stop_limit};
}

// Get the offsets for the start and stop predicates. The offsets
// are to be added to the index.
std::pair<Val*, Val*> getStartAndStopOffsets(
    IterDomain* consumer_id,
    TensorView* consumer_tv,
    const std::unordered_map<IterDomain*, Val*>& consumer_start_index_map,
    const std::unordered_map<IterDomain*, Val*>& consumer_stop_index_map,
    bool padding_predicate,
    bool unswitch,
    bool non_divisible_pred) {
  // By default, the offsets for the start and stop predicates are
  // just zero. All halo-related adjustments are done at root domains,
  // so consumer_id is not a root domain, no adjustment is required.
  if (consumer_id->definition() != nullptr && !non_divisible_pred) {
    return {
        GpuLower::current()->kernel()->zeroVal(),
        GpuLower::current()->kernel()->zeroVal()};
  }

  auto consumer_def = consumer_tv->definition();

  Val* start_offset = GpuLower::current()->kernel()->zeroVal();
  Val* stop_offset = GpuLower::current()->kernel()->zeroVal();

  // These adjustments are not required when predicating non-divisible splits
  if (!non_divisible_pred) {
    if (consumer_def->isA<ShiftOp>()) {
      std::tie(start_offset, stop_offset) = getStartAndStopOffsetsForShift(
          consumer_tv, consumer_id, padding_predicate);
    } else if (consumer_def->isA<GatherOp>()) {
      std::tie(start_offset, stop_offset) = getStartAndStopOffsetsForGather(
          consumer_tv,
          consumer_id,
          consumer_start_index_map,
          consumer_stop_index_map,
          padding_predicate);
    }

    // Adjustment for partial split
    auto partial_split_offset =
        getGlobalConsumerOffsetWithPartialSplit(consumer_id);
    start_offset =
        SimplifyingIrBuilder::addExpr(start_offset, partial_split_offset);
    stop_offset =
        SimplifyingIrBuilder::addExpr(stop_offset, partial_split_offset);

    // If generating a predicate for unswitch, adjust the stop offset to
    // accommodate the addition of halo to the loop stop. See the
    // comment in getPredicateIndexingFromIdGraph as well.
    if (unswitch) {
      TORCH_INTERNAL_ASSERT(
          !padding_predicate, "Unswitch should not use the padding predicate");
      auto stop_unswitch_offset =
          getUnswitchStopOffset(consumer_id, consumer_tv);
      stop_offset =
          SimplifyingIrBuilder::addExpr(stop_offset, stop_unswitch_offset);
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

  start_offset = SimplifyingIrBuilder::subExpr(start_offset, limits.first);
  stop_offset = SimplifyingIrBuilder::subExpr(stop_offset, limits.second);

  return {start_offset, stop_offset};
}

// A partial value of a start offset is returned if determined to be
// safe. Nullptr is returned if it can be omitted completely.
Val* simplifyStartOffset(Val* start_offset) {
  // Start predicate can be omitted when start_offset >= 0.
  auto offset_val = start_offset->as<Int>()->value();
  if (offset_val.has_value() && offset_val.value() >= 0) {
    return nullptr;
  }

  // start_offset may look like min(0, window_index - pad). Then, can
  // remove min and leave the rhs only.
  auto def = dynamic_cast<BinaryOp*>(start_offset->definition());
  if (def != nullptr && def->getBinaryOpType() == BinaryOpType::Min &&
      def->lhs()->isZeroInt()) {
    return def->rhs();
  }

  return start_offset;
}

bool canOmitStopPredicate(
    Val* stop_index,
    Val* stop_offset,
    IterDomain* contig_id) {
  bool index_simple = stop_index->definition() == nullptr;
  // The definition may be just adding the magic zero, which can be
  // effectively considered "simple"
  if (!index_simple && isProtectedWithMagicZero(stop_index)) {
    // Make sure the lhs of stop_index is simple.
    auto lhs = stop_index->definition()->as<BinaryOp>()->lhs();
    if (lhs->definition() == nullptr) {
      index_simple = true;
    }
  }

  if (!index_simple) {
    return false;
  }

  const auto gpu_lower = GpuLower::current();

  // Stop predicate: stop_index + stop_offset < extent, where
  // stop_index ranges from 0 to (extent + halo), so this can be
  // omitted if extent + halo + stop_offset < extent, i.e., halo +
  // stop_offset <= 0.

  auto stop_offset_val = stop_offset->as<Int>()->value();

  // If they are not compile-time constant, can't prove the
  // condition.
  if (!stop_offset_val.has_value()) {
    return false;
  }

  // Note that when a root domain is halo extended, it is the domain
  // to be predicated, not its merged contig id even if it exists. So,
  // if contig_id does not have root axis info, contig_id is
  // guaranteed to have no halo.
  auto halo_ext = gpu_lower->haloInfo().hasRootAxisInfo(contig_id)
      ? gpu_lower->haloInfo().getRootAxisInfo(contig_id).width()
      : 0;

  if (halo_ext + stop_offset_val.value() > 0) {
    return false;
  }

  // When the domain is parallelized, the parallel dimension must be
  // exact. Otherwise, there would be extra threads/blocks that need
  // to be predicated out.
  if (isParallelTypeThread(contig_id->getParallelType())) {
    if (!gpu_lower->parallelDimensionMap().isExact(
            contig_id->getParallelType())) {
      return false;
    }
    // If the domain has halo, the loop is expanded by the halo
    // extent, so we can't prove the loop extent is the same as the
    // parallel dimension.
    if (halo_ext != 0) {
      return false;
    }
  }

  return true;
}

std::pair<Val*, Val*> hoistPredicates(
    Val* start_index,
    Val* stop_index,
    const std::vector<kir::ForLoop*>& loops,
    std::vector<IterDomain*> loop_domains,
    const std::unordered_map<IterDomain*, Val*>& start_initial_loop_index_map,
    const std::unordered_map<IterDomain*, Val*>& stop_initial_loop_index_map,
    kir::ForLoop* unswitch_or_vec_loop,
    IterDomain* predicated_consumer_id,
    TensorView* predicated_consumer_tv) {
  const std::pair<Val*, Val*> same_indices{start_index, stop_index};

  if (isOptionDisabled(DisableOption::IndexHoist)) {
    return same_indices;
  }

  const auto start_is_same_as_stop = stop_index == start_index;

  Val* hoisted_stop_index = nullptr;

  if (stop_index->definition() == nullptr) {
    // If the index doens't have an expression, nothing to hoist
    hoisted_stop_index = stop_index;
  } else {
    bool inserted = false;
    std::tie(hoisted_stop_index, inserted) =
        GpuLower::current()->commonIndexMap().insert(
            predicated_consumer_id,
            predicated_consumer_tv->domain(),
            loop_domains,
            stop_initial_loop_index_map,
            loops,
            stop_index);
  }

  Val* hoisted_start_index = nullptr;
  if (start_is_same_as_stop) {
    hoisted_start_index = hoisted_stop_index;
  } else if (start_index->definition() == nullptr) {
    hoisted_start_index = start_index;
  } else {
    bool inserted = false;
    std::tie(hoisted_start_index, inserted) =
        GpuLower::current()->commonIndexMap().insert(
            predicated_consumer_id,
            predicated_consumer_tv->domain(),
            loop_domains,
            start_initial_loop_index_map,
            loops,
            start_index);
  }

  return {hoisted_start_index, hoisted_stop_index};
}

// Updates a loop index map with a loop index protected by magic zero
std::unordered_map<IterDomain*, Val*> updateInitialLoopIndexMap(
    const std::unordered_map<IterDomain*, Val*>& initial_loop_index_map,
    const IndexMagicZeroInfo& magic_zero_info) {
  if (magic_zero_info.original_loop_index != nullptr) {
    TORCH_INTERNAL_ASSERT(magic_zero_info.protected_loop_index != nullptr);
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        magic_zero_info.loop_id, IdMappingMode::EXACT);
    auto updated_map = initial_loop_index_map;
    updated_map[concrete_loop_id] = magic_zero_info.protected_loop_index;
    return updated_map;
  } else {
    return initial_loop_index_map;
  }
}

} // namespace

// Returns predicates and the concrete (by loop map) root domains they cover
std::vector<RootPredicateInfo> Index::getReferenceRootPredicates(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    kir::ForLoop* unswitch_or_vec_loop,
    bool shift_padding) {
  FUSER_PERF_SCOPE("GpuLower::Lower::Index::getReferenceRootPredicates");

  const auto gpu_lower = GpuLower::current();

  const bool is_unswitch = unswitch_or_vec_loop != nullptr;

  // Nothing needs to be done when padding is not required.
  if (shift_padding && !needsPadding(consumer_tv)) {
    return {RootPredicateInfo::getFalseInfo()};
  }

  auto db_axis = gpu_lower->doubleBufferInfo().getDoubleBufferAxis(consumer_tv);

  // Indexing is done without considering contig merging. Actual
  // predicated domains are determined by considering contiguity.
  const ContigIDs contig_finder(
      consumer_tv->domain()->domain(),
      consumer_tv->getMaybeRFactorDomain(),
      std::vector<bool>(consumer_tv->getMaybeRFactorDomain().size(), false),
      {});

  // Generate start and stop indexing from idgraph.
  //
  // Both start and stop positions may need to be predicated. Indexing
  // differs when generating predicates for unswitch.
  // NOTE: If we could find-and-replace KIR nodes, we could just
  // generate one index map, clone it and replace the loop-to-index
  // mappings of unswitched loops for the start predicate.

  auto stop_indexing_from_idgraph = getPredicateIndexingFromIdGraph(
      loops, consumer_tv, unswitch_or_vec_loop, db_axis, false);
  const auto consumer_stop_indexing = stop_indexing_from_idgraph.index;
  const auto& consumer_stop_index_map = consumer_stop_indexing.indexMap();

  // If not unswitch, share the same indexing map as the stop index
  // map
  const auto start_indexing_from_idgraph = is_unswitch
      ? getPredicateIndexingFromIdGraph(
            loops, consumer_tv, unswitch_or_vec_loop, db_axis, true)
      : stop_indexing_from_idgraph;
  const auto consumer_start_indexing = start_indexing_from_idgraph.index;
  const auto& consumer_start_index_map = consumer_start_indexing.indexMap();

  // Get the contiguous ids we need to generate predicates for
  auto contig_id_infos =
      getPredicateContigIds(consumer_tv, consumer_stop_index_map);

  auto non_divisible_splits =
      getNonDivisibleConsumerDomainsToPredicate(consumer_tv);
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

    const auto consumer_stop_indexing_it =
        consumer_stop_index_map.find(contig_id);

    // First condition below happens with Misaligned predicates, where
    // inner-most vectorized loops are not included in the loops
    // parameter. Predicates involving vectorized loops are separately
    // generated in lower_misaligned_vectorization.
    //
    // Second condition is simply to avoid predication on broadcasting axes as
    // it's not required.
    if (consumer_stop_indexing_it == consumer_stop_index_map.end() ||
        consumer_stop_indexing_it->second->isZeroInt()) {
      continue;
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

    std::tie(info.start_offset_, info.stop_offset_) = getStartAndStopOffsets(
        contig_id,
        consumer_tv,
        consumer_start_index_map,
        consumer_stop_index_map,
        shift_padding,
        unswitch_or_vec_loop != nullptr,
        contig_id_entry.is_non_divisible_split);

    auto stop_index = consumer_stop_indexing_it->second;
    auto start_index = consumer_start_index_map.at(contig_id);

    IndexMagicZeroInfo start_magic_zero_info;
    IndexMagicZeroInfo stop_magic_zero_info;

    // When the start and stop indices are not the same, apply the
    // magic-zero protection separately for both of them.
    if (stop_index != start_index) {
      start_magic_zero_info = protectPredicateIndexWithMagicZero(
          start_index, start_indexing_from_idgraph, loops);
      stop_magic_zero_info = protectPredicateIndexWithMagicZero(
          stop_index, stop_indexing_from_idgraph, loops);
    } else {
      stop_magic_zero_info = protectPredicateIndexWithMagicZero(
          stop_index, stop_indexing_from_idgraph, loops);
      start_magic_zero_info = stop_magic_zero_info;
    }

    start_index = start_magic_zero_info.index;
    stop_index = stop_magic_zero_info.index;

    // Update the loop-index map with the magic-zero protection info
    // before passing it to the hoisting function
    std::tie(start_index, stop_index) = hoistPredicates(
        start_index,
        stop_index,
        loops,
        stop_indexing_from_idgraph.resolved_loop_domains,
        updateInitialLoopIndexMap(
            start_indexing_from_idgraph.initial_concrete_index_map,
            start_magic_zero_info),
        updateInitialLoopIndexMap(
            stop_indexing_from_idgraph.initial_concrete_index_map,
            stop_magic_zero_info),
        unswitch_or_vec_loop,
        contig_id,
        consumer_tv);

    // Build predicates for start positions as:
    //   start_index + start_offset >= 0
    auto start_offset = simplifyStartOffset(info.start_offset_);
    if (start_offset == nullptr) {
      info.start_predicate_ = GpuLower::current()->kernel()->trueVal();
    } else {
      auto offsetted_start_index =
          SimplifyingIrBuilder::addExpr(start_index, start_offset);
      auto start_pred =
          SimplifyingIrBuilder::geExpr(
              offsetted_start_index, GpuLower::current()->kernel()->zeroVal())
              ->as<Bool>();
      info.start_predicate_ = start_pred;
    }

    // Build predicates for stop positions as:
    //   stop_index + stop_offset < IterDomain::extent
    auto stop_offset = info.stop_offset_;
    if (canOmitStopPredicate(stop_index, stop_offset, contig_id)) {
      info.stop_predicate_ = GpuLower::current()->kernel()->trueVal();
    } else {
      auto offsetted_stop_index =
          SimplifyingIrBuilder::addExpr(stop_index, stop_offset);
      auto stop_pred = SimplifyingIrBuilder::ltExpr(
                           offsetted_stop_index, contig_id->extent())
                           ->as<Bool>();
      info.stop_predicate_ = stop_pred;
    }

    for (auto consumer_id : contig_id_entry.covered_ids) {
      info.root_ids_.insert(consumer_id);
    }
    pred_info_vec.emplace_back(info);
  }

  return pred_info_vec;
}

RootPredicateInfo RootPredicateInfo::getFalseInfo() {
  RootPredicateInfo info;
  info.start_predicate_ = GpuLower::current()->kernel()->falseVal();
  info.stop_predicate_ = GpuLower::current()->kernel()->falseVal();

  return info;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
