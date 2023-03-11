#include <arith.h>
#include <index_compute.h>
#include <instrumentation.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <kernel_expr_evaluator.h>
#include <kernel_ir.h>
#include <lower2device.h>
#include <lower_index_compute.h>
#include <lower_shift.h>
#include <lower_utils.h>

#include <functional>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

Expr* ShiftPredicateInserter::insert(
    Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    Bool* thread_pred,
    bool within_unswitch) {
  const auto gpu_lower = GpuLower::current();

  TensorView* out_tv = ir_utils::getTvOutput(expr);
  TORCH_INTERNAL_ASSERT(out_tv != nullptr, "Missing TensorView output");

  const bool needs_shift_predicate =
      gpu_lower->haloInfo()->needsShiftPredicate(out_tv->definition());
  if (!needs_shift_predicate) {
    return expr;
  }

  // The conditional branches to create:
  //
  // if (shift_pred) {
  //   consumer = producer;
  // } else {
  //   if (padding_pred) {
  //     consumer = 0;
  //   }
  // }

  kir::Predicate* thread_pred_expr = nullptr;
  if (within_unswitch) {
    thread_pred_expr = IrBuilder::create<kir::Predicate>(thread_pred);
  }

  kir::Predicate* shift_pred = within_unswitch
      ? thread_pred_expr
      : IrBuilder::create<kir::Predicate>(
            PredicateType::Shift, expr, thread_pred);

  // If the expr involves a thread-block barrier, set the predicate of
  // the expr with shift_pred. Since the expr is not shift, the
  // padding is safe to omit.
  if (lower_utils::hasBlockSync(expr, gpu_lower->threadPredMap())) {
    return expr->withPredicate(shift_pred);
  }

  auto shift_ite = IrBuilder::create<kir::IfThenElse>(shift_pred);

  auto& scope = loops.back()->body();

  // Insert the if statement
  scope.insert_before(expr, shift_ite);

  // Remove the expr from the list
  scope.erase(expr);

  // Place the expr inside the if statement
  shift_ite->thenBody().push_back(expr);

  // No padding condition is required if this is within unswitch.
  if (within_unswitch) {
    return expr;
  }

  // Padding by zero
  kir::Predicate* padding_pred = IrBuilder::create<kir::Predicate>(
      PredicateType::Padding, expr, thread_pred);
  auto bounds_ite = IrBuilder::create<kir::IfThenElse>(padding_pred);
  const int pad_value = 0;
  auto pad_expr = IrBuilder::create<UnaryOp>(
      UnaryOpType::Set, out_tv, IrBuilder::create<Int>(pad_value));
  bounds_ite->thenBody().push_back(pad_expr);
  // Insert the else block
  shift_ite->elseBody().push_back(bounds_ite);

  return expr;
}

int AxisHaloInfo::width() const {
  return width(0) + width(1);
}

int AxisHaloInfo::width(int pos) const {
  TORCH_INTERNAL_ASSERT(pos >= 0 && pos < 2);
  return widths_[pos];
}

void AxisHaloInfo::setWidth(int pos, int width) {
  TORCH_INTERNAL_ASSERT(pos >= 0 && pos < 2);
  widths_[pos] = width;
}

void AxisHaloInfo::merge(int pos, int other) {
  auto new_width = std::max(width(pos), other);
  setWidth(pos, new_width);
}

void AxisHaloInfo::merge(const AxisHaloInfo& other) {
  for (const auto i : c10::irange(widths_.size())) {
    merge(i, other.width(i));
  }
}

bool AxisHaloInfo::hasHalo() const {
  return std::any_of(
      widths_.begin(), widths_.end(), [](auto w) { return w != 0; });
}

std::string AxisHaloInfo::toString() const {
  std::stringstream ss;
  ss << "<" << width(0) << ", " << width(1) << ">";
  return ss.str();
}

bool HaloInfo::hasRootAxisInfo(IterDomain* id) const {
  return root_axis_map_.find(id) != root_axis_map_.end();
}

const AxisHaloInfo& HaloInfo::getRootAxisInfo(IterDomain* id) const {
  // TODO: Enable this check, was failing in many tests
  // TORCH_INTERNAL_ASSERT(
  //     id->definition() == nullptr || id->isRFactorProduct(),
  //     "Invalid IterDomain: ",
  //     id);
  auto it = root_axis_map_.find(id);
  TORCH_INTERNAL_ASSERT(
      it != root_axis_map_.end(),
      "Halo root axis info not found for ",
      id->toString());
  return it->second;
}

void HaloInfo::setRootAxisInfo(
    IterDomain* id,
    const AxisHaloInfo& root_axis_info) {
  root_axis_map_[id] = root_axis_info;

  initializeFromRootAxisInfo(id);
  return;
}

HaloInfo::HaloInfo(Fusion* fusion, std::shared_ptr<const ComputeAtMap> ca_map)
    // Make a copy of the permissive map for extent comparators
    : permissive_map_(ca_map->idGraph().permissiveNodes()) {
  const auto vals = fusion->usedMathVals();
  auto tvs = ir_utils::filterByType<TensorView>(vals);

  // Initialize all root axis info
  for (auto tv : tvs) {
    for (auto root_axis : tv->getRootDomain()) {
      setRootAxisInfo(root_axis, AxisHaloInfo());
    }
    // Just adds a placeholder to make it not fail. Reduction and
    // rfactor support is not yet in place.
    if (tv->hasRFactor()) {
      for (auto rf_root_axis : tv->getRFactorDomain()) {
        setRootAxisInfo(rf_root_axis, AxisHaloInfo());
      }
    }
  }

  // Propagate backward halo information of root axes from fusion
  // outputs to inputs
  auto exprs = fusion->exprs();
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    auto expr = *it;
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    propagateRootAxisInfo(expr);
  }

  // Propagates halo information from root axes down to leaf axes
  for (auto tv : tvs) {
    build(tv->domain());
  }

  if (isDebugDumpEnabled(DebugDumpOption::Halo)) {
    std::cout << toString() << std::endl;
  }

  // Note that validation requires consumer halo info
  for (auto tv : tvs) {
    validate(tv, ca_map);
  }
}

void HaloInfo::propagateRootAxisInfo(Expr* expr) {
  for (auto output : expr->outputs()) {
    auto out_tv = dynamic_cast<TensorView*>(output);
    if (out_tv == nullptr) {
      continue;
    }
    for (auto input : expr->inputs()) {
      auto in_tv = dynamic_cast<TensorView*>(input);
      if (in_tv == nullptr) {
        continue;
      }
      propagateRootAxisInfo(in_tv, out_tv, expr);
    }
  }
}

void HaloInfo::propagateRootAxisInfo(
    TensorView* producer,
    TensorView* consumer,
    Expr* expr) {
  // Do not add halo to input tensors
  if (producer->isFusionInput()) {
    return;
  }

  auto c2p = PairwiseRootDomainMap(producer, consumer)
                 .mapConsumerToProducer(consumer->domain(), producer->domain());

  const auto& c_root = consumer->getRootDomain();

  for (const auto i : c10::irange(c_root.size())) {
    auto c_id = c_root[i];
    auto it = c2p.find(c_id);
    if (it == c2p.end()) {
      // nothing to propagate
      continue;
    }

    // propagate root-axis halo info from c_id to p_id

    auto p_id = it->second;

    AxisHaloInfo p_info;
    if (hasRootAxisInfo(p_id)) {
      p_info = getRootAxisInfo(p_id);
    }
    const auto c_info = getRootAxisInfo(c_id);

    // If the root axes are broadcast, no halo should be associated
    // with them.
    if (c_id->isBroadcast()) {
      TORCH_INTERNAL_ASSERT(!c_info.hasHalo());
      p_info.merge(c_info);
      setRootAxisInfo(p_id, p_info);
      continue;
    } else if (p_id->isRFactorProduct()) {
      TORCH_INTERNAL_ASSERT(
          !c_info.hasHalo(),
          "Propagating halo info to a rfactor producer domain not yet supported.");
      continue;
    }

    // If the defining expression is shift, adjust the producer halo
    // width based on the shift offset. If the shift offset is
    // positive, create halo at offset zero of the producer axis so
    // that the consumer can safely access the producer. If the offset
    // is negative, halo is created at the other end of the axis.
    // If the expr is not shift, just merge the consumer halo info
    // to the producer halo info so that the producer halo can be the
    // maximum of all its consumers.
    if (auto shift_op = dynamic_cast<ShiftOp*>(expr)) {
      const auto offset = shift_op->offset(i);
      if (offset == 0) {
        p_info.merge(c_info);
      } else {
        int pos = (offset > 0) ? 0 : 1;
        p_info.merge(pos, c_info.width(pos) + std::abs(offset));
      }
    } else if (auto gather_op = dynamic_cast<GatherOp*>(expr)) {
      const auto window_dim = gather_op->windowShape()[i];
      if (window_dim == 1) {
        p_info.merge(c_info);
        continue;
      }
      const auto pad_dim0 = gather_op->padWidth()[i][0];
      p_info.merge(0, c_info.width(0) + pad_dim0);
      // The right-side halo is propagated as:
      //   consumer_right_halo + (window_dim - 1 - left_padding)
      p_info.merge(1, c_info.width(1) + window_dim - 1 - pad_dim0);
    } else {
      p_info.merge(c_info);
    }
    setRootAxisInfo(p_id, p_info);
  }
}

void HaloInfo::insertToInheritanceMap(
    TensorDomain* td,
    IterDomain* parent,
    IterDomain* child) {
  // Check each root domain to see if its set includes the parent. If
  // so, adds the child to the same set.
  bool inserted = false;
  for (auto root_axis : td->getRootDomain()) {
    auto it = inheritance_map_.find(root_axis);
    if (it == inheritance_map_.end()) {
      continue;
    }
    auto& id_set = it->second;
    if (id_set.find(parent) != id_set.end()) {
      id_set.insert(child);
      inserted = true;
    }
  }
  // No matching set found. This should not happen.
  TORCH_INTERNAL_ASSERT(inserted);
}

void HaloInfo::initializeFromRootAxisInfo(IterDomain* id) {
  TORCH_INTERNAL_ASSERT(hasRootAxisInfo(id));

  const auto& halo_info = getRootAxisInfo(id);
  auto halo_width = halo_info.width();

  if (!halo_info.hasHalo()) {
    setHaloWidth(id, 0);
    return;
  }

  auto expanded_extent =
      IrBuilder::addExpr(id->extent(), IrBuilder::create<Int>(halo_width));
  extent_map_[id] = expanded_extent;
  halo_width_map_[id] = halo_width;

  inheritance_map_[id] = {id};
}

void HaloInfo::setHaloWidth(IterDomain* id, int halo_width) {
  halo_width_map_[id] = halo_width;
}

// Propagate extent information from root axes to descendants
void HaloInfo::build(TensorDomain* td) {
  auto exprs = DependencyCheck::getAllExprsBetween(
      {td->getMaybeRFactorDomain().begin(), td->getMaybeRFactorDomain().end()},
      {td->domain().begin(), td->domain().end()});

  // Track IDs that are generated by merging halo-extended IDs
  std::unordered_set<IterDomain*> merged_shifted_ids;

  // Propagate halo information by traversing IterDomain
  // expressions. We populate extent_map_ and
  // halo_width_map_.
  // - extent_map_ maps to Expr* representing the
  // extent of each axis including its halo. If no mapping exists for
  // a particular axis in extent_map_, it means the axis does not have
  // halo.
  // - halo_width_map_ just maps to the integer size of the halo,
  // which is used for extent comparison (e.g., extentLessEqual).
  //
  // - When expr is split: if the halo width of the input axis is
  // zero, both the split outputs get zero halo in halo_width_map_. No
  // mapping is added for extent_map_. Otherwise, the halo is
  // propagated only to the inner output, so the inner output gets the
  // same halo width and its mapping is created in extent_map_.
  //
  // One major assumption here is that splitting an axis that is
  // an output of merging halo-extended axes is not allowed. This is
  // because it is unclear how to split the halo part of the merged
  // axis. This is unlikely to be a real limitation in practice.
  //
  // - When expr is merge: if either of the inputs has halo, a mapping
  // for the output is created in extent_map_. No mapping is created
  // for halo_width_map_ (see the comment on HaloInfo::halo_width_map_
  // in lower_shift.h). If both of them don't have halo, just adds a
  // new mapping of the output to zero in halo_width_map_. Also adds
  // it to a set (merged_shifted_ids) to track which axes are merge
  // outputs of halo-extended axes.

  for (auto expr : exprs) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      // Merge-then-split of halo-extended IDs is not allowed
      TORCH_INTERNAL_ASSERT(
          merged_shifted_ids.find(split->in()) == merged_shifted_ids.end(),
          "Splitting IterDomain that is a merged domain of halo-extended domains is not allowed");

      auto in_id = split->in();

      // If no halo info is found, nothing needs to be done. This ID
      // must be an ancestor of a domain set by setRootAxisInfo.
      if (!hasHaloWidth(in_id)) {
        continue;
      }

      const auto halo_width = getHaloWidth(in_id);

      if (halo_width == 0) {
        setHaloWidth(split->outer(), 0);
        setHaloWidth(split->inner(), 0);
        continue;
      }

      // propagate to inner domain
      auto out_id = split->inner();

      auto expanded_extent =
          SimplifyingIrBuilder::addExpr(out_id->extent(), halo_width);
      extent_map_.insert({out_id, expanded_extent});

      setHaloWidth(split->outer(), 0);
      setHaloWidth(split->inner(), halo_width);

      insertToInheritanceMap(td, in_id, split->inner());
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      // If either of the two inputs has halo extension, propagate it
      // to the merged output ID
      auto inner_extent = getExtent(merge->inner());
      auto outer_extent = getExtent(merge->outer());
      if (inner_extent != nullptr || outer_extent != nullptr) {
        if (inner_extent == nullptr) {
          inner_extent = merge->inner()->extent();
        } else {
          insertToInheritanceMap(td, merge->inner(), merge->out());
        }
        if (outer_extent == nullptr) {
          outer_extent = merge->outer()->extent();
        } else {
          insertToInheritanceMap(td, merge->outer(), merge->out());
        }
        auto expanded_extent =
            SimplifyingIrBuilder::mulExpr(outer_extent, inner_extent);
        extent_map_.insert({merge->out(), expanded_extent});
        // Splitting the output of this merge is not allowed, so
        // remember it
        merged_shifted_ids.insert(merge->out());
        // Note that halo_width_map_ is not updated
      } else {
        setHaloWidth(merge->out(), 0);
      }
    } else if (auto swizzle = dynamic_cast<Swizzle2D*>(expr)) {
      // Assume no halo on swizzled domain for now.
      TORCH_INTERNAL_ASSERT(
          getExtent(swizzle->inX()) == nullptr,
          "Halo is not supported with swizzle. Halo-extended ID: ",
          swizzle->inX()->toString(),
          " used in ",
          swizzle->toString());
      TORCH_INTERNAL_ASSERT(
          getExtent(swizzle->inY()) == nullptr,
          "Halo is not supported with swizzle. Halo-extended ID: ",
          swizzle->inY()->toString(),
          " used in ",
          swizzle->toString());
      for (auto id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
        setHaloWidth(id, 0);
      }
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported expr: ", expr);
    }
  }
}

//! Restriction 1: When allocation is outside of a shifted
//! axis, the shifted axis must be guaranteed to have a smaller extent
//! than the concrete axis. For now, shifted axes always mean expanded
//! allocations when the axis is located inside the allocation
//! point. This restriction is validated at the allocation lowering
//! pass.
//!
//! Restriction 2: If an expanded axis is parallelized, its memory
//! must be accessible by all other threads. More specifically:
//! - TIDx: It must be on shared memory. May want to consider
//! utilizing the shuffle instructions as well.
//! - BIDx: Not supported. If on global memory, Cooperative Launch
//! may be used to support it, however, it's unclear in what
//! situations block-level parallelization should be used.
//!
//! Other types of parallelization should be supported except for
//! vectorization. Vectorization should be eventually supported but
//! needs further work.
void HaloInfo::validate(
    TensorView* tv,
    std::shared_ptr<const ComputeAtMap> ca_map) const {
  const auto mem_type = tv->getMemoryType();

  for (auto axis : tv->domain()->domain()) {
    auto concrete_id = ca_map->getConcreteMappedID(axis, IdMappingMode::LOOP);

    // The extent is assumed to be the same
    TORCH_INTERNAL_ASSERT(
        extentEqual(axis, concrete_id),
        "Axis does not have the same exact size with its concrete ID due to halo extension.",
        " Tensor: T",
        tv->name(),
        ", Axis: ",
        axis,
        ", concrete ID: ",
        concrete_id);

    auto halo_extent = getExtent(axis);

    // If no halo extent is associated with this axis, it means the
    // axis is not extended.
    if (halo_extent == nullptr) {
      continue;
    }

    // Enforce restrictions on parallelization and memory type
    const auto ptype = concrete_id->getParallelType();

    if (ptype == ParallelType::Serial) {
      continue;
    }

    // Only threading parallelism is considered for now
    TORCH_CHECK(
        isParallelTypeThread(ptype), "Unsupported parallel type: ", ptype);

    bool shared_mem_needed = false;
    for (auto use : tv->uses()) {
      if (!ir_utils::isTvOp(use)) {
        continue;
      }
      if (use->isA<ShiftOp>() || use->isA<GatherOp>()) {
        shared_mem_needed = true;
        break;
      }
      auto consumer = use->outputs()[0]->as<TensorView>();
      // Find the corresponding axis in the consumer
      auto it = std::find_if(
          consumer->domain()->domain().begin(),
          consumer->domain()->domain().end(),
          [&](IterDomain* consumer_axis) {
            return ca_map->areMapped(
                axis, consumer_axis, IdMappingMode::PERMISSIVE);
          });
      if (it == consumer->domain()->domain().end()) {
        continue;
      }
      if (!extentEqual(axis, *it)) {
        shared_mem_needed = true;
        break;
      }
    }

    if (!shared_mem_needed) {
      continue;
    }

    if (isParallelTypeThreadDim(ptype)) {
      // If all the consumers have the same extent and none of the
      // expressions is shift, any memory should be fine. Otherwise, it
      // must be accessible by all threads involved in the
      // parallelization.
      TORCH_CHECK(
          mem_type == MemoryType::Shared,
          "TV",
          tv->name(),
          " must be allocated on shared memory as its halo-extended axis is parallelized by ",
          ptype);

    } else if (isParallelTypeBlockDim(ptype)) {
      TORCH_CHECK(
          false,
          "Block-based parallelization of a halo-extended axis is not supported: ",
          axis);
    }
  }
  return;
}

Val* HaloInfo::getExtent(IterDomain* id) const {
  auto it = extent_map_.find(id);
  if (it != extent_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

int HaloInfo::getHaloWidth(IterDomain* id) const {
  auto it = halo_width_map_.find(id);
  TORCH_INTERNAL_ASSERT(it != halo_width_map_.end());
  return it->second;
}

bool HaloInfo::hasHaloWidth(IterDomain* id) const {
  return halo_width_map_.find(id) != halo_width_map_.end();
}

const std::unordered_set<IterDomain*>& HaloInfo::getChildDomains(
    IterDomain* root_id) const {
  auto it = inheritance_map_.find(root_id);
  TORCH_INTERNAL_ASSERT(
      it != inheritance_map_.end(),
      "Domain not found in the inheritance map: ",
      root_id);
  return it->second;
}

bool HaloInfo::isHaloInherited(IterDomain* root_id, IterDomain* id) const {
  return getChildDomains(root_id).count(id) > 0;
}

std::unordered_set<IterDomain*> HaloInfo::getRootDomains(IterDomain* id) const {
  std::unordered_set<IterDomain*> id_set;

  for (const auto& kv : inheritance_map_) {
    if (kv.second.count(id) > 0) {
      id_set.insert(kv.first);
    }
  }

  return id_set;
}

namespace {

//! Prove if the comparison operator, cmp, is true with the extents of
//! id1 and id2, including their halo. The comparison is done
//! conservatively, meaning false negative is possible.
//!
//! It is assumed that id1 and id2 are mapped with the CA Loop map, so
//! what is checked here is only about halo
//! sizes using HaloInfo::halo_width_map_. Since it does not have
//! mappings for merged axes, each axis of merge inputs are
//! individually compared, and only when both of the input axes
//! return true, the merge output axis returns true.
template <typename Cmp>
bool extentCompare(
    const HaloInfo& halo_map,
    IterDomain* id1,
    IterDomain* id2,
    Cmp cmp,
    const DisjointSets<IterDomain*>& permissive_map) {
  TORCH_INTERNAL_ASSERT(
      permissive_map.strictAreMapped(id1, id2), "Invalid axes to compare");

  // It's invalid to compare two axes and when only either of them has
  // halo.

  if (halo_map.hasHaloWidth(id1)) {
    TORCH_INTERNAL_ASSERT(
        halo_map.hasHaloWidth(id2), "Invalid comparison: ", id1, " and ", id2);
    // Both axes have halo. We assume the axes themselves have equal
    // extents, excluding halo, as they are mapped with the CA
    // map. So, we just need to compare the halo width of each axis.
    return cmp(halo_map.getHaloWidth(id1), halo_map.getHaloWidth(id2));
  } else {
    TORCH_INTERNAL_ASSERT(!halo_map.hasHaloWidth(id2));
    // Both don't have halo. The only case this can happen must be
    // both axes are the output of a merge expression, so each merge
    // input is recursively compared, and returns true only when both
    // inputs return.
    if (auto merge1 = dynamic_cast<Merge*>(id1->definition())) {
      auto merge2 = dynamic_cast<Merge*>(id2->definition());
      TORCH_INTERNAL_ASSERT(
          merge2 != nullptr, "Invalid comparison: ", id1, " and ", id2);
      auto inner_le = extentCompare(
          halo_map, merge1->inner(), merge2->inner(), cmp, permissive_map);
      auto outer_le = extentCompare(
          halo_map, merge1->outer(), merge2->outer(), cmp, permissive_map);
      return inner_le && outer_le;
    } else {
      // This is not considered. Should never reach here.
      TORCH_INTERNAL_ASSERT(false, "Invalid comparison: ", id1, " and ", id2);
    }
  }
}

} // namespace

bool HaloInfo::extentLessEqual(IterDomain* id1, IterDomain* id2) const {
  return extentCompare(*this, id1, id2, std::less_equal<>(), permissive_map_);
}

bool HaloInfo::extentEqual(IterDomain* id1, IterDomain* id2) const {
  return extentCompare(*this, id1, id2, std::equal_to<>(), permissive_map_);
}

std::string HaloInfo::toString() const {
  std::stringstream ss;

  ss << "HaloInfo:\n";

  if (root_axis_map_.empty()) {
    return ss.str();
  }

  Fusion* fusion = root_axis_map_.begin()->first->fusion();

  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    const auto& root = tv->getRootDomain();
    ss << "TV" << tv->name() << " root domain: ";
    for (auto axis : root) {
      ss << axis << " -> " << getRootAxisInfo(axis).toString() << ", ";
    }
    ss << "\n";
  }

  return ss.str();
}

bool HaloInfo::needsShiftPredicate(Expr* expr) const {
  // In lowering shift and gather turn into a unary op. We really need the shift
  // expr. Do a round about trick to grab it:
  auto tv_out = ir_utils::getTvOutput(expr);
  auto consumer_td = tv_out->domain();
  auto shift_expr = dynamic_cast<ShiftOp*>(tv_out->definition());
  auto gather_expr = dynamic_cast<GatherOp*>(tv_out->definition());
  for (const auto i : c10::irange(consumer_td->getRootDomain().size())) {
    auto consumer_id = consumer_td->getRootDomain()[i];
    const auto consumer_halo_info = getRootAxisInfo(consumer_id);
    if (consumer_halo_info.hasHalo() ||
        (shift_expr != nullptr && shift_expr->offset(i) != 0 &&
         !consumer_id->isBroadcast()) ||
        (gather_expr != nullptr && gather_expr->windowShape()[i] != 1 &&
         !consumer_id->isBroadcast())) {
      return true;
    }
  }
  return false;
}

std::unordered_map<IterDomain*, Val*> HaloInfo::buildConcreteHaloExtentMap(
    const LoopIndexing& loop_indexing) const {
  // Use a local workspace to avoid re-defining halo info.
  HaloInfo local_halo_info = *GpuLower::current()->haloInfo();

  auto global_halo_info = GpuLower::current()->haloInfo();

  // Setup root:
  for (auto consumer_root_id : loop_indexing.consumerTv()->getRootDomain()) {
    auto consumer_index_concrete_id =
        GpuLower::current()->caMap()->getConcreteMappedID(
            consumer_root_id, IdMappingMode::EXACT);
    local_halo_info.setRootAxisInfo(
        consumer_index_concrete_id,
        global_halo_info->getRootAxisInfo(consumer_root_id));
  }

  // Track IDs that are generated by merging halo-extended IDs
  std::unordered_set<IterDomain*> merged_shifted_ids;

  for (auto expr : loop_indexing.getForwardExprList()) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      // Merge-then-split of halo-extended IDs is not allowed
      TORCH_INTERNAL_ASSERT(
          merged_shifted_ids.find(split->in()) == merged_shifted_ids.end(),
          "Splitting IterDomain that is a merged domain of halo-extended domains is not allowed");

      auto in_id = GpuLower::current()->caMap()->getConcreteMappedID(
          split->in(), IdMappingMode::EXACT);

      // If no halo info is found, nothing needs to be done. This ID
      // must be an ancestor of a domain set by setRootAxisInfo.
      if (!local_halo_info.hasHaloWidth(in_id)) {
        continue;
      }

      const auto halo_width = local_halo_info.getHaloWidth(in_id);

      if (halo_width == 0) {
        local_halo_info.setHaloWidth(
            GpuLower::current()->caMap()->getConcreteMappedID(
                split->outer(), IdMappingMode::EXACT),
            0);
        local_halo_info.setHaloWidth(
            GpuLower::current()->caMap()->getConcreteMappedID(
                split->inner(), IdMappingMode::EXACT),
            0);
        continue;
      }

      // propagate to inner domain
      auto out_id = GpuLower::current()->caMap()->getConcreteMappedID(
          split->inner(), IdMappingMode::EXACT);

      auto expanded_extent =
          SimplifyingIrBuilder::addExpr(out_id->extent(), halo_width);
      local_halo_info.extent_map_.insert({out_id, expanded_extent});

      local_halo_info.setHaloWidth(
          GpuLower::current()->caMap()->getConcreteMappedID(
              split->outer(), IdMappingMode::EXACT),
          0);
      local_halo_info.setHaloWidth(
          GpuLower::current()->caMap()->getConcreteMappedID(
              split->inner(), IdMappingMode::EXACT),
          halo_width);

      // TODO: add support for inheritance map
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      // If either of the two inputs has halo extension, propagate it
      // to the merged output ID
      auto inner_extent = local_halo_info.getExtent(
          GpuLower::current()->caMap()->getConcreteMappedID(
              merge->inner(), IdMappingMode::EXACT));
      auto outer_extent = local_halo_info.getExtent(
          GpuLower::current()->caMap()->getConcreteMappedID(
              merge->outer(), IdMappingMode::EXACT));
      if (inner_extent != nullptr || outer_extent != nullptr) {
        if (inner_extent == nullptr) {
          inner_extent = merge->inner()->extent();
        }
        if (outer_extent == nullptr) {
          outer_extent = merge->outer()->extent();
        }
        auto expanded_extent =
            SimplifyingIrBuilder::mulExpr(outer_extent, inner_extent);
        local_halo_info.extent_map_.insert(
            {GpuLower::current()->caMap()->getConcreteMappedID(
                 merge->out(), IdMappingMode::EXACT),
             expanded_extent});
        // Splitting the output of this merge is not allowed, so
        // remember it
        merged_shifted_ids.insert(
            GpuLower::current()->caMap()->getConcreteMappedID(
                merge->out(), IdMappingMode::EXACT));
        // Note that halo_width_map_ is not updated
      } else {
        local_halo_info.setHaloWidth(
            GpuLower::current()->caMap()->getConcreteMappedID(
                merge->out(), IdMappingMode::EXACT),
            0);
      }
    } else if (auto swizzle_2d = dynamic_cast<Swizzle2D*>(expr)) {
      // Swizzle with halo not yet supported, just set the width
      //  to zero at the moment.
      TORCH_INTERNAL_ASSERT(
          local_halo_info.getHaloWidth(
              GpuLower::current()->caMap()->getConcreteMappedID(
                  swizzle_2d->inX(), IdMappingMode::EXACT)) == 0 &&
              local_halo_info.getHaloWidth(
                  GpuLower::current()->caMap()->getConcreteMappedID(
                      swizzle_2d->inY(), IdMappingMode::EXACT)) == 0,
          "Swizzle on ID with halo not yet supported.");
      TORCH_INTERNAL_ASSERT("Swizzle on ID with halo not yet supported.");
      local_halo_info.setHaloWidth(
          GpuLower::current()->caMap()->getConcreteMappedID(
              swizzle_2d->outX(), IdMappingMode::EXACT),
          0);
      local_halo_info.setHaloWidth(
          GpuLower::current()->caMap()->getConcreteMappedID(
              swizzle_2d->outY(), IdMappingMode::EXACT),
          0);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported expr: ", expr);
    }
  }

  return local_halo_info.extent_map_;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
