#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <functional>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void ShiftPredicateInserter::insert(
    kir::Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::Bool* thread_pred,
    bool within_unswitch) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  kir::TensorView* out_tv = ir_utils::getTVOutput(expr);
  TORCH_INTERNAL_ASSERT(out_tv != nullptr, "Missing kir::TensorView output");

  TensorView* out_fuser_tv = out_tv->fuserTv();
  const bool needs_shift_predicate =
      gpu_lower->haloInfo().needsShiftPredicate(out_fuser_tv->definition());
  if (!needs_shift_predicate) {
    return;
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
    thread_pred_expr = ir_builder.create<kir::Predicate>(thread_pred);
  }

  kir::Predicate* shift_pred = within_unswitch
      ? thread_pred_expr
      : ir_builder.create<kir::Predicate>(
            PredicateType::Shift, expr, thread_pred);

  // If the expr involves a thread-block barrier, set the predicate of
  // the expr with shift_pred. Since the expr is not shift, the
  // padding is safe to omit.
  if (ir_utils::hasBlockSync(expr, gpu_lower->threadPredMap())) {
    expr->setPredicate(shift_pred);
    return;
  }

  auto shift_ite = ir_builder.create<kir::IfThenElse>(shift_pred);

  auto& scope = loops.back()->body();

  // Insert the if statement
  scope.insert_before(expr, shift_ite);

  // Remove the expr from the list
  scope.erase(expr);

  // Place the expr inside the if statement
  shift_ite->thenBody().push_back(expr);

  // No padding condition is required if this is within unswitch.
  if (within_unswitch) {
    return;
  }

  // Padding by zero
  kir::Predicate* padding_pred = ir_builder.create<kir::Predicate>(
      PredicateType::Padding, expr, thread_pred);
  auto bounds_ite = ir_builder.create<kir::IfThenElse>(padding_pred);
  const int pad_value = 0;
  auto pad_expr = ir_builder.create<kir::UnaryOp>(
      UnaryOpType::Set, out_tv, ir_builder.create<kir::Int>(pad_value));
  bounds_ite->thenBody().push_back(pad_expr);
  // Insert the else block
  shift_ite->elseBody().push_back(bounds_ite);
}

AxisHaloInfo::AxisHaloInfo() {
  auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());
  setWidth(0, ir_builder.zeroVal());
  setWidth(1, ir_builder.zeroVal());
}

kir::Int* AxisHaloInfo::width() const {
  auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());
  return ir_builder.addExpr(width(0), width(1))->as<kir::Int>();
}

kir::Int* AxisHaloInfo::width(int pos) const {
  TORCH_INTERNAL_ASSERT(pos >= 0 && pos < 2);
  TORCH_INTERNAL_ASSERT(widths_[pos] != nullptr);
  return widths_[pos];
}

void AxisHaloInfo::setWidth(int pos, kir::Int* width) {
  TORCH_INTERNAL_ASSERT(pos >= 0 && pos < 2);
  widths_[pos] = width;
}

void AxisHaloInfo::merge(int pos, kir::Int* other) {
  auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());
  auto cur = width(pos);
  kir::Int* new_width = nullptr;
  if (cur->isConst() && other->isConst()) {
    new_width = ir_builder.create<kir::Int>(
        std::max(cur->value().value(), other->value().value()));
  } else if (cur->isZeroInt()) {
    new_width = other;
  } else if (other->isZeroInt()) {
    new_width = cur;
  } else {
    new_width = ir_builder.maxExpr(width(pos), other)->as<kir::Int>();
  }
  setWidth(pos, new_width);
}

void AxisHaloInfo::merge(const AxisHaloInfo& other) {
  for (const auto i : c10::irange(widths_.size())) {
    merge(i, other.width(i));
  }
}

bool AxisHaloInfo::hasHalo() const {
  return std::any_of(
      widths_.begin(), widths_.end(), [](auto w) { return !w->isZeroInt(); });
}

std::string AxisHaloInfo::toString() const {
  std::stringstream ss;
  ss << "<" << kir::toString(width(0)) << ", " << kir::toString(width(1))
     << ">";
  return ss.str();
}

bool HaloInfo::hasRootAxisInfo(IterDomain* id) const {
  return root_axis_map_.find(id) != root_axis_map_.end();
}

bool HaloInfo::hasRootAxisInfo(kir::IterDomain* id) const {
  return kir_root_axis_map_.find(id) != kir_root_axis_map_.end();
}

const AxisHaloInfo& HaloInfo::getRootAxisInfo(IterDomain* id) const {
  auto it = root_axis_map_.find(id);
  TORCH_INTERNAL_ASSERT(
      it != root_axis_map_.end(), "Halo root axis info not found for ", id);
  return it->second;
}

AxisHaloInfo& HaloInfo::getRootAxisInfo(IterDomain* id) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<AxisHaloInfo&>(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<const HaloInfo*>(this)->getRootAxisInfo(id));
}

const AxisHaloInfo& HaloInfo::getRootAxisInfo(kir::IterDomain* id) const {
  TORCH_INTERNAL_ASSERT(
      id->definition() == nullptr || id->isRFactorProduct(),
      "Invalid IterDomain: ",
      id);
  auto it = kir_root_axis_map_.find(id);
  TORCH_INTERNAL_ASSERT(
      it != kir_root_axis_map_.end(),
      "Halo root axis info not found for ",
      kir::toString(id));
  return it->second;
}

AxisHaloInfo& HaloInfo::getRootAxisInfo(kir::IterDomain* id) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<AxisHaloInfo&>(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<const HaloInfo*>(this)->getRootAxisInfo(id));
}

void HaloInfo::setRootAxisInfo(
    IterDomain* id,
    const AxisHaloInfo& root_axis_info) {
  root_axis_map_[id] = root_axis_info;
  kir_root_axis_map_
      [GpuLower::current()->lowerValue(id)->as<kir::IterDomain>()] =
          root_axis_info;

  initializeFromRootAxisInfo(id);
  return;
}

void HaloInfo::build(Fusion* fusion) {
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
    validate(tv);
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

  auto gpu_lower = GpuLower::current();
  kir::SimplifyingIrBuilder ir_builder(gpu_lower->kernel());

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
        p_info.merge(
            pos,
            ir_builder.addExpr(c_info.width(pos), std::abs(offset))
                ->as<kir::Int>());
      }
    } else if (auto gather_op = dynamic_cast<GatherOp*>(expr)) {
      const auto window_dim =
          gpu_lower->lowerValue(gather_op->windowShape()[i]);
      if (window_dim->isOneInt()) {
        p_info.merge(c_info);
        continue;
      }
      const auto& pad_dim = gather_op->padWidth()[i];
      const auto pad_dim0 = gpu_lower->lowerValue(pad_dim[0])->as<kir::Int>();
      p_info.merge(
          0, ir_builder.addExpr(c_info.width(0), pad_dim0)->as<kir::Int>());
      // The right-side halo is propagated as:
      //   consumer_right_halo + (window_dim - 1 - left_padding)
      p_info.merge(
          1,
          ir_builder
              .subExpr(
                  ir_builder.addExpr(c_info.width(1), window_dim),
                  ir_builder.addExpr(pad_dim0, 1))
              ->as<kir::Int>());
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

  auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  const auto& halo_info = getRootAxisInfo(id);
  auto halo_width = halo_info.width();

  if (!halo_info.hasHalo()) {
    halo_width_map_[id] = ir_builder.zeroVal();
    return;
  }

  auto expanded_extent =
      ir_builder.addExpr(gpu_lower->lowerValue(id->extent()), halo_width);
  kir_extent_map_[gpu_lower->lowerValue(id)->as<kir::IterDomain>()] =
      expanded_extent;
  halo_width_map_[id] = halo_width;

  inheritance_map_[id] = {id};
}

// Propagate extent information from root axes to descendants
void HaloInfo::build(TensorDomain* td) {
  auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

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

      const auto& halo_width_it = halo_width_map_.find(in_id);

      // If no halo info is found, nothing needs to be done. This ID
      // must be an ancestor of a domain set by setRootAxisInfo.
      if (halo_width_it == halo_width_map_.end()) {
        continue;
      }

      const auto halo_width = halo_width_it->second;

      if (halo_width->isZeroInt()) {
        halo_width_map_.insert({split->outer(), halo_width});
        halo_width_map_.insert({split->inner(), halo_width});
        continue;
      }

      // propagate to inner domain
      auto out_id = split->inner();

      auto expanded_extent = ir_builder.addExpr(
          gpu_lower->lowerValue(out_id->extent()), halo_width);
      kir_extent_map_.insert(
          {gpu_lower->lowerValue(out_id)->as<kir::IterDomain>(),
           expanded_extent});

      halo_width_map_.insert({split->outer(), ir_builder.zeroVal()});
      halo_width_map_.insert({split->inner(), halo_width});

      insertToInheritanceMap(td, in_id, split->inner());
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      // If either of the two inputs has halo extension, propagate it
      // to the merged output ID
      auto inner_extent = getExtent(merge->inner());
      auto outer_extent = getExtent(merge->outer());
      if (inner_extent != nullptr || outer_extent != nullptr) {
        if (inner_extent == nullptr) {
          inner_extent = gpu_lower->lowerValue(merge->inner()->extent());
        } else {
          insertToInheritanceMap(td, merge->inner(), merge->out());
        }
        if (outer_extent == nullptr) {
          outer_extent = gpu_lower->lowerValue(merge->outer()->extent());
        } else {
          insertToInheritanceMap(td, merge->outer(), merge->out());
        }
        auto expanded_extent = ir_builder.mulExpr(outer_extent, inner_extent);
        kir_extent_map_.insert(
            {gpu_lower->lowerValue(merge->out())->as<kir::IterDomain>(),
             expanded_extent});
        // Splitting the output of this merge is not allowed, so
        // remember it
        merged_shifted_ids.insert(merge->out());
        // Note that halo_width_map_ is not updated
      } else {
        halo_width_map_.insert({merge->out(), ir_builder.zeroVal()});
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
void HaloInfo::validate(TensorView* tv) const {
  const auto& par_map = GpuLower::current()->caParallelMap();
  const auto& loop_map = GpuLower::current()->caLoopMap();
  const auto mem_type = tv->getMemoryType();

  for (auto axis : tv->domain()->domain()) {
    auto concrete_id = par_map.getConcreteMappedID(axis);

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
      if (!ir_utils::isTVOp(use)) {
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
            return loop_map.areMapped(axis, consumer_axis);
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

kir::Val* HaloInfo::getExtent(IterDomain* id) const {
  auto kir_id = GpuLower::current()->lowerValue(id)->as<kir::IterDomain>();
  return getExtent(kir_id);
}

kir::Val* HaloInfo::getExtent(kir::IterDomain* id) const {
  auto it = kir_extent_map_.find(id);
  if (it != kir_extent_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

kir::Int* HaloInfo::getHaloWidth(IterDomain* id) const {
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
    Cmp cmp) {
  auto gpu_lower = GpuLower::current();
  TORCH_INTERNAL_ASSERT(
      gpu_lower->caLoopMap().areMapped(id1, id2), "Invalid axes to compare");

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
      auto inner_le =
          extentCompare(halo_map, merge1->inner(), merge2->inner(), cmp);
      auto outer_le =
          extentCompare(halo_map, merge1->outer(), merge2->outer(), cmp);
      return inner_le && outer_le;
    } else {
      // This is not considered. Should never reach here.
      TORCH_INTERNAL_ASSERT(false, "Invalid comparison: ", id1, " and ", id2);
    }
  }
}

} // namespace

bool HaloInfo::extentLessEqual(IterDomain* id1, IterDomain* id2) const {
  auto cmp = [](kir::Int* x, kir::Int* y) {
    if (x == y) {
      return true;
    }
    auto xv = x->value();
    auto yv = y->value();
    return xv.has_value() && yv.has_value() && xv.value() <= yv.value();
  };
  return extentCompare(*this, id1, id2, cmp);
}

bool HaloInfo::extentEqual(IterDomain* id1, IterDomain* id2) const {
  // Returns true only when x and y are proven to be the same. The
  // analysis is not comprehensive and can prove in rather trivial
  // cases only. Specifically:
  //   - x and y are the same pointers
  //   - Both have static values and they are the same
  //   - Both are defined by the same expression and the inputs are
  //     proven to be equal
  std::function<bool(kir::Int*, kir::Int*)> cmp = [&](kir::Int* x,
                                                      kir::Int* y) {
    if (x == y) {
      return true;
    }

    auto xv = x->value();
    auto yv = y->value();
    if (xv.has_value() && yv.has_value() && xv.value() == yv.value()) {
      return true;
    }

    // Check if both are defined by an expression of the same type. If
    // so, recursively check the input operands.
    auto x_def = x->definition();
    auto y_def = y->definition();
    if (x_def && y_def &&
        ((x_def->isA<kir::UnaryOp>() && y_def->isA<kir::UnaryOp>() &&
          x_def->as<kir::UnaryOp>()->operation() ==
              y_def->as<kir::UnaryOp>()->operation()) ||
         (x_def->isA<kir::BinaryOp>() && y_def->isA<kir::BinaryOp>() &&
          x_def->as<kir::BinaryOp>()->operation() ==
              y_def->as<kir::BinaryOp>()->operation()))) {
      for (const auto i : c10::irange(x_def->inputs().size())) {
        auto x_input = dynamic_cast<kir::Int*>(x_def->inputs()[i]);
        auto y_input = dynamic_cast<kir::Int*>(y_def->inputs()[i]);
        // Both must be kir::Int
        TORCH_INTERNAL_ASSERT(x_input && y_input);
        if (!cmp(x_input, y_input)) {
          return false;
        }
      }
      return true;
    }

    return false;
  };
  return extentCompare(*this, id1, id2, cmp);
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
  auto consumer_td = ir_utils::getTVOutput(expr)->domain();
  auto shift_expr = dynamic_cast<ShiftOp*>(expr);
  auto gather_expr = dynamic_cast<GatherOp*>(expr);
  for (const auto i : c10::irange(consumer_td->getRootDomain().size())) {
    auto consumer_id = consumer_td->getRootDomain()[i];
    const auto consumer_halo_info = getRootAxisInfo(consumer_id);
    if (consumer_halo_info.hasHalo() ||
        (shift_expr != nullptr && shift_expr->offset(i) != 0 &&
         !consumer_id->isBroadcast()) ||
        (gather_expr != nullptr && !gather_expr->windowShape()[i]->isOneInt() &&
         !consumer_id->isBroadcast())) {
      return true;
    }
  }
  return false;
}

bool HaloInfo::needsShiftPredicate(kir::Expr* expr) const {
  const auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
  auto fuser_expr = out_tv->fuserTv()->definition();
  TORCH_INTERNAL_ASSERT(fuser_expr != nullptr);
  return needsShiftPredicate(fuser_expr);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
