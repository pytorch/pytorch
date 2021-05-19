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

namespace {

// utility function
kir::Bool* makeAndExpr(kir::Val* lhs, kir::Val* rhs) {
  TORCH_INTERNAL_ASSERT(!(lhs == nullptr && rhs == nullptr));
  if (lhs == nullptr) {
    return rhs->as<kir::Bool>();
  } else if (rhs == nullptr) {
    return lhs->as<kir::Bool>();
  } else {
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    return ir_builder.andExpr(lhs, rhs)->as<kir::Bool>();
  }
}

// utility function
kir::Val* makeAddExpr(kir::Val* lhs, int rhs) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr);
  if (rhs == 0) {
    return lhs;
  } else if (rhs > 0) {
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    return ir_builder.addExpr(lhs, ir_builder.create<kir::Int>(rhs));
    return lhs;
  } else {
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    return ir_builder.subExpr(lhs, ir_builder.create<kir::Int>(-rhs));
  }
}

} // namespace

void ShiftPredicateInserter::insert(
    kir::Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::Bool* thread_pred) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // thread predication is not supported yet
  TORCH_INTERNAL_ASSERT(
      thread_pred->isConst() && thread_pred->value().value(),
      "Thread predication is not supported for expressions with halo-extended outputs");

  kir::TensorView* out_tv = nullptr;
  for (auto out : expr->outputs()) {
    if (out->isA<kir::TensorView>()) {
      out_tv = out->as<kir::TensorView>();
    }
  }
  TORCH_INTERNAL_ASSERT(out_tv != nullptr, "Missing kir::TensorView output");

  const auto predicates = getPredicate(expr, loops, out_tv);
  const auto shift_pred = predicates[0];
  const auto padding_pred = predicates[1];

  // If null, no specific predicate is needed.
  if (shift_pred == nullptr) {
    TORCH_INTERNAL_ASSERT(
        padding_pred == nullptr,
        "Invalid combination of shift_pred and padding_pred.",
        " shift_pred is nullptr, but padding_pred is not.");
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

  auto shift_ite = ir_builder.create<kir::IfThenElse>(shift_pred);

  auto& scope = loops.back()->body();

  // Insert the if statement
  scope.insert_before(expr, shift_ite);

  // Remove the expr from the list
  scope.erase(expr);

  // Place the expr inside the if statement
  shift_ite->thenBody().push_back(expr);

  // Pading by zero
  auto bounds_ite = ir_builder.create<kir::IfThenElse>(padding_pred);
  const int pad_value = 0;
  auto pad_expr = ir_builder.create<kir::UnaryOp>(
      UnaryOpType::Set, out_tv, ir_builder.create<kir::Int>(pad_value));
  bounds_ite->thenBody().push_back(pad_expr);
  // Insert the else block
  shift_ite->elseBody().push_back(bounds_ite);
}

std::array<kir::Bool*, 2> ShiftPredicateInserter::getPredicate(
    const kir::Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::TensorView* out_tv) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  TensorView* out_fuser_tv = out_tv->fuserTv();

  const bool needs_shift_predicate =
      gpu_lower->haloInfo().needsShiftPredicate(out_fuser_tv->definition());

  if (!needs_shift_predicate) {
    return {nullptr, nullptr};
  }

  const auto& root_domain = out_fuser_tv->getRootDomain();

  auto shift_expr = dynamic_cast<ShiftOp*>(out_fuser_tv->definition());

  // Creates indices at the root domain.
  // Set contiguity of all axes false as separate indices are needed for each
  // root axis.
  // Note: separate indices should be needed only for axes that
  // require shift predication, so other axes could use the actual
  // contiguity information. See a TODO item of issue #877.
  const auto pred_contiguity = std::vector<bool>(root_domain.size(), false);
  auto indices =
      Index::getConsumerRootPredIndices(out_tv, loops, pred_contiguity).first;
  TORCH_INTERNAL_ASSERT(indices.size() == root_domain.size());

  kir::Bool* shift_pred = nullptr;
  kir::Bool* padding_pred = nullptr;

  for (size_t i = 0; i < root_domain.size(); ++i) {
    auto root_id = root_domain[i];

    const auto halo_info = gpu_lower->haloInfo().getRootAxisInfo(root_id);

    const int shift_offset =
        (shift_expr != nullptr) ? shift_expr->offset(i) : 0;

    // "left" means halo at offset zero.
    // shifted accesses when idx >= left_limit. padding if idx <
    // left_limit.

    // The elements at the left halo region are just set by the
    // padding value.
    unsigned left_limit = halo_info.width(0);

    // If the defining expr is ShiftOp and its offset is positive,
    // consumer access at 0 to the offset corresponds to
    // out-of-bound producer access unless the producer has halo as
    // well. For now, always add predication assuming no halo on the
    // producer. This should be reivisted for performance
    // optimization (#877).
    if (shift_offset > 0) {
      left_limit += (unsigned)shift_offset;
    }

    // any access < left_limit must be just padding
    if (left_limit > 0) {
      shift_pred = makeAndExpr(
          shift_pred,
          ir_builder.geExpr(
              indices[i], ir_builder.create<kir::Int>(left_limit)));
    }

    auto shift_max_offset = makeAddExpr(
        out_tv->domain()->rootDomain()[i]->extent(), halo_info.width(0));

    // If the shift offset is negative, the maximum index is extent -
    // abs(shift_offset). Instead of subtracting shift_offset from
    // extent, which can result in wrap around, add the absolute value
    // of the shift offset to the index
    auto shift_max_pred_idx = indices[i];
    if (shift_offset < 0) {
      shift_max_pred_idx = makeAddExpr(shift_max_pred_idx, -shift_offset);
    }

    shift_pred = makeAndExpr(
        shift_pred, ir_builder.ltExpr(shift_max_pred_idx, shift_max_offset));

    auto padding_max_offset = makeAddExpr(
        out_tv->domain()->rootDomain()[i]->extent(), halo_info.width());

    padding_pred = makeAndExpr(
        padding_pred, ir_builder.ltExpr(indices[i], padding_max_offset));
  }

  return {shift_pred, padding_pred};
}

const AxisHaloInfo& HaloInfo::getRootAxisInfo(IterDomain* id) const {
  TORCH_INTERNAL_ASSERT(
      id->definition() == nullptr || id->isRFactorProduct(),
      "Invalid IterDomain: ",
      id);
  auto it = root_axis_map_.find(id);
  TORCH_INTERNAL_ASSERT(
      it != root_axis_map_.end(), "Halo root axis info not found for ", id);
  return it->second;
}

AxisHaloInfo& HaloInfo::getRootAxisInfo(IterDomain* id) {
  return const_cast<AxisHaloInfo&>(
      const_cast<const HaloInfo*>(this)->getRootAxisInfo(id));
}

const AxisHaloInfo& HaloInfo::getRootAxisInfo(kir::IterDomain* id) const {
  TORCH_INTERNAL_ASSERT(
      id->definition() == nullptr || id->isRFactorProduct(),
      "Invalid IterDomain: ",
      id);
  auto it = kir_root_axis_map_.find(id);
  TORCH_INTERNAL_ASSERT(
      it != kir_root_axis_map_.end(), "Halo root axis info not found for ", id);
  return it->second;
}

AxisHaloInfo& HaloInfo::getRootAxisInfo(kir::IterDomain* id) {
  return const_cast<AxisHaloInfo&>(
      const_cast<const HaloInfo*>(this)->getRootAxisInfo(id));
}

void HaloInfo::setRootAxisInfo(
    IterDomain* id,
    const AxisHaloInfo& root_axis_info) {
  TORCH_INTERNAL_ASSERT(
      id->definition() == nullptr || id->isRFactorProduct(),
      "Invalid IterDomain: ",
      id);
  root_axis_map_[id] = root_axis_info;
  kir_root_axis_map_
      [GpuLower::current()->lowerValue(id)->as<kir::IterDomain>()] =
          root_axis_info;
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

  for (size_t i = 0; i < c_root.size(); ++i) {
    auto c_id = c_root[i];
    auto it = c2p.find(c_id);
    if (it == c2p.end()) {
      // nothing to propagate
      continue;
    }

    // propagate root-axis halo info from c_id to p_id

    auto p_id = it->second;

    auto p_info = getRootAxisInfo(p_id);
    const auto c_info = getRootAxisInfo(c_id);

    // If the defining expression is shift, adjust the producer halo
    // width based on the shift offset. If the shift offset is
    // positive, create halo at offset zero of the producer axis so
    // that the consumer can safely access the producer. If the offset
    // is negative, halo is created at the other end of the axis.
    // If the expr is not shift, just merge the consumer halo info
    // to the producer halo info so that the producer halo can be the
    // maximum of all its consumers.
    if (auto shift_op = dynamic_cast<ShiftOp*>(expr)) {
      const int offset = shift_op->offset(i);
      if (offset == 0) {
        p_info.merge(c_info);
      } else {
        int pos = (offset > 0) ? 0 : 1;
        p_info.merge(pos, c_info.width(pos) + std::abs(offset));
      }
    } else {
      p_info.merge(c_info);
    }
    setRootAxisInfo(p_id, p_info);
  }
}

// Propagate extent information from root axes to descendants
void HaloInfo::build(TensorDomain* td) {
  auto gpu_lower = GpuLower::current();

  for (auto root_axis : td->getRootDomain()) {
    const auto& halo_info = getRootAxisInfo(root_axis);
    auto halo_width = halo_info.width();

    // There should be no existing mapping. Note that at one point it
    // wasn't the case as root axes were reused when creating
    // reference tensors.
    // TODO: This is not the case actually. Root domains are reused
    // when creating some TensorDomains, so a single IterDomain can
    // show up multiple times. That itself should be fixed, but for
    // now disable this assertion.
    // TORCH_INTERNAL_ASSERT(
    // halo_width_map_.find(root_axis) == halo_width_map_.end(),
    // "Invalid domain: ", root_axis, " of ", td->getRootDomain());

    if (halo_width == 0) {
      halo_width_map_.insert({root_axis, 0});
      continue;
    }

    auto expanded_extent = add(root_axis->extent(), new Int(halo_width));
    extent_map_.insert({root_axis, expanded_extent});
    kir_extent_map_.insert(
        {gpu_lower->lowerValue(root_axis)->as<kir::IterDomain>(),
         gpu_lower->lowerValue(expanded_extent)});
    halo_width_map_.insert({root_axis, halo_width});
  }

  auto exprs = ExprSort::getExprs(
      td->fusion(),
      std::vector<Val*>(td->domain().begin(), td->domain().end()));

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

      // There must be always a mapping for the input axis of a split
      // expr. The only exception is when the input axis is an output
      // of merge, but that's excluded by the assertion above.
      const auto& halo_width_it = halo_width_map_.find(in_id);
      TORCH_INTERNAL_ASSERT(halo_width_it != halo_width_map_.end());

      const auto halo_width = halo_width_it->second;

      if (halo_width == 0) {
        halo_width_map_.insert({split->outer(), 0});
        halo_width_map_.insert({split->inner(), 0});
        continue;
      }

      // propagate to inner domain
      auto out_id = split->inner();

      auto expanded_extent = add(out_id->extent(), new Int(halo_width));
      extent_map_.insert({out_id, expanded_extent});
      kir_extent_map_.insert(
          {gpu_lower->lowerValue(out_id)->as<kir::IterDomain>(),
           gpu_lower->lowerValue(expanded_extent)});

      halo_width_map_.insert({split->outer(), 0});
      halo_width_map_.insert({split->inner(), halo_width});
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      // If either of the two inputs has halo extension, propagate it
      // to the merged output ID
      if (extent_map_.find(merge->inner()) != extent_map_.end() ||
          extent_map_.find(merge->outer()) != extent_map_.end()) {
        auto inner_extent = getExtent(merge->inner());
        if (inner_extent == nullptr) {
          inner_extent = merge->inner()->extent();
        }
        auto outer_extent = getExtent(merge->outer());
        if (outer_extent == nullptr) {
          outer_extent = merge->outer()->extent();
        }
        auto expanded_extent = mul(outer_extent, inner_extent);
        extent_map_.insert({merge->out(), expanded_extent});
        kir_extent_map_.insert(
            {gpu_lower->lowerValue(merge->out())->as<kir::IterDomain>(),
             gpu_lower->lowerValue(expanded_extent)});
        // Splitting the output of this merge is not allowed, so
        // remember it
        merged_shifted_ids.insert(merge->out());
        // Note that halo_width_map_ is not updated
      } else {
        halo_width_map_.insert({merge->out(), 0});
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
      if (use->isA<ShiftOp>()) {
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

Val* HaloInfo::getExtent(IterDomain* id) const {
  auto it = extent_map_.find(id);
  if (it != extent_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

kir::Val* HaloInfo::getExtent(kir::IterDomain* id) const {
  auto it = kir_extent_map_.find(id);
  if (it != kir_extent_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

unsigned HaloInfo::getHaloWidth(IterDomain* id) const {
  auto it = halo_width_map_.find(id);
  TORCH_INTERNAL_ASSERT(it != halo_width_map_.end());
  return it->second;
}

bool HaloInfo::hasHaloWidth(IterDomain* id) const {
  return halo_width_map_.find(id) != halo_width_map_.end();
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
  return extentCompare(*this, id1, id2, std::less_equal<unsigned>());
}

bool HaloInfo::extentEqual(IterDomain* id1, IterDomain* id2) const {
  return extentCompare(*this, id1, id2, std::equal_to<unsigned>());
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
    ss << "TV" << tv->name() << ": ";
    for (auto axis : root) {
      ss << axis << " -> " << getRootAxisInfo(axis).toString() << ", ";
    }
    ss << "\n";
  }

  return ss.str();
}

bool HaloInfo::needsShiftPredicate(Expr* expr) {
  auto consumer_td = ir_utils::getTVOutput(expr)->domain();
  auto shift_expr = dynamic_cast<ShiftOp*>(expr);
  for (size_t i = 0; i < consumer_td->getRootDomain().size(); ++i) {
    auto consumer_id = consumer_td->getRootDomain()[i];
    const auto consumer_halo_info = getRootAxisInfo(consumer_id);
    if (consumer_halo_info.hasHalo() ||
        (shift_expr != nullptr && shift_expr->offset(i) != 0)) {
      return true;
    }
  }
  return false;
}

bool HaloInfo::needsShiftPredicate(kir::Expr* expr) {
  const auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
  // TODO: There can be two definitions for Rfactor tensors.
  auto fuser_expr = out_tv->fuserTv()->definition();
  TORCH_INTERNAL_ASSERT(fuser_expr != nullptr);
  return needsShiftPredicate(fuser_expr);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
