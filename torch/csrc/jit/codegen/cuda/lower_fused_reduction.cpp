#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/codegen/cuda/lower_fused_reduction.h>

#include <algorithm>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! An instance of reduction patterns to fuse
class FusedReductionBroadcastInfo : public PolymorphicBase {
 public:
  FusedReductionBroadcastInfo(ReductionOp* reduction, bool with_broadcast)
      : reductions_({reduction}), with_broadcast_({with_broadcast}) {}

  FusedReductionBroadcastInfo(WelfordOp* welford, bool with_broadcast)
      : reductions_({welford}), with_broadcast_({with_broadcast}) {}

  FusedReductionBroadcastInfo(
      GroupedReductionOp* grouped_rop,
      bool with_broadcast)
      : reductions_({grouped_rop}), with_broadcast_({with_broadcast}) {}

  const std::vector<Expr*>& reductions() const {
    return reductions_;
  }

  const std::vector<bool>& withBroadcast() const {
    return with_broadcast_;
  }

 private:
  // Holds ReductionOp, WelfordOp or GroupedReductionOp.
  std::vector<Expr*> reductions_;
  // True each reduction also broadcasts
  std::vector<bool> with_broadcast_;
};

//! Inspect a fusion to detect eligible sequences of expressions to
//! use the fused reduction kernel
class FusionInspector : private IterVisitor {
 public:
  static std::vector<FusedReductionBroadcastInfo> run(Fusion* fusion) {
    FusionInspector inspector(fusion);
    return inspector.fusion_list_;
  }

 private:
  FusionInspector(Fusion* fusion) {
    traverse(fusion);
  }

  using IterVisitor::handle;

  void handle(ReductionOp* rop) final {
    /// If it's a grid reduction, keep track of tensors that depend on
    /// this reduction.
    // Only consider when out is on register as that is assumed in the
    // fused reduction kernel.
    auto out = ir_utils::getTvOutput(rop);
    if (out->getMemoryType() == MemoryType::Local &&
        out->domain()->hasGridReduction()) {
      reduction_dep_[out].insert(rop);
    }
  }

  void handle(WelfordOp* wop) final {
    /// If it's a grid reduction, keep track of tensors that depend on
    /// this reduction.
    // Only consider when out is on register as that is assumed in the
    // fused reduction kernel.
    auto out = ir_utils::getTvOutput(wop);
    if (out->getMemoryType() == MemoryType::Local &&
        out->domain()->hasGridReduction()) {
      reduction_dep_[out].insert(wop);
    }
  }

  void handle(GroupedReductionOp* grouped_rop) final {
    auto out = ir_utils::getTvOutput(grouped_rop);
    if (out->getMemoryType() == MemoryType::Local &&
        out->domain()->hasGridReduction()) {
      reduction_dep_[out].insert(grouped_rop);
    }
  }

  void handle(Expr* expr) final {
    IterVisitor::handle(expr);
    for (auto in_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      for (auto reduction_op : reduction_dep_[in_tv]) {
        if (fused_exprs_.find(reduction_op) != fused_exprs_.end()) {
          continue;
        }
        for (auto out_tv :
             ir_utils::filterByType<TensorView>(expr->outputs())) {
          reduction_dep_[out_tv].insert(reduction_op);
        }
      }
    }
  }

  // In the case of welford, use the fused broadcast reduction when at
  // least one of the outputs is broadcast.
  void handle(BroadcastOp* bop) final {
    // Detect a pattern where a reduction is followed by a broadcast
    auto bop_out = bop->out()->as<TensorView>();
    auto bop_in = bop->in()->as<TensorView>();

    for (Expr* preceding_expr : reduction_dep_[bop_in]) {
      auto parallel_reduction_axes =
          getReductionParallelTypeStates(preceding_expr);

      // If not matching, propagate the reduction further down to
      // subsequent expressions
      if (!isBroadcastFuseable(bop_out, parallel_reduction_axes)) {
        continue;
      }

      if (fused_exprs_.find(preceding_expr) != fused_exprs_.end()) {
        // Already added to the fusion list. This can happen with
        // welford as there can be multiple broadcast consumer
        // expressions.
        continue;
      }

      if (preceding_expr->isA<ReductionOp>()) {
        fusion_list_.emplace_back(preceding_expr->as<ReductionOp>(), true);
      } else if (preceding_expr->isA<GroupedReductionOp>()) {
        fusion_list_.emplace_back(
            preceding_expr->as<GroupedReductionOp>(), true);
      } else if (preceding_expr->isA<WelfordOp>()) {
        fusion_list_.emplace_back(preceding_expr->as<WelfordOp>(), true);
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Invalid preceding expr: ", preceding_expr->toString());
      }

      fused_exprs_.insert(preceding_expr);
    }
  }

  ParallelTypeBitmap getReductionParallelTypeStates(Expr* expr) {
    ParallelTypeBitmap parallel_reduction_axes;

    for (auto id : ir_utils::getTvOutput(expr)->domain()->domain()) {
      auto pt = id->getParallelType();
      if (id->isReduction() && isParallelTypeThread(pt)) {
        parallel_reduction_axes.set(pt);
      }
    }

    return parallel_reduction_axes;
  }

  // Requires reduction parallel dimensions to exactly match parallel broadcast
  // dimensions
  bool isBroadcastFuseable(
      TensorView* broadcast_out,
      const ParallelTypeBitmap& parallel_reduction_axes) {
    const auto broadcast_parallel_types =
        GpuLower::current()->threadPredMap().getParallelBroadcastDomains(
            broadcast_out);

    // If no parallel broadcast, nothing to fuse
    if (broadcast_parallel_types.none()) {
      return false;
    }

    // Make sure the broadcast parallel types are the types reduced by
    // the preceding reduction op
    for (auto id : broadcast_out->domain()->domain()) {
      auto pt = id->getParallelType();
      if (!isParallelTypeThread(pt)) {
        continue;
      }
      // Parallel broadcast must be included in reduction_states
      if (id->isBroadcast() && broadcast_parallel_types.get(pt)) {
        if (!parallel_reduction_axes.get(pt)) {
          return false;
        }
      }
    }

    return true;
  }

 private:
  //! List of expression sequences to fuse
  std::vector<FusedReductionBroadcastInfo> fusion_list_;
  //! Keep track of fused reduction/welford exprs to avoid duplication
  std::unordered_set<Expr*> fused_exprs_;
  //! Keep track of ReductionOp/WelfordOp expressions that are
  //! (indirectly) input to a tensor
  std::unordered_map<TensorView*, std::unordered_set<Expr*>> reduction_dep_;
};

//! Transform a fusion to use the fused reduction kernel.
class FusionTransformer {
 public:
  static void run(
      Fusion* fusion,
      const std::vector<FusedReductionBroadcastInfo>& fusion_list) {
    FusionTransformer transformer(fusion, fusion_list);
  }

 private:
  FusionTransformer(
      Fusion* fusion,
      const std::vector<FusedReductionBroadcastInfo>& fusion_list)
      : fusion_(fusion), fusion_list_(fusion_list) {
    transform();
  }

  void transform() {
    for (const auto& info : fusion_list_) {
      transform(info);
    }
    // If the thread predicate map is modified, rebuild the
    // map. build() only updates mappings that need to be updated.
    if (thread_pred_map_modified_) {
      GpuLower::current()->threadPredMap().build(fusion_);
    }
  }

  void transform(const FusedReductionBroadcastInfo& info) {
    TORCH_INTERNAL_ASSERT(
        info.reductions().size() == 1, "Horizontal fusion not supported yet");

    for (const auto i : c10::irange(info.reductions().size())) {
      const auto expr = info.reductions().at(i);
      const auto with_broadcast = info.withBroadcast().at(i);
      Expr* fused_expr = nullptr;

      if (auto reduction = dynamic_cast<ReductionOp*>(expr)) {
        TORCH_INTERNAL_ASSERT(!reduction->isAllreduce());

        auto red_op_type = reduction->getReductionOpType();
        auto init = reduction->init();
        auto out = reduction->out();
        auto in = reduction->in();

        fusion_->removeExpr(reduction);

        fused_expr =
            IrBuilder::create<ReductionOp>(red_op_type, init, out, in, true);
      } else if (auto welford = dynamic_cast<WelfordOp*>(expr)) {
        TORCH_INTERNAL_ASSERT(!welford->isAllreduce());

        auto out_avg = welford->outAvg();
        auto out_var = welford->outVar();
        auto out_n = welford->outN();
        auto init_avg = welford->initAvg();
        auto init_var = welford->initVar();
        auto init_n = welford->initN();
        auto in_avg = welford->inAvg();
        auto in_var = welford->inVar();
        auto in_n = welford->inN();

        fusion_->removeExpr(welford);

        fused_expr = IrBuilder::create<WelfordOp>(
            WelfordTriplet{out_avg, out_var, out_n},
            WelfordTriplet{in_avg, in_var, in_n},
            WelfordTriplet{init_avg, init_var, init_n},
            true);
      } else if (auto grouped_rop = dynamic_cast<GroupedReductionOp*>(expr)) {
        TORCH_INTERNAL_ASSERT(!grouped_rop->isAllreduce());

        auto op_types = grouped_rop->getReductionOpTypes();
        auto init_vals = grouped_rop->initVals();
        auto outputs = grouped_rop->outputs();
        auto inputs = grouped_rop->inputs();

        fusion_->removeExpr(grouped_rop);

        fused_expr = IrBuilder::create<GroupedReductionOp>(
            op_types, init_vals, outputs, inputs, true);
      } else {
        TORCH_INTERNAL_ASSERT(false, "Invalid expr: ", expr->toString());
      }

      TORCH_INTERNAL_ASSERT(fused_expr != nullptr);

      // Do not just remove the broadcast but just reset the thread
      // predicate of the broadcast op. Since fusion is applied only
      // when all parallel broadcast domains are to be parallel
      // reduction, all parallel types can be reset.
      if (with_broadcast) {
        // It may be just fine to remove the broadcast expr, but
        // technically speaking that would violate the root domain mapping
        // as broadcast domains would appear in the consumer of the
        // broadcast output tensor without a broadcast expression.
        for (auto reduction_out :
             ir_utils::filterByType<TensorView>(fused_expr->outputs())) {
          for (auto id : reduction_out->domain()->domain()) {
            if (id->isReduction()) {
              GpuLower::current()->fusedReductionInfo().markAsAllreduce(id);
              GpuLower::current()->threadPredMap().markAsUpdated(reduction_out);
              thread_pred_map_modified_ = true;
            }
          }
        }
      }
    }
  }

 private:
  Fusion* fusion_ = nullptr;
  const std::vector<FusedReductionBroadcastInfo>& fusion_list_;
  bool thread_pred_map_modified_ = false;
};

} // namespace

void fuseReductionsAndBroadcasts(Fusion* fusion) {
  auto fusion_list = FusionInspector::run(fusion);
  FusionTransformer::run(fusion, fusion_list);
}

void FusedReductionInfo::markAsAllreduce(IterDomain* id) {
  allreduce_ids_.insert(id);
}

bool FusedReductionInfo::isAllreduce(IterDomain* id) const {
  return allreduce_ids_.find(id) != allreduce_ids_.end();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
