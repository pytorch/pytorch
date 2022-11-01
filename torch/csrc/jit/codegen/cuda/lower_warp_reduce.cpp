#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_warp_reduce.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! A helper class for EliminateDeadBroadcastAndAllocate. Eliminate
//! dead Allocate and Broadcast detected by EliminateDeadBroadcastAndAllocate.
class DeadTvEliminator : private kir::ExprMutator {
 public:
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      const std::unordered_set<TensorView*>& dead_tvs) {
    return DeadTvEliminator(exprs, dead_tvs).exprs_;
  }

 private:
  DeadTvEliminator(
      const std::vector<Expr*>& exprs,
      const std::unordered_set<TensorView*>& dead_tvs)
      : dead_tvs_(dead_tvs) {
    traverseAndInsert(exprs);
  }

  using kir::ExprMutator::handle;

  void handle(kir::Allocate* allocate) final {
    if (auto buffer_tv = dynamic_cast<TensorView*>(allocate->buffer())) {
      if (dead_tvs_.count(buffer_tv)) {
        registerRemove(allocate);
      }
    }
  }

  void handle(BroadcastOp* broadcast) final {
    if (auto out_ti = dynamic_cast<kir::TensorIndex*>(broadcast->out())) {
      if (dead_tvs_.count(out_ti->view())) {
        registerRemove(broadcast);
      }
    }
  }

 private:
  const std::unordered_set<TensorView*>& dead_tvs_;
};

//! A simple DCE for eliminating the
//!  parallel broadcasts that has been fused
//!  and their corresponding allocations
class EliminateDeadBroadcastAndAllocate {
 public:
  static std::vector<Expr*> run(const std::vector<Expr*>& exprs) {
    EliminateDeadBroadcastAndAllocate dce(exprs);
    return DeadTvEliminator::run(exprs, dce.dead_tvs_);
  }

 private:
  EliminateDeadBroadcastAndAllocate(const std::vector<Expr*>& exprs) {
    findLiveTvs(exprs);
    findDeadTvs();
  }

  void findLiveTvs(const std::vector<Expr*>& exprs) {
    for (auto expr : exprs) {
      if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
        findLiveTvs(for_loop->body().exprs());
        continue;
      } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
        findLiveTvs(ite->thenBody().exprs());
        findLiveTvs(ite->elseBody().exprs());
        continue;
      }

      if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
        if (allocate->memoryType() == MemoryType::Local) {
          if (auto tv = dynamic_cast<TensorView*>(allocate->buffer())) {
            // We know only tvs that we'd want to consider are broadcast outputs
            if (tv->definition()->isA<BroadcastOp>()) {
              candidate_tv_set_.insert(tv);
            }
          }
        }
      }

      for (auto inp : expr->inputs()) {
        if (auto ti = dynamic_cast<kir::TensorIndex*>(inp)) {
          if (candidate_tv_set_.count(ti->view())) {
            live_tvs_.insert(ti->view());
          }
        }
      }
    }
  }

  void findDeadTvs() {
    for (auto tv : candidate_tv_set_) {
      if (!live_tvs_.count(tv)) {
        dead_tvs_.insert(tv);
      }
    }
  }

 private:
  std::unordered_set<TensorView*> live_tvs_;
  std::unordered_set<TensorView*> dead_tvs_;
  std::unordered_set<TensorView*> candidate_tv_set_;
};

//! A pass to eliminate redundant parallel broadcasts that are consumers
//!  of warp reduction.
//! Detects the following pattern:
//!
//!  For ... (serial)
//!   For ... (serial)
//!     T1[0] = warp_reduce (T0[0])
//!     T2[0] = block_broadcast (T1[0])
//!
//!  The block_broadcast can then be eliminated given that both the warp
//!   reduce and the broadcast are known in compile-time to be parallelized
//!   on a single warp only.
//!
//!  Currently only limited to buffers of size-1 to avoid having to
//!   re-run indexing
//!
//!  This pass operates in 3 phases:
//!   1. FuseBroadcastWithWarpReduce identifies the broadcasts that can
//!    be removed, and generates a replacement map from the broadcast
//!    output to reduction output.
//!
//!   2. ir_utils::replaceInputsInExpr replaces applicable uses of
//!    the broadcast output with the corresponding reduction output.
//!
//!   3. EliminateDeadBroadcastAndAllocate removes the broadcast ops
//!    and corresponding allocations if they're un-used after step 2.
class FuseBroadcastWithWarpReduce : private kir::IrVisitor {
 public:
  static std::vector<Expr*> fuse(const std::vector<Expr*>& exprs) {
    FuseBroadcastWithWarpReduce fuse_broadcast_map(exprs);
    const auto replaced_inputs = ir_utils::replaceInputsInExpr(
        exprs, fuse_broadcast_map.val_replacement_map_);
    return EliminateDeadBroadcastAndAllocate::run(replaced_inputs);
  }

 private:
  FuseBroadcastWithWarpReduce(const std::vector<Expr*>& exprs) {
    // open stack space for global scope
    // The scope stack for tv_to_allocate wouldn't be needed
    //  if the allocations are guaranteed to be once and unique,
    //  which can currently be assumed but this pass tries not
    //  to rely on this assumption.
    running_tv_to_allocate_map_.emplace_back(
        std::make_unique<std::unordered_map<TensorView*, kir::Allocate*>>());
    running_visible_allocation_stack_.emplace_back(
        std::make_unique<std::vector<kir::Allocate*>>());
    kir::IrVisitor::handle(exprs);
  }

  void handle(Expr* expr) final {
    if (ir_utils::isTvOp(expr)) {
      // Process expr inputs if needs replacement
      for (auto inp : expr->inputs()) {
        if (auto input_ti = dynamic_cast<kir::TensorIndex*>(inp)) {
          auto replace = findMaybeReplacedTensorIndex(input_ti);
          if (replace.has_value()) {
            val_replacement_map_[input_ti] = replace.value();
          }
        }
      }
    }
    kir::IrVisitor::handle(expr);
  }

  bool openLoopNestLevel(IterDomain* id) {
    if (id->isThread() || id->getParallelType() == ParallelType::Unswitch) {
      return false;
    }
    if (id->getParallelType() == ParallelType::Serial ||
        id->getParallelType() == ParallelType::Unroll) {
      return !id->isBroadcast();
    }
    return true;
  }

  void handle(kir::ForLoop* for_loop) final {
    // Keep track of visible reduction outputs
    bool open_nest_level = openLoopNestLevel(for_loop->iter_domain());
    if (open_nest_level) {
      running_tv_to_allocate_map_.emplace_back(
          std::make_unique<std::unordered_map<TensorView*, kir::Allocate*>>());
      running_visible_allocation_stack_.emplace_back(
          std::make_unique<std::vector<kir::Allocate*>>());
    }
    for (auto expr : for_loop->body().exprs()) {
      handle(expr);
    }
    if (open_nest_level) {
      running_tv_to_allocate_map_.pop_back();
      running_visible_allocation_stack_.pop_back();
    }
  }

  void handle(kir::IfThenElse* ite) final {
    running_visible_allocation_stack_.emplace_back(
        std::make_unique<std::vector<kir::Allocate*>>());
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    running_visible_allocation_stack_.pop_back();
    running_visible_allocation_stack_.emplace_back(
        std::make_unique<std::vector<kir::Allocate*>>());
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
    running_visible_allocation_stack_.pop_back();
  }

  //! Place this allocate on the list of currently visible allocations,
  //!  organized by loop nest level.
  void handle(kir::Allocate* allocate) final {
    if (allocate->memoryType() != MemoryType::Local) {
      return;
    }
    if (auto tv = dynamic_cast<TensorView*>(allocate->buffer())) {
      if (tv->definition()) {
        if (tv->definition()->isA<ReductionOp>() ||
            tv->definition()->isA<BroadcastOp>()) {
          running_visible_allocation_stack_.back()->push_back(allocate);
        }
      }
    }
  }

  //! Checks if the given tv has been replaced by broadcast fusion.
  //!  returns the replaced TensorIndex if so.
  c10::optional<kir::TensorIndex*> findMaybeReplacedTensorIndex(
      kir::TensorIndex* tensor_index) {
    auto tv = tensor_index->view();
    auto tensor_index_it = running_tv_replacement_map_.find(tv);
    if (tensor_index_it != running_tv_replacement_map_.end()) {
      return tensor_index_it->second;
    }
    return c10::nullopt;
  }

  //! Iterate backwards on the currently visible loop scopes
  //!  and find the first allocation corresponding to the
  //!  given tv.
  kir::Allocate* getActiveAllocateFor(TensorView* tv) {
    for (auto frame_it = running_visible_allocation_stack_.rbegin();
         frame_it != running_visible_allocation_stack_.rend();
         frame_it++) {
      for (auto allocate_it = (*frame_it)->rbegin();
           allocate_it != (*frame_it)->rend();
           allocate_it++) {
        auto candidate_allocate = *allocate_it;
        if (candidate_allocate->buffer() == tv) {
          return candidate_allocate;
        }
      }
    }
    TORCH_INTERNAL_ASSERT(
        false, "lower_warp_reduce: cannot find allocation for this op");
    return nullptr;
  }

  bool isOpInputRegisterTV(Expr* expr) {
    for (auto inp : expr->inputs()) {
      if (auto inp_ti = dynamic_cast<kir::TensorIndex*>(inp)) {
        if (inp_ti->view()->getMemoryType() != MemoryType::Local) {
          return false;
        }
      }
    }

    return true;
  }

  bool isOpOutputRegisterTV(Expr* expr) {
    for (auto out : expr->outputs()) {
      if (auto out_ti = dynamic_cast<kir::TensorIndex*>(out)) {
        if (out_ti->view()->getMemoryType() != MemoryType::Local) {
          return false;
        }
      }
    }

    return true;
  }

  //! Updates map of serially visible reduction tvs, see comment on
  //!  running_tv_to_allocate_map_.
  void handle(ReductionOp* reduction) final {
    if (!isOpOutputRegisterTV(reduction)) {
      return;
    }
    auto reduction_ti_out = dynamic_cast<kir::TensorIndex*>(reduction->out());
    TORCH_INTERNAL_ASSERT(
        reduction_ti_out,
        "lower_warp_reduce: Pass needs to be run after indexing");

    // keep track of which reduction buffer this expr writes into
    auto reduction_allocate = getActiveAllocateFor(reduction_ti_out->view());
    running_tv_to_allocate_map_.back()->operator[](reduction_ti_out->view()) =
        reduction_allocate;
  }

  void handle(BroadcastOp* broadcast) final {
    if (!isOpInputRegisterTV(broadcast) || !isOpOutputRegisterTV(broadcast)) {
      return;
    }
    tryAddOutputToReplaceMap(broadcast);
  }

  //! Detects if this broadcast can be fused with the producer reduction.
  //!  adds the output of broadcast to replacement map if all above mentioned
  //!  conditions check.
  void tryAddOutputToReplaceMap(BroadcastOp* broadcast) {
    if (auto in_ti = dynamic_cast<kir::TensorIndex*>(broadcast->in())) {
      if (!in_ti->view()->definition()->isA<ReductionOp>()) {
        return;
      }
      auto out_ti = broadcast->out()->as<kir::TensorIndex>();
      auto out_tv = out_ti->view();

      // check reduction-broadcast mapping:
      if (!canFuseBroadcastWithWarpReduction(
              out_tv->definition()->as<BroadcastOp>())) {
        return;
      }

      // check buffers are size-1
      auto reduction_allocate_it =
          running_tv_to_allocate_map_.back()->find(in_ti->view());
      if (reduction_allocate_it == running_tv_to_allocate_map_.back()->end()) {
        // The producer reduction is not in the serially visible scope,
        //  as defined in openLoopNestLevel. There still could be some
        //  cases that we could fuse but disabled for simplicity.
        return;
      }

      // Cannot replace if either the reduction buffer or broadcast buffer does
      // not have
      //  a size of 1, since it would have required re-indexing.
      if (!reduction_allocate_it->second->size()->isConstInt() ||
          reduction_allocate_it->second->size()->evaluateInt() != 1) {
        return;
      }

      auto broadcast_allocate = getActiveAllocateFor(out_tv);
      if (!broadcast_allocate->size()->isConstInt() ||
          broadcast_allocate->size()->evaluateInt() != 1) {
        return;
      }

      // Write the tv in to the replacement map
      //  so the future uses of this tv will put
      //  the tensorIndex's in the actual replacement map.
      running_tv_replacement_map_[out_tv] = in_ti;
    }
  }

  // Checks if the given IterDomain is mapped to a single warp,
  //  i.e. they are known at compile time to be of constant
  //   size of warp_size and they are paralleled on TIDx
  int warp_size = at::cuda::warp_size();
  bool isSingleWarp(IterDomain* id) {
    if (id->getParallelType() != ParallelType::TIDx) {
      return false;
    }

    if (!GpuLower::current()->getWarpPaddedParallelInfo().is_tidx_single_warp) {
      return false;
    }

    // Prioritize checking for padded dimension
    if (id->getMaybeSizeAfterPadding().has_value()) {
      return id->getMaybeSizeAfterPadding().value() == warp_size;
    }

    if (id->extent()->isConstScalar()) {
      return id->extent()->evaluateInt() == warp_size;
    }

    return false;
  }

  // Check if this broadcast can be fused with the producer reduction
  //  Assumes:
  //   1. Already checked the producer of input is a reduction
  //   2. Already checked the producer reduction is in the same loop nest
  //  Checks:
  //   1. Reduction is only non-trivially parallel on TIDx as a single warp
  //   2. Broadcast is only non-trivially parallel on TIDx as a single warp
  bool canFuseBroadcastWithWarpReduction(BroadcastOp* broadcast) {
    auto reduction_out_tv = broadcast->in()->as<TensorView>();
    auto broadcast_out_tv = broadcast->out()->as<TensorView>();

    bool reduction_has_single_warp = false, broadcast_has_single_warp = false;

    for (auto id : reduction_out_tv->domain()->domain()) {
      if (id->isReduction() && id->isThread() && !isSingleWarp(id)) {
        return false;
      }
      if (id->isReduction() && isSingleWarp(id)) {
        reduction_has_single_warp = true;
      }
    }
    for (auto id : broadcast_out_tv->domain()->domain()) {
      if (id->isBroadcast() && id->isThread() && !isSingleWarp(id)) {
        return false;
      }
      if (id->isBroadcast() && isSingleWarp(id)) {
        broadcast_has_single_warp = true;
      }
    }
    return reduction_has_single_warp && broadcast_has_single_warp;
  }

 private:
  //! A naive record of kir tv's that will need replacement at each expr,
  //!  could need some extension for more precise scope based analysis in the
  //!  future especially if we have more complex IfThenElse blocks than
  //!  predicates and unroll.
  std::unordered_map<TensorView*, kir::TensorIndex*>
      running_tv_replacement_map_;

  //! Keeps track of the allocated buffers that the exprs will write/read
  //!  at each expr. Each outer vector element records the allocations at each
  //!  running scope level as this pass iterate through the loop nest.
  std::vector<std::unique_ptr<std::vector<kir::Allocate*>>>
      running_visible_allocation_stack_;

  //! A different version of running_visible_allocation_stack_ constructed for
  //! convenience,
  //!  the difference is that thread loops, serial broadcast loops, and
  //!  IfThenElse's are not modeled as another scope to model the textual
  //!  visibility on the generated kernel. The model of IfThenElse assumes the
  //!  only ITE's we have are predicates and unrolls, which might need to be
  //!  more precise.
  std::vector<std::unique_ptr<std::unordered_map<TensorView*, kir::Allocate*>>>
      running_tv_to_allocate_map_;

  //! This map is the final output of this pass and a val replacement map will
  //! be run using
  //!  it. All keys and values are TensorIndex's, and before this pass each
  //!  TensorIndex is uniquely generated by lower_index pass for each access of
  //!  a tv.
  std::unordered_map<Val*, Val*> val_replacement_map_;
};

} // namespace

std::vector<Expr*> fuseWarpReduce(const std::vector<Expr*> exprs) {
  return FuseBroadcastWithWarpReduce::fuse(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
