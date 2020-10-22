#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Scan through Kernel IR to insert Sync nodes to avoid
//! Write-After-Read (WAR) race condition
//!
class LocalSyncInserter {
  using TvSet = std::unordered_set<const kir::TensorView*>;

 public:
  //! Write-After-Read race conditions are only found within for-loops.
  //! Sync nodes are inserted directly into the for-loops.
  //! The expressions are modified in-place and exprs is const.
  static void insertSyncs(const std::vector<kir::Expr*>& exprs) {
    LocalSyncInserter sync_inserter;
    for (auto expr : exprs) {
      sync_inserter.handle(expr);
    }
  }

  const auto& initial() const {
    return initial_;
  }

  const auto& final() const {
    return final_;
  }

  const auto& all_smem_inputs() const {
    return all_smem_inputs_;
  }

  const auto& all_smem_outputs() const {
    return all_smem_outputs_;
  }

 private:
  // TODO(kir): this is a place where a mutable IR visitor may be appropriate
  void handle(kir::Expr* expr) {
    if (ir_utils::isTVOp(expr)) {
      // For this SyncInserter
      initial_sync_ ? addInputSmemTvs(expr, final_)
                    : addOutputSmemTvs(expr, initial_);

      // For parent SyncInserter
      addOutputSmemTvs(expr, all_smem_outputs_);
      addInputSmemTvs(expr, all_smem_inputs_);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    }
  }

  void handle(kir::IfThenElse* ite) {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

  void handle(kir::ForLoop* fl) {
    // Track if last op in body is sync in nested for-loop
    bool is_last_op_sync_ = false;
    for (auto expr : fl->body().exprs()) {
      is_last_op_sync_ = false;
      if (expr->isA<kir::Sync>()) {
        initial_sync_ = true;
        final_.clear();
      } else if (expr->isA<kir::ForLoop>()) {
        // Recursively handle nested for-loop
        LocalSyncInserter child_sync_inserter;
        child_sync_inserter.handle(expr);
        const auto& child_inputs = child_sync_inserter.all_smem_inputs();
        const auto& child_outputs = child_sync_inserter.all_smem_outputs();

        // Default - Track all smem inputs / outputs
        all_smem_inputs_.insert(child_inputs.begin(), child_inputs.end());
        all_smem_outputs_.insert(child_outputs.begin(), child_outputs.end());

        if (!initial_sync_) {
          // Parent - None
          if (!child_sync_inserter.initial_sync_) {
            // Child - None
            // Append All Child Outputs to Parent Initial
            initial_.insert(child_outputs.begin(), child_outputs.end());
          } else if (child_sync_inserter.has_war_hazard_sync_) {
            // Child - WAR race
            // Parent first sync
            // Inherit Child Initial / Clear Parent Final
            initial_sync_ = true;
            is_last_op_sync_ = true;
            initial_.insert(
                child_sync_inserter.initial().begin(),
                child_sync_inserter.initial().end());
            final_.clear();
          } else {
            // Child - 1+
            // Parent first sync
            // Inherit Child Initial + Final
            initial_sync_ = true;
            initial_.insert(
                child_sync_inserter.initial().begin(),
                child_sync_inserter.initial().end());
            final_.insert(
                child_sync_inserter.final().begin(),
                child_sync_inserter.final().end());
          }
        } else {
          // Parent - 1+
          if (!child_sync_inserter.initial_sync_) {
            // Child - None
            // Append All Child to Parent Last
            final_.insert(child_inputs.begin(), child_inputs.end());
          } else if (child_sync_inserter.has_war_hazard_sync_) {
            // Child - WAR race
            // Clear Parent Last / Discard Child Initial
            is_last_op_sync_ = true;
            final_.clear();
          } else {
            // Child - 1+
            // Inherit Child Final / Discard Child Initial
            final_.insert(
                child_sync_inserter.final().begin(),
                child_sync_inserter.final().end());
          }
        }
      } else {
        handle(expr);
      }
    }

    // This level of the nested for-loop may not exist in the kernel.
    // However, subsequent levels can exist, so we handle the body of the
    // for-loop first.
    if (!fl->iter_domain()->isThread() && !fl->iter_domain()->isBroadcast()) {
      // Determine if any smem TV is written to at beginning of the for-loop
      // and whether that smem TV is read from at the end of the for-loop
      // Insert new SyncThreads at end of for-loop to prevent WAR race condition
      //
      // TODO: replace __syncthreads with __threadfence for alias ops
      //
      if (detectIntersection(initial_, final_) &&
          !fl->body().exprs().back()->isA<kir::Sync>() && !is_last_op_sync_) {
        // std::cout << "WAR race detected; Add Sync" << std::endl;
        has_war_hazard_sync_ = true;
        kir::IrBuilder ir_builder(GpuLower::current()->kernel());
        fl->body().push_back(ir_builder.create<kir::Sync>(true));
      }
    }
  }

  static bool detectIntersection(const TvSet& left, const TvSet& right) {
    for (auto item : left) {
      if (right.find(item) != right.end()) {
        return true;
      }
    }
    return false;
  }

  static void addOutputSmemTvs(const kir::Expr* expr, TvSet& set) {
    for (auto out : expr->outputs()) {
      if (auto tv = dynamic_cast<kir::TensorView*>(out)) {
        if (tv->memoryType() == MemoryType::Shared) {
          set.insert(tv);
        }
      }
    }
  }

  static void addInputSmemTvs(const kir::Expr* expr, TvSet& set) {
    for (auto in : expr->inputs()) {
      if (auto tv = dynamic_cast<kir::TensorView*>(in)) {
        if (tv->memoryType() == MemoryType::Shared) {
          set.insert(tv);
        }
      }
    }
  }

 private:
  // Track Shared Memory Inputs (Reads) for parent for-loop
  TvSet all_smem_inputs_;

  // Track Shared Memory Outputs (Writes) for parent for-loop
  TvSet all_smem_outputs_;

  // Shared Memory Writes at beginning of the for-loop
  // before first SyncThreads
  TvSet initial_;

  // Shared Memory Reads at end of the for-loop
  // Cleared after each SyncThreads
  TvSet final_;

  // Track first sync found in for-loop
  bool initial_sync_ = false;

  // Track sync was inserted for war hazard
  bool has_war_hazard_sync_ = false;
};

} // namespace

std::vector<kir::Expr*> insertThreadSynchronization(
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("insertThreadSynchronization");
  LocalSyncInserter::insertSyncs(exprs);
  return exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
