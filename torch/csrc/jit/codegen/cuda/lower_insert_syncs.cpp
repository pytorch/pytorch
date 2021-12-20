#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Scan through Kernel IR for-loops to insert Sync nodes to avoid
//! Write-After-Read (WAR) race condition.
//!
//! Example:
//!   for () {
//!     smem_buf[threadIdx.x] = x;
//!     __syncthreads();
//!     buf[threadId.x] = smem_buf[threadIdx.x + 1];
//!  }
//!
//! In this case, additional syncthreads is needed at the end of the
//! loop body to avoid a hazard with smem_buf.

//! Keeping track the allocations of SMEM TVs
class SmemAllocMap {
 public:
  //! Insert a new node if it's a SMEM allocation
  void insert(kir::Allocate* alloc) {
    if (auto tv = dynamic_cast<kir::TensorView*>(alloc->buffer())) {
      if (tv->memoryType() == MemoryType::Shared) {
        // Note that a TensorView can have two allocations due to
        // unswitch.
        auto p = map_.insert({tv, alloc});
        // If there's an existing entry, reset it with the new
        // alloc. Currently, the existing alloc is actually the same
        // as the new one as each expression is just inserted to both
        // then and else parts of the unswitched loop, but this should
        // be changed.
        if (!p.second) {
          p.first->second = alloc;
        }
      }
    }
  }

  //! Get the buffer that is actually allocated for a given TV
  kir::TensorView* getRealBuffer(kir::TensorView* tv) const {
    auto it = map_.find(tv);
    TORCH_INTERNAL_ASSERT(
        it != map_.end(), "Allocation not found for ", kir::toString(tv));
    const kir::Allocate* alloc = it->second;
    while (alloc->alias()) {
      alloc = alloc->alias();
    }
    auto buf = alloc->buffer();
    TORCH_INTERNAL_ASSERT(buf->isA<kir::TensorView>());
    return buf->as<kir::TensorView>();
  }

 private:
  std::unordered_map<kir::TensorView*, kir::Allocate*> map_;
};

//! Insert WAR sync for a given ForLoop
class LocalSyncInserterForLoop {
  using TvSet = std::unordered_set<const kir::TensorView*>;

 public:
  //! Insert Sync nodes at the end of a given for-loop when a WAR
  //! hazard may happen.
  LocalSyncInserterForLoop(kir::ForLoop* fl, SmemAllocMap& alloc_map)
      : alloc_map_(alloc_map) {
    for (auto expr : fl->body().exprs()) {
      handle(expr);
    }

    // No need to insert sync when the loop is not actually generated
    if (fl->iter_domain()->isThread() || fl->iter_domain()->isBroadcast()) {
      return;
    }

    // Determine if any smem TV is written to at beginning of the for-loop
    // and whether that smem TV is read from at the end of the for-loop
    // Insert new SyncThreads at end of for-loop to prevent WAR race condition
    //
    // TODO: replace __syncthreads with __threadfence for alias ops
    //
    if (detectIntersection(initial_, final_) &&
        !fl->body().exprs().back()->isA<kir::Sync>() && !is_last_op_sync_) {
      kir::IrBuilder ir_builder(GpuLower::current()->kernel());
      fl->body().push_back(ir_builder.create<kir::Sync>(true));
      initial_sync_ = true;
      is_last_op_sync_ = true;
      final_.clear();
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

  void handle(kir::Expr* expr) {
    if (ir_utils::isTVOp(expr)) {
      is_last_op_sync_ = false;

      // For this SyncInserter
      if (initial_sync_) {
        addInputSmemTvs(expr, final_);
      } else {
        addInputSmemTvs(expr, final_);
        addOutputSmemTvs(expr, initial_);
      }

      // For parent SyncInserter
      addOutputSmemTvs(expr, all_smem_outputs_);
      addInputSmemTvs(expr, all_smem_inputs_);
    } else if (auto sync = dynamic_cast<kir::Sync*>(expr)) {
      handle(sync);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    } else if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
      alloc_map_.insert(alloc);
    }
  }

  void handle(kir::Sync* sync) {
    is_last_op_sync_ = true;
    initial_sync_ = true;
    final_.clear();
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
    LocalSyncInserterForLoop child_sync_inserter(fl, alloc_map_);

    const auto& child_inputs = child_sync_inserter.all_smem_inputs();
    const auto& child_outputs = child_sync_inserter.all_smem_outputs();
    const bool maybe_skipped = !fl->start()->isZeroInt() &&
        !isParallelTypeThread(fl->iter_domain()->parallelType());

    // Default - Track all smem inputs / outputs
    all_smem_inputs_.insert(child_inputs.begin(), child_inputs.end());
    all_smem_outputs_.insert(child_outputs.begin(), child_outputs.end());

    // Propagate the last_op_sync flag from the child loop. If the
    // child is deterministically executed at least once, just set the
    // flag with the child flag. Otherwise, conservatively set the
    // flag, i.e., if the current flag is true and the child flag is
    // also true, we can say the last op is still sync.
    if (!maybe_skipped) {
      is_last_op_sync_ = child_sync_inserter.is_last_op_sync_;
    } else {
      is_last_op_sync_ =
          is_last_op_sync_ && child_sync_inserter.is_last_op_sync_;
    }

    // When the child is not guaranteed to have sync.
    if (!child_sync_inserter.initial_sync_) {
      // If no sync is yet found, add the child outputs to
      // initial.
      if (!initial_sync_) {
        initial_.insert(child_outputs.begin(), child_outputs.end());
      }
      // Add the child inputs to final even when inital_sync is false,
      // which only means sync may not be found yet.
      final_.insert(child_inputs.begin(), child_inputs.end());
    } else {
      // Similar to the above case, but here, the child is guaranteed
      // to have sync, so we only need to look at initial and final.
      if (!initial_sync_) {
        initial_.insert(
            child_sync_inserter.initial().begin(),
            child_sync_inserter.initial().end());
      }
      if (!maybe_skipped) {
        initial_sync_ = true;
        final_.clear();
      }
      final_.insert(
          child_sync_inserter.final().begin(),
          child_sync_inserter.final().end());
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

  void addOutputSmemTvs(const kir::Expr* expr, TvSet& set) {
    for (auto out : expr->outputs()) {
      if (auto tv = dynamic_cast<kir::TensorView*>(out)) {
        if (tv->memoryType() == MemoryType::Shared) {
          auto real_tv = alloc_map_.getRealBuffer(tv);
          set.insert(real_tv);
        }
      }
    }
  }

  void addInputSmemTvs(const kir::Expr* expr, TvSet& set) {
    for (auto in : expr->inputs()) {
      if (auto tv = dynamic_cast<kir::TensorView*>(in)) {
        if (tv->memoryType() == MemoryType::Shared) {
          auto real_tv = alloc_map_.getRealBuffer(tv);
          set.insert(real_tv);
        }
      }
    }
  }

 private:
  //! Allocation map of SMEM buffers
  SmemAllocMap& alloc_map_;

  //! Track Shared Memory Inputs (Reads) for parent for-loop
  TvSet all_smem_inputs_;

  //! Track Shared Memory Outputs (Writes) for parent for-loop
  TvSet all_smem_outputs_;

  //! Shared Memory Writes at beginning of the for-loop
  //! before first SyncThreads
  TvSet initial_;

  //! Shared Memory Reads at end of the for-loop
  //! Cleared after each SyncThreads
  TvSet final_;

  //! Track first sync deterministically found in for-loop. Even when a
  //! child loop has a sync, if it may not be executed due to non-zero
  //! start value, this flag remains false.
  bool initial_sync_ = false;

  //! Track if last op is sync
  bool is_last_op_sync_ = false;
};

class LocalSyncInserter {
 public:
  //! Write-After-Read race conditions are only found within for-loops.
  //! Sync nodes are inserted directly into the for-loops.
  //! The expressions are modified in-place and exprs is const.
  static void insertSyncs(const std::vector<kir::Expr*>& exprs) {
    LocalSyncInserter inserter;
    inserter.insert(exprs);
  }

 private:
  void insert(const std::vector<kir::Expr*>& exprs) {
    for (auto expr : exprs) {
      if (auto fl = dynamic_cast<kir::ForLoop*>(expr)) {
        LocalSyncInserterForLoop sync_inserter(fl, alloc_map_);
      } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
        insert(ite->thenBody().exprs());
        insert(ite->elseBody().exprs());
      } else if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
        alloc_map_.insert(alloc);
      }
    }
  }

 private:
  SmemAllocMap alloc_map_;
};

class ExprFlattener : private kir::IrVisitor {
 private:
  void handle(kir::Expr* expr) {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      expr->accept(this);
    } else {
      exprs_.push_back(expr);
    }
  }

  void visit(const kir::ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      handle(expr);
    }
  }

  void visit(const kir::IfThenElse* ite) final {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

 private:
  std::vector<kir::Expr*> exprs_;

 public:
  //! Flattens scopes extracting out a single ordered list of exprs.
  static std::vector<kir::Expr*> flatten(
      const std::vector<kir::Expr*>& loop_nests) {
    ExprFlattener flattener;
    for (auto expr : loop_nests) {
      flattener.handle(expr);
    }
    return flattener.exprs_;
  }
};

class ValidatePlacementAfterWrites : private kir::IrVisitor {
 public:
  //! Validate no expr in writes found under loop
  static void validate(
      kir::ForLoop* loop,
      const std::unordered_set<kir::Expr*>& writes) {
    ValidatePlacementAfterWrites validator(writes);
    validator.handle(loop);
  }

 private:
  ValidatePlacementAfterWrites(const std::unordered_set<kir::Expr*>& writes)
      : writes_(writes) {}

  void handle(kir::Expr* expr) {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      expr->accept(this);
    } else {
      TORCH_INTERNAL_ASSERT(
          writes_.find(expr) == writes_.end(),
          "Block sync must be placed after ",
          kir::toString(expr));
    }
  }

  void visit(const kir::ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      handle(expr);
    }
  }

  void visit(const kir::IfThenElse* ite) final {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

 private:
  const std::unordered_set<kir::Expr*>& writes_;
};

class ReadAfterWriteSyncs : public kir::MutableIrVisitor {
 private:
  //! Traverse up the loop stack from loops_it and if a halo loop is
  //! found, place a given sync expr before the outer-most halo loop.
  bool insertBeforeHaloLoop(
      std::vector<kir::ForLoop*>::iterator loops_it,
      kir::Sync* sync_expr,
      const std::unordered_set<kir::Expr*>& writes) {
    std::vector<kir::ForLoop*>::iterator halo_loop_it;
    bool halo_loop_found = false;

    while (true) {
      if ((*loops_it)->iter_domain()->isThreadDim() &&
          (*loops_it)->iter_domain()->extent() != (*loops_it)->stop()) {
        halo_loop_found = true;
        halo_loop_it = loops_it;
      }

      if (loops_it == for_loops_.begin()) {
        break;
      }
      --loops_it;
    }

    // No halo loop found. Do not place the sync expr here. Return
    // false to indicate nothing is done.
    if (!halo_loop_found) {
      return false;
    }

    auto halo_loop = *halo_loop_it;

    // Make sure there's no write to the smem buffer inside the halo
    // loop. syncthreads is moved before the halo loop, so having
    // writes inside the loop invalidates the consistency.
    ValidatePlacementAfterWrites::validate(halo_loop, writes);

    if (halo_loop_it == for_loops_.begin()) {
      // place in global scope
      auto place_before_it =
          std::find(loop_nests_.begin(), loop_nests_.end(), halo_loop);
      TORCH_INTERNAL_ASSERT(place_before_it != loop_nests_.end());
      loop_nests_.insert(place_before_it, sync_expr);
    } else {
      auto place_in = *(halo_loop_it - 1);
      place_in->body().insert_before(halo_loop, sync_expr);
    }

    return true;
  }

  void handle(kir::Expr* expr) {
    if (!ir_utils::isTVOp(expr) || expr->isA<kir::Allocate>()) {
      expr->accept(this);
      return;
    }

    if (sync_after_.size() > 0 && sync_after_.front() == expr) {
      sync_after_.pop_front();
      auto last_writes = last_writes_.front();
      last_writes_.pop_front();
      // Found that a sync is needed
      TORCH_INTERNAL_ASSERT(expr->outputs()[0]->isA<kir::TensorView>());
      auto out_tv = expr->outputs()[0]->as<kir::TensorView>();

      // Find where a sync needs to be inserted
      // This is very similar to how allocations are placed, simply place sync
      // after the expression instead of placing like allocation where it goes
      // before.
      // TODO: This may be a common operation, could be worth making a utility
      // out of or saving state for tensor view ID -> for loop
      // TODO: Explicitly test the 3 cases below

      kir::IrBuilder ir_builder(GpuLower::current()->kernel());
      auto sync_expr = ir_builder.create<kir::Sync>();
      if (out_tv->fuserTv()->getComputeAtPosition() == 0) {
        // Sync should be placed at global scope, after its outer most loop if
        // it has one.
        kir::Expr* place_after = for_loops_.size() > 0 ? for_loops_[0] : expr;
        // Find location in loop_nests_
        auto place_after_it =
            std::find(loop_nests_.begin(), loop_nests_.end(), place_after);
        TORCH_INTERNAL_ASSERT(
            place_after_it != loop_nests_.end(),
            "Could not figure out where to place synchronization. ",
            "Tried to place after, ",
            toString(place_after),
            ", but could not find this expression at the global scope.");
        loop_nests_.insert(place_after_it + 1, sync_expr);
      } else {
        // Find the last loop in computeAt of out_tv, this is the loop where we
        // would place an allocation for out_tv
        auto fuser_tv = out_tv->fuserTv();
        auto lowered_local_id =
            GpuLower::current()
                ->lowerValue(fuser_tv->axis(
                    (int)out_tv->fuserTv()->getComputeAtPosition() - 1))
                ->as<kir::IterDomain>();

        auto loops_it = std::find_if(
            for_loops_.begin(),
            for_loops_.end(),
            [&lowered_local_id](const auto& loop) {
              return GpuLower::current()->caLoopMap().areMapped(
                         loop->iter_domain(), lowered_local_id) ||
                  loop->iter_domain()->parallelType() == ParallelType::Unroll;
            });

        TORCH_INTERNAL_ASSERT(loops_it != for_loops_.end());

        // block sync must be placed before halo-extended loops
        if (insertBeforeHaloLoop(loops_it, sync_expr, last_writes)) {
          return;
        }

        auto place_in = *loops_it;
        kir::Expr* place_after = nullptr;

        if (loops_it + 1 == for_loops_.end()) {
          // Inline allocation, place after expr
          place_after = expr;
        } else {
          // Place allocation after the last computeAt axis
          // TODO: may be more efficient to place after the first non-computeAt
          // axis
          place_after = *(loops_it + 1);
        }

        place_in->body().insert_after(place_after, sync_expr);
      }
    }
  }

  void visit(kir::ForLoop* fl) final {
    for_loops_.push_back(fl);
    // Modifying in place, make a copy of the vector
    const std::vector<kir::Expr*> exprs = fl->body().exprs();
    for (auto expr : exprs) {
      handle(expr);
    }
    for_loops_.pop_back();
  }

  void visit(kir::IfThenElse*) final {
    TORCH_INTERNAL_ASSERT(
        false,
        "Pass does not support conditional statements, ",
        "this pass should be run before any conditionals are placed in code.");
  }

  // Clear the modify status for all shared memory buffers
  static void cleanSharedMemory(
      std::unordered_map<kir::Val*, kir::Expr*>& smem) {
    smem.clear();
  }

  // Return a set of expressions that modify shared-memory
  // tensors. Expressions are excluded when syncthreads are already
  // placed.
  std::unordered_set<kir::Expr*> isModifiedSharedMemory(
      const std::unordered_map<kir::Val*, kir::Expr*>& smem,
      const std::vector<kir::Val*>& tvs) const {
    std::unordered_set<kir::Expr*> last_writes;
    for (auto tv : tvs) {
      auto it = smem.find(tv);
      if (it != smem.end()) {
        last_writes.insert(it->second);
      }
    }
    return last_writes;
  }

  ReadAfterWriteSyncs(std::vector<kir::Expr*> _loop_nests)
      : loop_nests_(std::move(_loop_nests)) {
    // Fusion shared_memory values
    // Tracks if shared memory is modified
    std::unordered_map<kir::Val*, kir::Expr*> smem;

    // Flatten all the expressions
    auto flattened_exprs = ExprFlattener::flatten(loop_nests_);

    kir::Expr* prev_tv_expr = nullptr;
    for (auto expr : flattened_exprs) {
      if (!ir_utils::isTVOp(expr) || expr->isA<kir::Allocate>()) {
        continue;
      }

      auto last_writes = isModifiedSharedMemory(smem, expr->inputs());
      if (!last_writes.empty()) {
        TORCH_INTERNAL_ASSERT(
            prev_tv_expr != nullptr,
            "Can't require sync on inputs, however, detected it's needed.");
        sync_after_.push_back(prev_tv_expr);
        last_writes_.push_back(last_writes);
        cleanSharedMemory(smem);
      }

      for (auto out : expr->outputs()) {
        if (out->isA<kir::TensorView>()) {
          if (out->as<kir::TensorView>()->memoryType() == MemoryType::Shared) {
            smem[out] = expr;
          }
        }
      }

      prev_tv_expr = expr;
    }

    // Insert read after write syncs
    const std::vector<kir::Expr*> exprs = loop_nests_;
    for (auto expr : exprs) {
      handle(expr);
    }

    TORCH_INTERNAL_ASSERT(
        sync_after_.empty(), "Didn't place all required syncs.");
  }

 private:
  //! Keep track of expressions that must be followed by syncthreads
  std::deque<kir::Expr*> sync_after_;

  //! Keep track of write expressions that must be placed before
  //! syncthreads.
  //!
  //! syncthreads is placed after for each expression of
  //! sync_after_. However, if it's inside a loop with halo, it must
  //! be placed before that. last_writes_ keeps track of expressions
  //! modifying the smem buffer each syncthreads is used for so that
  //! it is not placed before those write expressions.
  std::deque<std::unordered_set<kir::Expr*>> last_writes_;

  //! Keep track of for loops while inserting syncthreads
  std::vector<kir::ForLoop*> for_loops_;

  //! Loop-nests where syncthreads are inserted
  std::vector<kir::Expr*> loop_nests_;

 public:
  static std::vector<kir::Expr*> insert(
      const std::vector<kir::Expr*>& loop_nests) {
    ReadAfterWriteSyncs inserter(loop_nests);
    return inserter.loop_nests_;
  }
};

} // namespace

std::vector<kir::Expr*> insertRawThreadSynchronization(
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertRawThreadSynchronization");
  return ReadAfterWriteSyncs::insert(exprs);
}

std::vector<kir::Expr*> insertWarThreadSynchronization(
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertWarThreadSynchronization");
  LocalSyncInserter::insertSyncs(exprs);
  return exprs;
}
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
