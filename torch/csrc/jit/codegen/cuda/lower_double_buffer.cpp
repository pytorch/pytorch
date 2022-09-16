#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/codegen/cuda/lower_double_buffer.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

unsigned int getDoubleBufferAxisPosition(const TensorView* tv) {
  // Double-buffering prefetches the next subregion of the tensor by
  // doubling the allocation. The subregion is defined by the axes
  // at the CA position till the inner-most position. There must be
  // at least one axis that is outside (left) of the CA position,
  // which defines the loop where prefetching is applied. Therefore,
  // the CA position must be larger than 0.

  TORCH_INTERNAL_ASSERT(tv->getComputeAtPosition() > 0);

  // Unroll must not exist outside of double-buffer axis
  auto first_unroll_it = std::find_if(
      tv->domain()->domain().begin(),
      tv->domain()->domain().end(),
      [](const auto axis) {
        return axis->getParallelType() == ParallelType::Unroll;
      });

  const int first_unroll_pos =
      std::distance(tv->domain()->domain().begin(), first_unroll_it);

  const int unroll_or_ca_pos =
      std::min((int)tv->getComputeAtPosition(), first_unroll_pos);

  TORCH_INTERNAL_ASSERT(
      unroll_or_ca_pos > 0,
      "Invalid tensor to double-buffer. Valid double buffer axis not found due to Unroll. ",
      tv->toString());

  int valid_pos = -1;
  // Skip parallelized or broadcast axes
  for (int i = unroll_or_ca_pos - 1; i >= 0; --i) {
    auto pt = tv->axis(i)->getParallelType();
    if (!isParallelTypeThread(pt) && !tv->axis(i)->isBroadcast()) {
      valid_pos = i;
      break;
    }
  }

  TORCH_INTERNAL_ASSERT(
      valid_pos >= 0,
      "Invalid tensor to double-buffer. Valid double buffer axis not found. ",
      tv->toString());

  return valid_pos;
}

IterDomain* getDoubleBufferAxis(const TensorView* tv) {
  return tv->axis((int)getDoubleBufferAxisPosition(tv));
}

void validateDoubleBufferedTensor(const TensorView* tv) {
  auto double_buffer_pos = getDoubleBufferAxisPosition(tv);

  // Like vectorization, only UnaryOp::Set with another TensorView is
  // considered.
  auto def = tv->definition();
  TORCH_INTERNAL_ASSERT(
      (def->isA<UnaryOp>() &&
       def->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Set) ||
          // Load store op should generally support double buffering.
          def->isA<LoadStoreOp>(),
      "Invalid tensor to double-buffer. Only tensor defined by UnaryOp::Set is supported: ",
      def->toString());

  TORCH_INTERNAL_ASSERT(
      def->input(0)->isA<TensorView>(),
      "Invalid tensor to double-buffer. Only tensor defined by UnaryOp::Set with TensorView is supported: ",
      def->toString());

  // Require the producer tensor to have been computed entirely for
  // the double-buffering loop. Otherwise, the producer itself would
  // also need to be double-bufferred.
  auto producer = def->input(0)->as<TensorView>();
  TORCH_INTERNAL_ASSERT(
      producer->getComputeAtPosition() <= double_buffer_pos,
      "Invalid tensor to double-buffer. The computeAt position of the producer tensor must be moved left: ",
      producer->toString());

  // Not strictly necessary, but only gmem -> smem or local and smem -> local
  // are allowed.
  const auto p_mem_type = producer->getMemoryType();
  const auto c_mem_type = tv->getMemoryType();
  TORCH_INTERNAL_ASSERT(
      (p_mem_type == MemoryType::Global &&
       (c_mem_type == MemoryType::Shared || c_mem_type == MemoryType::Local)) ||
          (p_mem_type == MemoryType::Shared && c_mem_type == MemoryType::Local),
      "Invalid tensor to double-buffer: ",
      tv->toString(),
      ". Producer memory type: ",
      p_mem_type,
      ". Consumer memory type: ",
      c_mem_type);

  return;
}

namespace {

// Initial inspection of a fusion to find and validate double buffered tensors
class DoubleBufferFusionInspector : private IterVisitor {
 public:
  DoubleBufferFusionInspector(Fusion* fusion, DoubleBufferInfo& db_info)
      : db_info_(db_info) {
    traverse(fusion);
  }

 private:
  using IterVisitor::handle;

  void handle(TensorView* tv) final {
    if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        tv->definition(), "Fusion input shouldn't be double buffered.", tv);

    validateDoubleBufferedTensor(tv);

    auto db_axis = getDoubleBufferAxis(tv);

    db_info_.setDoubleBufferAxis(tv, db_axis);
  }

 private:
  DoubleBufferInfo& db_info_;
};

// The epilogue loop is only created when the producer of a double
// buffer tensor is on smem, in which case it would otherwise require
// an additional predicate to guard buffer overruns. When it's on
// gmem, that isn't the case, so it does not need to create an
// epilogue loop.
bool requireEpilogue(const std::vector<Expr*>& exprs) {
  return std::any_of(exprs.begin(), exprs.end(), [](const Expr* expr) {
    return expr->input(0)->as<TensorView>()->getMemoryType() ==
        MemoryType::Shared;
  });
}

// Replicates double buffer loops for Prologue, Main, and
// Epilogue. Prologue only copies the load expressions of double
// buffered tensors, whereas Epilogue does any expression other than
// the loads. Main copies everything.
class DoubleBufferLoopCloner : public kir::IrVisitor {
 public:
  static kir::ForLoop* clone(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type) {
    DoubleBufferLoopCloner cloner(
        double_buffer_loop, double_buffer_load_exprs, loop_type);
    cloner.clone();
    return cloner.cloned_top_level_loop_;
  }

 private:
  DoubleBufferLoopCloner(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type)
      : double_buffer_loop_(double_buffer_loop),
        double_buffer_load_exprs_(double_buffer_load_exprs),
        loop_type_(loop_type) {}

  using kir::IrVisitor::handle;

  void clone() {
    const auto gpu_lower = GpuLower::current();

    // Cloning the double buffer loop as follows:
    //
    // Prologue: 0 to 1
    // Main: 0 to (extent-1)
    // Epilogue: (extent-1) to extent

    auto index = GpuLower::current()->caMap()->getIndexVariable(
        double_buffer_loop_->iter_domain(), loop_type_);
    auto start = double_buffer_loop_->start();
    auto stop = double_buffer_loop_->stop();
    auto stage_depth = gpu_lower->doubleBufferInfo().getStageDepthFor(
        double_buffer_loop_->iter_domain());

    if (loop_type_ == DoubleBufferLoopStage::Prolog) {
      TORCH_INTERNAL_ASSERT(start->isZeroInt());
      stop = SimplifyingIrBuilder::create<Int>(stage_depth - 1);
    } else if (
        loop_type_ == DoubleBufferLoopStage::Main &&
        requireEpilogue(double_buffer_load_exprs_)) {
      stop = IrBuilder::subExpr(
          double_buffer_loop_->stop(), gpu_lower->kernel()->oneVal());
    } else if (loop_type_ == DoubleBufferLoopStage::Epilog) {
      TORCH_INTERNAL_ASSERT(requireEpilogue(double_buffer_load_exprs_));
      start = IrBuilder::subExpr(
          double_buffer_loop_->stop(),
          SimplifyingIrBuilder::create<Int>(stage_depth - 1));
    }

    cloned_top_level_loop_ = IrBuilder::create<kir::ForLoop>(
        double_buffer_loop_->iter_domain(),
        index,
        start,
        stop,
        gpu_lower->kernel()->oneVal(),
        false,
        nullptr,
        double_buffer_loop_->isUnrollRequired(),
        loop_type_);

    handle(double_buffer_loop_);

    if (stage_depth > 2) {
      cloned_top_level_loop_->body().push_back(
          IrBuilder::create<kir::CpAsyncCommit>());
    }
  }

  void handle(kir::ForLoop* fl) final {
    kir::ForLoop* cloned_loop = fl == double_buffer_loop_
        ? cloned_top_level_loop_
        : IrBuilder::create<kir::ForLoop>(fl);

    cloned_scopes_.push_back(&cloned_loop->body());

    kir::IrVisitor::handle(fl);

    cloned_scopes_.pop_back();

    // Add the cloned loop into the parent loop body only when the
    // cloned loop contains expressions.
    if (!cloned_loop->body().empty() && !cloned_scopes_.empty()) {
      cloned_scopes_.back()->push_back(cloned_loop);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    TORCH_INTERNAL_ASSERT(false, "No IfThenElse should exist yet");
  }

  void handle(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::handle(expr);
      return;
    }

    TORCH_INTERNAL_ASSERT(!cloned_scopes_.empty());

    if (loop_type_ == DoubleBufferLoopStage::Main) {
      cloned_scopes_.back()->push_back(expr);
      return;
    }

    // In Prologue and Epilogue, either load expressions or anything
    // else are copied. Note that there can be multiple exprs defining
    // double buffered TVs (e.g., buffer initialization).

    auto out_tv = ir_utils::getTvOutput(expr);
    const auto is_double_buffer_load_expr = std::any_of(
        double_buffer_load_exprs_.begin(),
        double_buffer_load_exprs_.end(),
        [out_tv](const auto load_expr) {
          auto double_buffer_tv = ir_utils::getTvOutput(load_expr);
          TORCH_INTERNAL_ASSERT(double_buffer_tv != nullptr);
          return out_tv == double_buffer_tv;
        });
    if ((loop_type_ == DoubleBufferLoopStage::Prolog &&
         is_double_buffer_load_expr) ||
        (loop_type_ == DoubleBufferLoopStage::Epilog &&
         !is_double_buffer_load_expr)) {
      cloned_scopes_.back()->push_back(expr);
    }
  }

 private:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<Expr*>& double_buffer_load_exprs_;
  const DoubleBufferLoopStage loop_type_;

  kir::ForLoop* cloned_top_level_loop_ = nullptr;
  std::deque<kir::Scope*> cloned_scopes_;
};

using InsertionInfo = std::unordered_map<kir::ForLoop*, std::vector<Expr*>>;

// Traverse lowered loop-nests and find all double buffer loops and
// associated load expressions.
class DoubleBufferLoopNestInspector : private kir::IrVisitor {
 public:
  static InsertionInfo run(const std::vector<Expr*>& exprs) {
    DoubleBufferLoopNestInspector inspector(exprs);
    return inspector.insertion_info_;
  }

 private:
  DoubleBufferLoopNestInspector(const std::vector<Expr*>& exprs) {
    handle(exprs);
  }

  using kir::IrVisitor::handle;

  // Collect double buffer related information on a expr
  //  that is a memory load, i.e. a LoadStore or a Set.
  void handlePossibleLoadExpr(Expr* expr) {
    const auto gpu_lower = GpuLower::current();

    auto out_tv = ir_utils::getTvOutput(expr);

    if (out_tv == nullptr) {
      return;
    }

    // Ignore init loop
    if (!(out_tv->isDoubleBuffered() || out_tv->isCircularBuffered()) ||
        !expr->input(0)->isA<TensorView>()) {
      return;
    }

    auto double_buffer_loop =
        gpu_lower->doubleBufferInfo().getDoubleBufferLoop(out_tv, for_loops_);

    TORCH_INTERNAL_ASSERT(
        double_buffer_loop != nullptr,
        "No double buffer loop found for a double buffered tensor: ",
        out_tv->toString());

    validateDoubleBufferLoop(double_buffer_loop);

    insertion_info_[double_buffer_loop].push_back(expr);
  }

  void handle(UnaryOp* uop) final {
    handlePossibleLoadExpr(uop);
  }

  void handle(LoadStoreOp* ldst) final {
    handlePossibleLoadExpr(ldst);
  }

  static void validateDoubleBufferLoop(kir::ForLoop* loop) {
    TORCH_INTERNAL_ASSERT(
        loop->start()->isZeroInt(), "Unsupported loop: ", loop->toString());
    TORCH_INTERNAL_ASSERT(
        loop->step()->isOneInt(), "Unsupported loop: ", loop->toString());
    TORCH_INTERNAL_ASSERT(
        !loop->vectorize(),
        "Vectorized loop should not be the allocation loop for double-buffered tensor: ",
        loop->toString());
    TORCH_INTERNAL_ASSERT(
        !loop->vectorize_shift(),
        "Vectorize shift loop should not be the allocation loop for double-buffered tensor: ",
        loop->toString());
  }

  InsertionInfo insertion_info_;
};

// Apply double buffering transformations
class DoubleBufferInserter : private kir::ExprMutator {
 public:
  // When there exist multiple double buffer loops, apply
  // transformations to inner-most loops first. A single ExprMutator
  // pass can only process one loop.
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      InsertionInfo insertion_info) {
    auto inserted_exprs = exprs;
    while (!insertion_info.empty()) {
      DoubleBufferInserter inserter(inserted_exprs, insertion_info);
      inserted_exprs = inserter.exprs_;
    }
    return inserted_exprs;
  }

 private:
  DoubleBufferInserter(
      const std::vector<Expr*>& exprs,
      InsertionInfo& insertion_info)
      : insertion_info_(insertion_info) {
    auto num_double_buffer_loops = insertion_info.size();
    traverseAndInsert(exprs);
    TORCH_INTERNAL_ASSERT(processed_loop_ != nullptr);
    TORCH_INTERNAL_ASSERT(insertion_info.size() == num_double_buffer_loops - 1);
  }

  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* loop) final {
    kir::ExprMutator::handle(loop);

    // If another loop is already taken care of, no more loop should
    // be done in the same pass
    if (processed_loop_ != nullptr) {
      return;
    }

    auto it = insertion_info_.find(loop);
    if (it == insertion_info_.end()) {
      return;
    }

    insert(loop, it->second);
    processed_loop_ = loop;
    insertion_info_.erase(loop);
  }

  void insert(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& loads) {
    auto prologue_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, DoubleBufferLoopStage::Prolog);
    registerInsertBefore(double_buffer_loop, prologue_loop);

    auto write_to_smem =
        std::any_of(loads.begin(), loads.end(), [](const Expr* expr) {
          return expr->output(0)->as<TensorView>()->getMemoryType() ==
              MemoryType::Shared;
        });

    // RAW sync is not inserted for double buffered tensors. The only
    // exception is the prologue load.
    bool insert_cpasync_wait = false;
    if (write_to_smem) {
      // Here the initial sync before entering double buffer loop is
      //  inserted.

      // If any of the double buffered tensor in this double buffer
      //  loop is async copy. We want to wait for the gmem loads to
      //  finish before synchronizing the block.
      if (std::any_of(loads.begin(), loads.end(), ir_utils::isCpAsyncOp)) {
        auto stage_depth =
            GpuLower::current()->doubleBufferInfo().getStageDepthFor(
                double_buffer_loop->iter_domain());
        auto cp_async_wait =
            IrBuilder::create<kir::CpAsyncWait>(stage_depth - 2);
        registerInsertBefore(double_buffer_loop, cp_async_wait);
        insert_cpasync_wait = true;
      }

      // Insert the initial block sync before entering main loop.
      if (std::any_of(loads.begin(), loads.end(), [](Expr* expr) {
            return GpuLower::current()
                ->syncMap()
                .needsRawSync(ir_utils::getTvOutput(expr))
                .hasTID();
          })) {
        // If any of the double buffered loads require sync, as indicated
        //  by sync info map, insert the sync before entering the double buffer
        //  loop.
        // TODO:
        //  Currently not supporting double buffer in gmem, but short to mid
        //  term not yet a priority to go for this case.
        auto sync = IrBuilder::create<kir::BlockSync>(false);
        registerInsertBefore(double_buffer_loop, sync);
      }
    }

    auto main_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, DoubleBufferLoopStage::Main);

    registerReplace(double_buffer_loop, main_loop);

    // Insert the wait instruction in this pass instead
    //  of relying on WAR sync pass to do it.
    // The WAR sync pass today would insert the wait function
    //  exactly where we need it but the purpose of this wait
    //  insertion isn't exactly WAR protection.
    //
    // TODO: [Double Buffer Sync]
    //  We might eventually want to move the block sync inserted
    //   by WAR pass here as well since this sync insertion is kind
    //   of both WAR and RAW (or neither RAW nor WAR, depends
    //   on how we look at it).
    // Eg. in the case when a intermediate
    //   tensor is double buffered.
    //
    //  __block_sync();    // This is the initial sync
    //  For i in ...       // Double buffer loop
    //     A[i%2] = ...;
    //     ...  = A[1-i%2];
    //     __block_sync();  // sync within loop
    //     ...
    //  The "sync within loop" can be placed anywhere in the
    //   double buffer loop while in the case of RAW and WAR
    //   there'd be extra insertion point restrictions.
    //  We are currently not actively exploring opportunities
    //   with this property of "double buffer sync" so this
    //   is more conceptual at the moment, aka low priority.
    if (insert_cpasync_wait) {
      insertCpAsyncWaitInMainLoop(main_loop);
    }

    if (requireEpilogue(loads)) {
      auto epilogue_loop = DoubleBufferLoopCloner::clone(
          double_buffer_loop, loads, DoubleBufferLoopStage::Epilog);
      registerInsertAfter(double_buffer_loop, epilogue_loop);
    }
  }

  // Simple conservative rule for inserting async copy wait
  //  primitive in the double buffer loop:
  void insertCpAsyncWaitInMainLoop(kir::ForLoop* main_loop) {
    TORCH_INTERNAL_ASSERT(
        !main_loop->body().empty(),
        "Double buffer sync insertion: empty main loop.");
    // Note: This pass explicitly assumes that WAR sync has been
    //  inserted so would need to be updated if we re-order the
    //  passes. Cleanups suggested in [Double Buffer Sync]
    //  would resolve this dependency on pass ordering.
    auto end_of_loop_expr = main_loop->body().exprs().back();
    auto stage_depth = GpuLower::current()->doubleBufferInfo().getStageDepthFor(
        main_loop->iter_domain());
    auto cp_async_wait = IrBuilder::create<kir::CpAsyncWait>(stage_depth - 2);

    // Check if a sync has been inserted by WAR sync pass.
    auto block_sync_it = std::find_if(
        main_loop->body().exprs().rbegin(),
        main_loop->body().exprs().rend(),
        [](const Expr* expr) { return expr->isA<kir::BlockSync>(); });
    if (block_sync_it == main_loop->body().exprs().rend()) {
      // If there's no sync, i.e. no tensor needs cross
      //  thread communication. We still need a wait but
      //  it can just be anywhere in the loop. Chose to
      //  place at the end arbitrarily.
      main_loop->body().insert_after(end_of_loop_expr, cp_async_wait);
    } else {
      // If a sync has been inserted, wait needs to be placed
      //  before the sync.
      main_loop->body().insert_before(*block_sync_it, cp_async_wait);
    }
  }

 private:
  InsertionInfo& insertion_info_;
  kir::ForLoop* processed_loop_ = nullptr;
};

} // namespace

void DoubleBufferInfo::build(Fusion* fusion) {
  DoubleBufferFusionInspector inspector(fusion, *this);

  // Build double buffered loop id's
  for (auto& info : map_) {
    auto double_buffer_axis = info.second.double_buffer_axis;
    // Keeps track of which loop disjoint set has been
    //  double buffered. In index allocation, one index
    //  variable would need to be allocated in each
    //  double buffer stage.
    concrete_double_buffered_loop_id_.insert(
        GpuLower::current()->caMap()->getConcreteMappedID(
            double_buffer_axis, IdMappingMode::LOOP));
  }
}

bool DoubleBufferInfo::isDoubleBufferedIterDomain(IterDomain* id) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
  return concrete_double_buffered_loop_id_.count(concrete_loop_id);
}

DoubleBufferInfo::TvInfo& DoubleBufferInfo::getTvInfo(const TensorView* tv) {
  TORCH_INTERNAL_ASSERT(
      tv->isDoubleBuffered() || tv->isCircularBuffered(),
      "Not a double-buffered tensor: ",
      tv->toString());
  return map_[tv];
}

void DoubleBufferInfo::setDoubleBufferAxis(
    const TensorView* tv,
    IterDomain* axis) {
  getTvInfo(tv).double_buffer_axis = axis;

  // Also validate the stage consistency with CA map.
  unsigned int stage_depth = 0;
  if (tv->isCircularBuffered()) {
    stage_depth = tv->circularBufferDepth();
  } else {
    // Double buffer is essentially
    //  circular buffer with depth 2.
    stage_depth = 2;
  }

  // Set and validate the new stage depth.
  setStageDepth(axis, stage_depth);
}

void DoubleBufferInfo::setStageDepth(IterDomain* id, unsigned int stage_depth) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);

  auto maybe_exisiting_depth_it = stage_depth_.find(concrete_loop_id);
  if (maybe_exisiting_depth_it == stage_depth_.end()) {
    stage_depth_[concrete_loop_id] = stage_depth;
  } else {
    TORCH_INTERNAL_ASSERT(
        stage_depth == maybe_exisiting_depth_it->second,
        "Unsupported multiple depth pipelining, was set to ",
        maybe_exisiting_depth_it->second,
        " by ",
        maybe_exisiting_depth_it->first->toString(),
        " and then set to ",
        stage_depth,
        " by ",
        concrete_loop_id->toString());
  }
}

IterDomain* DoubleBufferInfo::getDoubleBufferAxis(const TensorView* tv) {
  if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).double_buffer_axis;
}

unsigned int DoubleBufferInfo::getStageDepthFor(
    IterDomain* double_buffer_axis) {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      double_buffer_axis, IdMappingMode::LOOP);

  auto maybe_depth_it = stage_depth_.find(concrete_id);

  TORCH_INTERNAL_ASSERT(
      maybe_depth_it != stage_depth_.end(), "Stage depth not found");

  return maybe_depth_it->second;
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    IterDomain* axis,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto loop_it = std::find_if(loops.begin(), loops.end(), [&](const auto loop) {
    return GpuLower::current()->caMap()->areMapped(
               loop->iter_domain(), axis, IdMappingMode::EXACT) &&
        (!ignore_prologue ||
         loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Prolog);
  });

  if (loop_it != loops.end()) {
    return *loop_it;
  } else {
    return nullptr;
  }
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto axis = getDoubleBufferAxis(tv);

  if (axis == nullptr) {
    return nullptr;
  }

  return getDoubleBufferLoop(axis, loops, ignore_prologue);
}

void DoubleBufferInfo::setOriginalAllocSize(
    const TensorView* tv,
    Val* original_alloc_size) {
  getTvInfo(tv).original_alloc_size = original_alloc_size;
}

Val* DoubleBufferInfo::getOriginalAllocSize(const TensorView* tv) {
  if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).original_alloc_size;
}

std::vector<Expr*> DoubleBufferPass::run(const std::vector<Expr*>& exprs) {
  auto insertion_info = DoubleBufferLoopNestInspector::run(exprs);
  return DoubleBufferInserter::run(exprs, insertion_info);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
