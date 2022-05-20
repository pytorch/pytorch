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
      def->isA<UnaryOp>() &&
          def->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Set,
      "Invalid tensor to double-buffer. Only tensor defined by UnaryOp::Set is supported: ",
      def->toString());

  TORCH_INTERNAL_ASSERT(
      def->as<UnaryOp>()->in()->isA<TensorView>(),
      "Invalid tensor to double-buffer. Only tensor defined by UnaryOp::Set with TensorView is supported: ",
      def->toString());

  // Require the producer tensor to have been computed entirely for
  // the double-buffering loop. Otherwise, the producer itself would
  // also need to be double-bufferred.
  auto producer = def->as<UnaryOp>()->in()->as<TensorView>();
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
    if (!tv->isDoubleBuffered()) {
      return;
    }

    validateDoubleBufferedTensor(tv);

    auto db_axis = getDoubleBufferAxis(tv);

    db_info_.setDoubleBufferAxis(tv, db_axis);
  }

 private:
  DoubleBufferInfo& db_info_;
};

// The type of replicated double-buffer loops
enum class LoopType { Prologue, Main, Epilogue };

// The epilogue loop is only created when the producer of a double
// buffer tensor is on smem, in which case it would otherwise require
// an additional predicate to guard buffer overruns. When it's on
// gmem, that isn't the case, so it does not need to create an
// epilogue loop.
bool requireEpilogue(const std::vector<UnaryOp*>& exprs) {
  return std::any_of(exprs.begin(), exprs.end(), [](const UnaryOp* uop) {
    return uop->in()->as<TensorView>()->getMemoryType() == MemoryType::Shared;
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
      const std::vector<UnaryOp*>& double_buffer_load_exprs,
      LoopType loop_type) {
    DoubleBufferLoopCloner cloner(
        double_buffer_loop, double_buffer_load_exprs, loop_type);
    cloner.clone();
    return cloner.cloned_top_level_loop_;
  }

 private:
  DoubleBufferLoopCloner(
      kir::ForLoop* double_buffer_loop,
      const std::vector<UnaryOp*>& double_buffer_load_exprs,
      LoopType loop_type)
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

    auto index = IrBuilder::create<Int>(c10::nullopt);
    auto start = double_buffer_loop_->start();
    auto stop = double_buffer_loop_->stop();

    if (loop_type_ == LoopType::Prologue) {
      TORCH_INTERNAL_ASSERT(start->isZeroInt());
      stop = gpu_lower->kernel()->oneVal();
    } else if (
        loop_type_ == LoopType::Main &&
        requireEpilogue(double_buffer_load_exprs_)) {
      stop = IrBuilder::subExpr(
          double_buffer_loop_->stop(), gpu_lower->kernel()->oneVal());
    } else if (loop_type_ == LoopType::Epilogue) {
      TORCH_INTERNAL_ASSERT(requireEpilogue(double_buffer_load_exprs_));
      start = IrBuilder::subExpr(
          double_buffer_loop_->stop(), gpu_lower->kernel()->oneVal());
    }

    cloned_top_level_loop_ = IrBuilder::create<kir::ForLoop>(
        double_buffer_loop_->iter_domain(),
        index,
        start,
        stop,
        gpu_lower->kernel()->oneVal(),
        false,
        nullptr,
        double_buffer_loop_->isUnrollRequired());

    handle(double_buffer_loop_);
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

    if (loop_type_ == LoopType::Main) {
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
    if ((loop_type_ == LoopType::Prologue && is_double_buffer_load_expr) ||
        (loop_type_ == LoopType::Epilogue && !is_double_buffer_load_expr)) {
      cloned_scopes_.back()->push_back(expr);
    }
  }

 private:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<UnaryOp*>& double_buffer_load_exprs_;
  const LoopType loop_type_;

  kir::ForLoop* cloned_top_level_loop_ = nullptr;
  std::deque<kir::Scope*> cloned_scopes_;
};

using InsertionInfo = std::unordered_map<kir::ForLoop*, std::vector<UnaryOp*>>;

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

  void handle(UnaryOp* uop) final {
    const auto gpu_lower = GpuLower::current();

    auto out_tv = ir_utils::getTvOutput(uop);

    if (out_tv == nullptr) {
      return;
    }

    // Ignore init loop
    if (!out_tv->isDoubleBuffered() || !uop->in()->isA<TensorView>()) {
      return;
    }

    auto double_buffer_loop =
        gpu_lower->doubleBufferInfo().getDoubleBufferLoop(out_tv, for_loops_);

    TORCH_INTERNAL_ASSERT(
        double_buffer_loop != nullptr,
        "No double buffer loop found for a double buffered tensor: ",
        out_tv->toString());

    validateDoubleBufferLoop(double_buffer_loop);

    insertion_info_[double_buffer_loop].push_back(uop);
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
      const std::vector<UnaryOp*>& loads) {
    auto prologue_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, LoopType::Prologue);
    registerInsertBefore(double_buffer_loop, prologue_loop);

    auto write_to_smem =
        std::any_of(loads.begin(), loads.end(), [](const UnaryOp* uop) {
          return uop->out()->as<TensorView>()->getMemoryType() ==
              MemoryType::Shared;
        });

    // RAW sync is not inserted for double buffered tensors. The only
    // exception is the prologue load.
    if (write_to_smem) {
      auto sync = IrBuilder::create<kir::BlockSync>();
      registerInsertBefore(double_buffer_loop, sync);
    }

    auto main_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, LoopType::Main);
    registerReplace(double_buffer_loop, main_loop);

    if (requireEpilogue(loads)) {
      auto epilogue_loop = DoubleBufferLoopCloner::clone(
          double_buffer_loop, loads, LoopType::Epilogue);
      registerInsertAfter(double_buffer_loop, epilogue_loop);
    }
  }

 private:
  InsertionInfo& insertion_info_;
  kir::ForLoop* processed_loop_ = nullptr;
};

} // namespace

void DoubleBufferInfo::build(Fusion* fusion) {
  DoubleBufferFusionInspector inspector(fusion, *this);
}

DoubleBufferInfo::TvInfo& DoubleBufferInfo::getTvInfo(const TensorView* tv) {
  TORCH_INTERNAL_ASSERT(
      tv->isDoubleBuffered(), "Not a double-buffered tensor: ", tv->toString());
  return map_[tv];
}

void DoubleBufferInfo::setDoubleBufferAxis(
    const TensorView* tv,
    IterDomain* axis) {
  getTvInfo(tv).double_buffer_axis = axis;
}

IterDomain* DoubleBufferInfo::getDoubleBufferAxis(const TensorView* tv) {
  if (!tv->isDoubleBuffered()) {
    return nullptr;
  }

  return getTvInfo(tv).double_buffer_axis;
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    IterDomain* axis,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto loop_it = std::find_if(loops.begin(), loops.end(), [&](const auto loop) {
    return GpuLower::current()->caIndexMap().areMapped(
               loop->iter_domain(), axis) &&
        (!ignore_prologue || !loop->stop()->isOneInt());
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
  if (!tv->isDoubleBuffered()) {
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
