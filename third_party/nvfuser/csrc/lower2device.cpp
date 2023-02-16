#include <lower2device.h>

#include <ATen/cuda/CUDAContext.h>
#include <expr_simplifier.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <lower_alias_memory.h>
#include <lower_allocation.h>
#include <lower_divisible_split.h>
#include <lower_double_buffer.h>
#include <lower_expr_sort.h>
#include <lower_fusion_simplifier.h>
#include <lower_index.h>
#include <lower_insert_syncs.h>
#include <lower_instrument.h>
#include <lower_loops.h>
#include <lower_magic_zero.h>
#include <lower_misaligned_vectorization.h>
#include <lower_predicate.h>
#include <lower_replace_size.h>
#include <lower_shift.h>
#include <lower_unroll.h>
#include <lower_utils.h>
#include <lower_validation.h>
#include <lower_vectorize_welford.h>
#include <lower_warp_reduce.h>

#include <list>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

thread_local GpuLower* active_gpu_lower = nullptr; // NOLINT
namespace {

class KIRCleaner : public OptOutDispatch {
 public:
  //! Remove nop IR nodes
  static std::vector<Expr*> cleanUp(const std::vector<Expr*>& loop_nests) {
    KIRCleaner cleaner;
    std::vector<Expr*> out_loop_nests;
    for (auto loop_nest : loop_nests) {
      cleaner.handle(loop_nest);
      // No need to keep the loop nest if it's determined to be nop
      if (!cleaner.is_nop_) {
        out_loop_nests.push_back(loop_nest);
      }
    }
    return out_loop_nests;
  }

 private:
  using OptOutDispatch::handle;
  void handle(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      OptOutDispatch::handle(expr);
    } else {
      // Any non-scoping expr is not considered nop
      is_nop_ = false;
    }
  }

  void handle(kir::ForLoop* fl) final {
    auto exprs = fl->body().exprs();
    fl->body().clear();
    for (auto expr : exprs) {
      handle(expr);
      // Add the expr to the loop body only when the expr is not nop
      if (!is_nop_) {
        fl->body().push_back(expr);
      }
    }
    // The loop is nop when no expr exists in the body
    is_nop_ = fl->body().empty();
  }

  void handle(kir::IfThenElse* ite) final {
    const auto conditional = ite->predicate()->value();

    // Visit the then block
    auto then_exprs = ite->thenBody().exprs();
    ite->thenBody().clear();
    if (!conditional->isConst() || conditional->value().value()) {
      for (auto expr : then_exprs) {
        handle(expr);
        if (!is_nop_) {
          ite->thenBody().push_back(expr);
        }
      }
    }

    const bool then_nop = ite->thenBody().empty();

    // Visit the else block
    auto else_exprs = ite->elseBody().exprs();
    ite->elseBody().clear();
    if (!conditional->isConst() || !conditional->value().value()) {
      for (auto expr : else_exprs) {
        handle(expr);
        if (!is_nop_) {
          ite->elseBody().push_back(expr);
        }
      }
    }

    const bool else_nop = ite->elseBody().empty();

    // If the then block is nop but the else is not, invert the
    // conditional and move the exprs in the else block to the then
    // block.
    if (then_nop && !else_nop) {
      Bool* pred = ite->predicate()->value();
      Bool* not_pred = SimplifyingIrBuilder::notExpr(pred)->as<Bool>();
      ite->predicate()->setValue(not_pred);
      for (auto expr : ite->elseBody().exprs()) {
        ite->thenBody().push_back(expr);
      }
      ite->elseBody().clear();
    }

    // This IfThenElse is nop if both the then and else blocks are nop
    is_nop_ = then_nop && else_nop;
  }

 private:
  //! True if the last visited expr is nop
  bool is_nop_ = false;
};

// Convert bar sync to __syncthreads()
class ConvertAlignedBlockSync : kir::IrVisitor {
 public:
  static std::vector<Expr*> run(std::vector<Expr*> exprs) {
    ConvertAlignedBlockSync converter;
    converter.handle(exprs);
    return exprs;
  }

 private:
  using kir::IrVisitor::handle;

  void handle(kir::BlockSync* sync) final {
    // Inspect all the scope expressions
    for (auto expr : scope_exprs_) {
      // If predicates are thread dependent, can not use aligned sync.
      if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
        if (ite->predicate()->hasValue() &&
            getRegisterType(ite->predicate()->value()) ==
                RegisterType::GeneralPurpose) {
          return;
        }
        return;
      } else if (auto fl = dynamic_cast<kir::ForLoop*>(expr)) {
        // If the start, stop, step are not thread dependent
        //  then this for loop should be thread independent.
        if (getRegisterType(fl->start()) == RegisterType::GeneralPurpose ||
            getRegisterType(fl->stop()) == RegisterType::GeneralPurpose ||
            getRegisterType(fl->step()) == RegisterType::GeneralPurpose) {
          return;
        }
      }
    }

    // If all the checks above pass, convert this sync
    //  to aligned sync.
    sync->convertToAligned();
  }
};

std::vector<Expr*> convertAlignedBlockSync(std::vector<Expr*> exprs) {
  return ConvertAlignedBlockSync::run(exprs);
}

} // namespace

void GpuLower::collectPaddedParallelDims() {
  bool can_be_single_warp = true;

  auto warp_size = at::cuda::warp_size();

  auto used_vals = fusion_->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->domain()->domain()) {
      if (tv->definition()) {
        // TODO: Support GroupedReductionOp
        if (auto reduction = dynamic_cast<ReductionOp*>(tv->definition())) {
          if (ir_utils::getMaybeWarpReductionDim(
                  reduction->out(), reduction->in())
                  .has_value()) {
            warp_pad_info_.has_warp_reduction = true;
          }
        }
      }

      // Check ifi TIDx is padded in this kernel
      if (id->hasPaddingToMultipleOfWarp()) {
        TORCH_INTERNAL_ASSERT(
            id->getParallelType() == ParallelType::TIDx,
            "Padded types supported only on TIDx");
        warp_pad_info_.is_tidx_padded = true;
      }

      // Check all possible bindings of TIDx to see
      //  if TIDx will eventually be bound to a single warp.
      if (id->getParallelType() == ParallelType::TIDx) {
        auto size_after_padding = id->getMaybeSizeAfterPadding();
        bool padding_to_single_warp = size_after_padding.has_value() &&
            size_after_padding.value() == warp_size;

        if (id->extent()->isConstInt() &&
            id->extent()->evaluateInt() > warp_size &&
            !padding_to_single_warp) {
          // If we see any other TIDx binding that's larger than
          //  a warp or unknown, we shouldn't lower warp reduce
          //  to a single warp type.
          can_be_single_warp = false;
          warp_pad_info_.is_tidx_single_warp = false;
        } else if (can_be_single_warp) {
          if (padding_to_single_warp ||
              (id->extent()->isConstInt() &&
               id->extent()->evaluateInt() == warp_size)) {
            warp_pad_info_.is_tidx_single_warp = true;
          }
        }
      }
    }
  }
}

void assignRNGOffset(Fusion* fusion) {
  int counter = 0;
  for (auto expr : fusion->exprs()) {
    if (expr->isA<RNGOp>()) {
      auto rop = expr->as<RNGOp>();
      rop->setRNGOffset(counter++);
    }
  }
}

// Dump expr string if enable lower_verbose
void dumpExprsIfEnabled(
    const std::vector<Expr*>& exprs,
    std::string pass_name,
    bool force_enable = false) {
  auto enabled_by_env = [&pass_name]() {
    if (!isDebugDumpEnabled(DebugDumpOption::LowerVerbose)) {
      return false;
    }
    const auto& args = getDebugDumpArguments(DebugDumpOption::LowerVerbose);
    return (
        args.empty() ||
        std::find(args.begin(), args.end(), pass_name) != args.end());
  };
  if (force_enable || enabled_by_env()) {
    std::cout << "After " << pass_name << ":" << std::endl;
    for (auto exp : exprs) {
      std::cout << exp->toString() << std::endl;
    }
  }
}

void GpuLower::lower(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::lower");
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  TORCH_INTERNAL_ASSERT(
      active_gpu_lower == nullptr, "Nested lowering passes are not supported");

  struct LowerGuard {
    LowerGuard(GpuLower* gpu_lower) {
      active_gpu_lower = gpu_lower;
    }
    ~LowerGuard() {
      active_gpu_lower = nullptr;
    }
  } lower_guard(this);
  // Copy fusion into a new kernel for processing
  kernel_ = std::make_unique<kir::Kernel>(fusion, cparams_.index_type);
  // Alias the fusion kernel caries around as a view of itself.
  fusion_ = kernel_.get();

  // Convert tensor views of DataType::Index type to either Int or Int32
  for (auto tv : ir_utils::allTvs(fusion_)) {
    if (tv->dtype() == DataType::Index) {
      tv->resolveIndexDtype();
    }
  }
  assignRNGOffset(fusion_);

  FusionGuard fg(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "initialize lowering");

  // prepare for lowering
  validateIr(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateIr");

  // Checks if any TIDx dim is marked as padded to a warp. Also checks if we can
  // determine the padding is explicitly a single warp.
  collectPaddedParallelDims();
  dumpExprsIfEnabled(fusion_->exprs(), "collectPaddedParallelDims");

  // Replaces integers that are tensor sizes by named scalars as "T0.size[0]"
  replaceSymbolicSizes(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "replaceSymbolicSizes");

  // Build what's refered to as the compute at map. This map contains the
  // mappings of all iteration domains across the fusion. There are three types
  // of mappings Permissive, Exact, and Loop, see compute_at_map.h/cpp for more
  // information.
  compute_at_map_ = std::make_shared<ComputeAtMap>(fusion_);

  resolveComputeWith(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "resolveComputeWith");

  if (isDebugDumpEnabled(DebugDumpOption::ComputeAtMap)) {
    std::cout << compute_at_map_->toString() << std::endl;
  }
  compute_at_map_->validateAndPropagatePType();
  dumpExprsIfEnabled(fusion_->exprs(), "validateAndPropagatePType");

  // Uses compute_at_map, find all splits that are enforced to be divisible
  divisible_splits_ = getAllDivisibleSplits(fusion_, compute_at_map_.get());
  dumpExprsIfEnabled(fusion_->exprs(), "getAllDivisibleSplits");

  // Used in parallel dimension map
  concretized_broadcast_domains_ =
      std::make_shared<const ConcretizedBroadcastDomains>(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build ConcretizedBroadcastDomains");

  parallelDimensionMap().build(fusion_);
  if (isDebugDumpEnabled(DebugDumpOption::ParallelDimensions)) {
    std::cout << "Parallel dimension map:" << std::endl;
    std::cout << parallel_dimension_map_.toString() << std::endl;
  }
  dumpExprsIfEnabled(fusion_->exprs(), "build parallelDimensionMap");

  // Validate mma data format and compatibility if any on the fusion.
  validateMma(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateMma");

  // Validate swizzle usage on the fusion schedule.
  validateSwizzle(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateSwizzle");

  // Compute thread predicates. Depends on parallel_dimension_map_
  thread_pred_map_.build(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build thread_pred_map_");

  // Fuse cetain patterns of reductions, such as a grid reduction
  // followed by a grid broadcast. Only depends on parallelization and
  // thread predicate map.
  fuseReductionsAndBroadcasts(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "fuseReductionsAndBroadcasts");

  // Scan the whole fusion and build mappings about halo extensions of
  // all IterDomains
  halo_info_ = std::make_shared<HaloInfo>(fusion_, compute_at_map_);
  dumpExprsIfEnabled(fusion_->exprs(), "build HaloInfo");

  // Want to run this after parallel map and halo info map are
  // created. vectorized_accesses_ and vectorized_set_info_ are filled.
  validateAndCollectVectorizeInfo(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateAndCollectVectorizeInfo");

  // Depends on ComputeAtMap and HaloInfo.
  validateAndConvertIterDomainGrouping(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateAndConvertIterDomainGrouping");

  // Assumes all grouped reductions are convered to
  // GroupedReductionOp, which is done by
  // validateAndConvertIterDomainGrouping
  validateGroupedReductions(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateGroupedReductions");

  // all of the lookup TVs are fusion inputs
  validateLookupTV(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateLookupTV");

  // Depends on thread_pred_map_, validates parallelization collects which
  // tensor views need WAR or RAW syncs
  sync_map_ = std::make_shared<const SyncMap>(fusion_);
  if (isDebugDumpEnabled(DebugDumpOption::SyncMap)) {
    std::cout << sync_map_->toString() << std::endl;
  }
  dumpExprsIfEnabled(fusion_->exprs(), "SyncMap");

  partialSplitMap().build(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build partialSplitMap");

  validatePartialSplit(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validatePartialSplit");

  nonDivisibleSplitInfo().build(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build nonDivisibleSplitInfo");

  // Detects all exprssions that don't need predicates. Depends on
  // nonDivisibleSplitInfo.
  pred_elimination_ = std::make_unique<PredicateElimination>(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build predicateElimination");

  doubleBufferInfo().build(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build doubleBufferInfo");

  compute_at_map_->allocateIndexVariables();
  dumpExprsIfEnabled(fusion_->exprs(), "allocateIndexVariables");
  // Run our passes keeping the lowered expressions and forwarding
  // them

  // Reorder expressions for loop-nest generation respecting computeAt
  // relationships
  const auto exprs_sorted = reorderExprsForComputeAt();
  dumpExprsIfEnabled(exprs_sorted, "reorderExprsForComputeAt");

  // Generate loop-nests and place each expression at its
  // corresponding loop
  const auto exprs_lowered = LoopNestGenerator::loweredExprs(exprs_sorted);
  dumpExprsIfEnabled(exprs_lowered, "LoopNestGenerator");

  // Replace squeezes, Transpose, Shift, Gather, and View ops with
  // unary ops since they're not separately processed in lowering.
  const auto exprs_unary_replaced = unarySetOpInserter(exprs_lowered);
  dumpExprsIfEnabled(exprs_unary_replaced, "unarySetOpInserter");

  // Insert allocations
  const auto exprs_alloced = insertAllocations(exprs_unary_replaced);
  dumpExprsIfEnabled(exprs_alloced, "insertAllocations");

  // Insert read after write smem syncs
  const auto exprs_raw_sync = insertRawThreadSynchronization(exprs_alloced);
  dumpExprsIfEnabled(exprs_raw_sync, "insertRawThreadSynchronization");

  // Reuse memory locations
  const auto exprs_reuse_mem = reuseMemoryAllocations(exprs_raw_sync);
  dumpExprsIfEnabled(exprs_reuse_mem, "reuseMemoryAllocations");

  // Insert SyncThreads at end of for-loop to avoid WAR race condition
  const auto exprs_war_sync = insertWarThreadSynchronization(exprs_reuse_mem);
  dumpExprsIfEnabled(exprs_war_sync, "insertWarThreadSynchronization");

  const auto exprs_double_buffered = DoubleBufferPass::run(exprs_war_sync);
  dumpExprsIfEnabled(exprs_double_buffered, "DoubleBufferPass");

  // This pass inserts predicates as well as branches in the code. Up until now
  // the code is explicitly single shot for loop based. Need to be careful in
  // later passes when doing any kind of insertions in loop nest structure as
  // insertions could be on if then or else instead of directly on a for loop.
  const auto exprs_unrolled_loops =
      UnrollPass::runPass(fusion_, exprs_double_buffered);
  dumpExprsIfEnabled(exprs_unrolled_loops, "UnrollPass");

  commonScalarMap().initialize(exprs_unrolled_loops);

  const auto exprs_unrolled_mv_loops =
      processMisalignedVectorization(exprs_unrolled_loops);
  dumpExprsIfEnabled(exprs_unrolled_mv_loops, "processMisalignedVectorization");

  const auto exprs_indexed_loops =
      IndexLowering::getIndexedExprs(exprs_unrolled_mv_loops);
  dumpExprsIfEnabled(exprs_indexed_loops, "IndexLowering");

  // TODO: It seems this type of optimization would be far easier to implement
  // on fusion ir than kernel ir. We should likely refactor this to at least run
  // before allocation insertion.
  const auto exprs_with_fused_broadcast = fuseWarpReduce(exprs_indexed_loops);
  dumpExprsIfEnabled(exprs_with_fused_broadcast, "fuseWarpReduce");

  const auto exprs_conditional_loops =
      generateConditionalFromPredicate(exprs_with_fused_broadcast);
  dumpExprsIfEnabled(
      exprs_conditional_loops, "generateConditionalFromPredicate");

  const auto exprs_common_index_allocated =
      allocateCommonScalars(exprs_conditional_loops);
  dumpExprsIfEnabled(exprs_common_index_allocated, "allocateCommonScalars");

  std::vector<Expr*> exprs_welford_vectorized;
  if (!isOptionDisabled(DisableOption::WelfordVectorization)) {
    exprs_welford_vectorized = vectorizeWelford(exprs_common_index_allocated);
    dumpExprsIfEnabled(exprs_welford_vectorized, "vectorizeWelford");
  } else {
    exprs_welford_vectorized = exprs_common_index_allocated;
  }

  std::vector<Expr*> exprs_register_adjusted;
  if (isNvFuserZeroEnabled()) {
    // Insert fake zero updates to make sure nvrtc doesn't blow out register use
    // on index and predicate reuse
    exprs_register_adjusted = insertMagicZero(exprs_welford_vectorized);
    dumpExprsIfEnabled(exprs_register_adjusted, "insertMagicZero");
  } else {
    exprs_register_adjusted = exprs_welford_vectorized;
  }

  const auto exprs_cleaned_up_loops =
      KIRCleaner::cleanUp(exprs_register_adjusted);
  dumpExprsIfEnabled(exprs_cleaned_up_loops, "KIRCleaner");

  const auto exprs_instrumented = instrumentKernel(exprs_cleaned_up_loops);
  dumpExprsIfEnabled(exprs_instrumented, "instrumentKernel");

  const auto exprs_sync_aligned = convertAlignedBlockSync(exprs_instrumented);
  dumpExprsIfEnabled(exprs_sync_aligned, "convertAlignedBlockSync");

  // We now have the lowered expressions, finalize the kernel IR. This function
  // will also copy over some relevant information for code generation from
  // GpuLower.
  kernel_->finalize(exprs_sync_aligned);
}

kir::Kernel* GpuLower::kernel() const {
  TORCH_CHECK(kernel_);
  return kernel_.get();
}

GpuLower* GpuLower::current() {
  TORCH_INTERNAL_ASSERT(
      active_gpu_lower != nullptr, "No active GpuLower available");
  return active_gpu_lower;
}

bool GpuLower::hasCurrent() {
  return active_gpu_lower != nullptr;
}

void GpuLower::propagateExprInfo(const Expr* old_expr, const Expr* new_expr) {
  predicateElimination().propagateRemovalInfo(old_expr, new_expr);
  if (old_expr->isA<kir::Allocate>()) {
    auto alloc_info_it =
        localAllocationInfoMap().find(old_expr->as<kir::Allocate>());
    if (alloc_info_it != localAllocationInfoMap().end()) {
      auto alloc_info =
          std::make_unique<LocalAllocationInfo>(*(alloc_info_it->second));
      localAllocationInfoMap().emplace(
          new_expr->as<kir::Allocate>(), std::move(alloc_info));
    }
  }
}

bool GpuLower::resolveComputeWith(Fusion* fusion) {
  std::vector<Expr*> exprs_sorted;

  bool updated = false;
  for (auto val : fusion->usedMathVals()) {
    auto tv = dynamic_cast<TensorView*>(val);
    if (tv == nullptr) {
      continue;
    }
    if (tv->hasComputeWith()) {
      if (exprs_sorted.empty()) {
        exprs_sorted = reorderExprsForComputeAt();
      }
      if (tv->resolveComputeWith(exprs_sorted)) {
        updated = true;
        compute_at_map_->updateComputeWith(tv);
      }
    }
  }

  return updated;
}

} // namespace nvfuser
