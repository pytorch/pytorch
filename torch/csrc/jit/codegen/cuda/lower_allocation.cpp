#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_allocation.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class AllocationInserter : public kir::MutableIrVisitor {
 private:
  struct AllocationInformation {
    // The for loop that the allocation must be placed in, nullptr if not within
    // a loop
    kir::ForLoop* for_loop = nullptr;

    // The expression that this allocation must be placed before
    kir::Expr* place_before = nullptr;

    // The allocation position relative to buffer
    size_t alloc_pos = 0;

    // The buffer this allocation is for
    kir::TensorView* buffer = nullptr;

    // The allocation expression
    kir::Allocate* alloc_expr = nullptr;

    // Initialization
    kir::Expr* init_expr = nullptr;
  };

  // Find allocation point
  void findAllocationPosition(AllocationInformation& info, kir::Expr* expr) {
    size_t alloc_pos = 0;
    kir::ForLoop* for_loop = nullptr;
    auto fuser_tv = info.buffer->fuserTv();
    size_t fl_idx_next = 0;

    for (auto fl : for_loops) {
      if (alloc_pos == fuser_tv->getComputeAtPosition()) {
        break;
      }

      if (fuser_tv->axis(alloc_pos)->isReduction()) {
        const auto outputs =
            FusionGuard::getCurFusion()->getTerminatingOutputs();
        TORCH_INTERNAL_ASSERT(
            std::find(outputs.begin(), outputs.end(), fuser_tv) !=
                outputs.end(),
            "Invalid computeAt of T",
            fuser_tv->name(),
            ". A reducation axis is detected within computeAt axes even though it is not an output tensor.");
        break;
      }

      auto fl_id = fl->iter_domain();

      if (fl_id->parallelType() == ParallelType::Unroll) {
        break;
      }

      auto local_id = gpu_lower->lowerValue(fuser_tv->axis(alloc_pos))
                          ->as<kir::IterDomain>();

      if (gpu_lower->caLoopMap().areMapped(local_id, fl_id)) {
        alloc_pos++;
      }

      for_loop = fl;
      ++fl_idx_next;
    }

    info.alloc_pos = alloc_pos;
    info.for_loop = for_loop;

    if (info.for_loop == nullptr) {
      info.place_before = for_loops.size() > 0 ? for_loops[0] : expr;
    } else {
      if (info.for_loop == for_loops.back()) {
        // Inline allocation, place before expr
        info.place_before = expr;
      } else {
        // Place allocation after the last computeAt axis
        // TODO: may be more efficient to place before the first non-computeAt
        // axis
        info.place_before = for_loops.at(fl_idx_next);
      }
    }
  }

  // Create initialization expression if init_val is non-null.
  void createInitExpr(AllocationInformation& info, kir::Val* init_val) {
    if (init_val == nullptr) {
      info.init_expr = nullptr;
      return;
    }

    auto fuser_tv = info.buffer->fuserTv();

    std::vector<kir::IterDomain*> init_dims;
    for (size_t axis_i = info.alloc_pos; axis_i < fuser_tv->nDims(); axis_i++) {
      if (info.buffer->fuserTv()->axis(axis_i)->isReduction() ||
          info.buffer->fuserTv()->axis(axis_i)->isBroadcast()) {
        continue;
      }
      auto concrete_id =
          gpu_lower
              ->lowerValue(gpu_lower->caParallelMap().getConcreteMappedID(
                  fuser_tv->axis(axis_i)))
              ->as<kir::IterDomain>();
      init_dims.push_back(concrete_id);
    }
    kir::Expr* init_expr = ir_builder.create<kir::UnaryOp>(
        UnaryOpType::Set, info.buffer, init_val);
    for (auto init_loop_it = init_dims.rbegin();
         init_loop_it != init_dims.rend();
         ++init_loop_it) {
      auto id = *init_loop_it;
      kir::ForLoop* new_loop = nullptr;
      auto extent_with_halo = gpu_lower->haloInfo().getExtent(id);
      if (extent_with_halo) {
        new_loop = ir_builder.create<kir::ForLoop>(
            id,
            ir_builder.create<kir::Int>(c10::nullopt),
            nullptr,
            extent_with_halo,
            nullptr,
            false,
            nullptr);
      } else {
        new_loop = ir_builder.create<kir::ForLoop>(id);
      }
      new_loop->body().push_back(init_expr);
      init_expr = new_loop;
    }
    info.init_expr = init_expr;
  }

  std::vector<kir::Val*> getGlobalAllocationSizes(AllocationInformation& info) {
    const auto& domain = info.buffer->domain();
    const auto& maybe_rfactor_domain =
        domain->hasRFactor() ? domain->rfactorDomain() : domain->rootDomain();

    std::vector<kir::Val*> alloc_dims;

    for (const auto id : maybe_rfactor_domain) {
      if (id->isReduction() ||
          id->iterType() == IterType::BroadcastWithoutStride) {
        continue;
      }
      auto extent = id->extent();
      // Use halo-extended extent if found
      auto halo_extent = gpu_lower->haloInfo().getRootAxisInfo(id);
      if (halo_extent.hasHalo()) {
        extent = ir_builder.addExpr(extent, halo_extent.width());
      }
      alloc_dims.push_back(extent);
    }

    return alloc_dims;
  }

  // Get allocation extents of root axes with halo
  //
  // Allocation can be done with leaf IDs with halo as well, but
  // allocation size could be larger than necessary.
  //
  // For example, suppose the shift offset of an axis is 1. When it is
  // split by N, the halo size of the inner output is N+1. When the
  // allocation only has the inner split output, the allocation size
  // would be N+1. Suppose that ID is further split by M, the output
  // extents would be N/M and M+1. The allocation size based on the
  // leaves would be N/M*(M+1) or N+N/M, which is larger than N+1.
  //
  // This function tries to propagate back halo informatin to root
  // axes to avoid inflating allocations. It fails when merged domains
  // are split and only one of the split outputs is used for
  // allocations since in such a case we can't un-merge and properly
  // determine the extents of the merge inputs. Currently, that
  // results in an exception, but it may be more reasonable to simply
  // fall back to the leaf-based allocation.
  //
  // See the FusionShiftDoubleSplit test for an example case.
  std::vector<kir::Val*> getNonGlobalAllocExprWithHalo(
      TensorView* tv,
      const std::vector<IterDomain*>& alloc_domains) {
    std::vector<Val*> start_vals;
    std::transform(
        alloc_domains.begin(),
        alloc_domains.end(),
        std::back_inserter(start_vals),
        [](IterDomain* dom) { return dom->as<Val>(); });

    // Get all exprs involved in generating the allocation IDs
    auto exprs = ExprSort::getExprs(tv->fusion(), start_vals);

    // Get the halo extent if found
    auto getExtent = [this](IterDomain* id) {
      auto extent = gpu_lower->haloInfo().getExtent(id);
      if (extent == nullptr) {
        extent = gpu_lower->lowerValue(id->extent());
      }
      return extent;
    };

    std::unordered_map<IterDomain*, kir::Val*> known_extents;

    // IterDomains that are allocated fully. For example, if an ID is
    // split and only one of them is used for allocation, that's not
    // considered full. Only full domains can be unmerged, which is
    // needed to propagate back the halo information to root domains.
    std::unordered_set<IterDomain*> full_domains;

    for (auto alloc_domain : alloc_domains) {
      known_extents.insert({alloc_domain, getExtent(alloc_domain)});
      full_domains.insert(alloc_domain);
    }

    for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
      auto expr = *it;
      if (auto merge = dynamic_cast<Merge*>(expr)) {
        auto out_it = known_extents.find(merge->out());
        // If nothing is know about the out id, no propagation can be
        // done. Note that's not necessarily an error.
        if (out_it == known_extents.end()) {
          continue;
        }
        // Similarly, if the extent of the out id is not full extent,
        // we can't un-merge it.
        if (full_domains.find(merge->out()) == full_domains.end()) {
          continue;
        }
        // Since the extent of the out id is full, the extent of each
        // of the input axes is also full
        known_extents.insert({merge->inner(), getExtent(merge->inner())});
        full_domains.insert(merge->inner());
        known_extents.insert({merge->outer(), getExtent(merge->outer())});
        full_domains.insert(merge->outer());
        known_extents.erase(out_it);
      } else if (auto split = dynamic_cast<Split*>(expr)) {
        auto inner = split->inner();
        const auto inner_it = known_extents.find(inner);
        auto outer = split->outer();
        const auto outer_it = known_extents.find(outer);
        if (inner_it != known_extents.end() &&
            outer_it != known_extents.end()) {
          if (full_domains.find(inner) != full_domains.end() &&
              full_domains.find(outer) != full_domains.end()) {
            known_extents.insert({split->in(), getExtent(split->in())});
            full_domains.insert(split->in());
          } else {
            known_extents.insert(
                {split->in(),
                 ir_builder.mulExpr(outer_it->second, inner_it->second)});
          }
          known_extents.erase(inner_it);
          known_extents.erase(outer_it);
        } else if (inner_it != known_extents.end()) {
          known_extents.insert({split->in(), inner_it->second});
          known_extents.erase(inner_it);
        } else if (outer_it != known_extents.end()) {
          known_extents.insert({split->in(), outer_it->second});
          known_extents.erase(outer_it);
        }
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unexpected expr: ", expr);
      }
    }

    std::vector<kir::Val*> alloc_dims;

    for (auto root_axis : tv->getRootDomain()) {
      auto it = known_extents.find(root_axis);
      if (it == known_extents.end()) {
        continue;
      }
      alloc_dims.push_back(it->second);
      known_extents.erase(it);
    }

    // known_extents should have only mappings for root axes, so
    // if anything remains in the map, it's an error
    if (!known_extents.empty()) {
      std::stringstream ss;
      for (auto kv : known_extents) {
        ss << kv.first << " ";
      }
      TORCH_INTERNAL_ASSERT(
          false, "Non-root axes found for TV", tv->name(), ": ", ss.str());
    }

    return alloc_dims;
  }

  std::vector<kir::Val*> getNonGlobalAllocExpr(AllocationInformation& info) {
    auto fuser_tv = info.buffer->fuserTv();
    const auto memory_type = info.buffer->memoryType();
    TORCH_INTERNAL_ASSERT(
        memory_type != MemoryType::Global,
        "Invalid memory type: ",
        memory_type);

    std::vector<kir::Val*> alloc_dims;

    bool has_halo = false;
    std::vector<IterDomain*> alloc_domains;

    for (size_t axis_i = 0; axis_i < fuser_tv->nDims(); axis_i++) {
      const auto local_id =
          gpu_lower->lowerValue(fuser_tv->axis(axis_i))->as<kir::IterDomain>();

      if (
          // If we're reducing this dimension, don't use it in the allocation
          // computation
          local_id->isReduction() ||
          // If this is a broadcast dimension, don't use it in the allocation
          // computation
          local_id->isBroadcast()) {
        continue;
      }

      auto concrete_id =
          gpu_lower
              ->lowerValue(gpu_lower->caParallelMap().getConcreteMappedID(
                  fuser_tv->axis(axis_i)))
              ->as<kir::IterDomain>();
      const bool is_block_dim =
          isParallelTypeBlockDim(concrete_id->parallelType());
      const bool is_thread_dim =
          isParallelTypeThreadDim(concrete_id->parallelType());
      const bool is_thread = isParallelTypeThread(concrete_id->parallelType());

      if (axis_i < info.alloc_pos) {
        // Even when the axis is outside the allocation position, if the
        // tensor is shared with respect to the axis, the buffer size
        // needs to be expanded for the axis. Sharing occurs in two
        // cases: 1) the tensor is on shared memory with the axis
        // parallelized by TIDs, and 2) the tensor is on global memory
        // with the axis parallelized by TIDs or BIDs.
        if (!((memory_type == MemoryType::Shared && is_thread_dim) ||
              (memory_type == MemoryType::Global && is_thread))) {
          continue;
        }
        alloc_domains.push_back(fuser_tv->axis(axis_i));
      } else {
        if (
            // If shared memory, don't use any IDs bound to a grid dimension
            (memory_type == MemoryType::Shared && is_block_dim) ||
            // If local memory, don't use any IDs bound to a grid or block
            // dimension
            (memory_type == MemoryType::Local && is_thread)) {
          continue;
        }
        alloc_domains.push_back(fuser_tv->axis(axis_i));
      }

      auto extent = concrete_id->extent();

      if (gpu_lower->haloInfo().getExtent(fuser_tv->axis(axis_i)) != nullptr) {
        has_halo = true;
      }

      alloc_dims.push_back(extent);
    }

    // When an axis with halo extension is detected, propagate back
    // the halo extents from leaf IDs to root IDs
    if (has_halo) {
      return getNonGlobalAllocExprWithHalo(fuser_tv, alloc_domains);
    }

    return alloc_dims;
  }

  void createAllocExpr(AllocationInformation& info, bool is_output) {
    if (is_output) {
      info.alloc_expr = nullptr;
      return;
    }

    std::vector<kir::Val*> alloc_dims;
    const MemoryType memory_type = info.buffer->memoryType();

    if (memory_type == MemoryType::Global) {
      alloc_dims = getGlobalAllocationSizes(info);
    } else {
      alloc_dims = getNonGlobalAllocExpr(info);
    }

    if (alloc_dims.size() == 0 &&
        info.buffer->domain()->noReductions().size() != 0) {
      alloc_dims.push_back(ir_builder.create<kir::Int>(1));
    }

    // Create the allocation node
    info.alloc_expr = ir_builder.create<kir::Allocate>(
        info.buffer, info.buffer->memoryType(), alloc_dims);
  }

  void handle(kir::Expr* expr) {
    if (!ir_utils::isTVOp(expr) || expr->isA<kir::Allocate>()) {
      expr->accept(this);
      return;
    }

    // // Found where the allocation needs to be inserted

    for (auto out : expr->outputs()) {
      if (!out->isA<kir::TensorView>()) {
        continue;
      }

      auto out_tv = out->as<kir::TensorView>();
      auto default_val =
          gpu_lower->predicateElimination().getInitValue(out_tv->fuserTv());

      kir::Val* init = nullptr;
      if (expr->isA<kir::ReductionOp>() && out_tv->fuserTv()->hasReduction()) {
        TORCH_INTERNAL_ASSERT(
            default_val == nullptr,
            "Reduction should not have a default initialization value for predicate elimination.");
        init = expr->as<kir::ReductionOp>()->init();
      } else if (expr->isA<kir::WelfordOp>()) {
        TORCH_INTERNAL_ASSERT(
            default_val == nullptr,
            "Welford should not have a default initialization value for predicate elimination.");
        const auto welford = expr->as<kir::WelfordOp>();
        if (out->id() == welford->outVar()->id()) {
          init = welford->initVar() == nullptr
              ? ir_builder.create<kir::Double>(0)
              : welford->initVar();
        } else if (out->id() == welford->outAvg()->id()) {
          init = welford->initAvg() == nullptr
              ? ir_builder.create<kir::Double>(0)
              : welford->initAvg();
        } else {
          TORCH_INTERNAL_ASSERT(
              out->id() == welford->outN()->id(), "Unreachable");
          init = welford->initN();
        }
      } else if (default_val != nullptr) {
        init = default_val;
      }

      const bool is_output = gpu_lower->kernel()->isOutput(out);

      // Don't need to alloc outputs, and if we don't need to initialize we're
      // done.
      if (is_output && init == nullptr) {
        continue;
      }

      AllocationInformation allocation;
      allocation.buffer = out_tv;
      findAllocationPosition(allocation, expr);
      createAllocExpr(allocation, is_output);
      createInitExpr(allocation, init);

      allocs.push_back(allocation);
    }
  }

  void visit(kir::ForLoop* fl) final {
    for_loops.push_back(fl);
    // Modifying in place, make a copy of the vector
    const std::vector<kir::Expr*> exprs = fl->body().exprs();
    for (auto expr : exprs) {
      handle(expr);
    }
    for_loops.pop_back();
  }

  void visit(kir::IfThenElse*) final {
    TORCH_INTERNAL_ASSERT(
        false,
        "Pass does not support conditional statements, ",
        "this pass should be run before any conditionals are placed in code.");
  }

  explicit AllocationInserter(std::vector<kir::Expr*> _loop_nests)
      : loop_nests_(std::move(_loop_nests)),
        gpu_lower(GpuLower::current()),
        ir_builder(gpu_lower->kernel()) {
    // Compute all allocations
    const std::vector<kir::Expr*> exprs = loop_nests_;
    for (auto expr : exprs) {
      handle(expr);
    }

    // First, place allocations of dynamic smem tensors at the very
    // beginning of the expr list. Traverse backward as they should be
    // placed in topological order.
    for (auto it = allocs.rbegin(); it != allocs.rend(); ++it) {
      const auto& alloc = *it;
      if (alloc.alloc_expr == nullptr) {
        continue;
      }
      // Dynamic smem exprs need to be at the beginning of the kernel outside
      // for loops
      if (alloc.buffer->memoryType() == MemoryType::Shared &&
          !kir::ExpressionEvaluator::isConst(alloc.alloc_expr->size())) {
        loop_nests_.insert(loop_nests_.begin(), alloc.alloc_expr);
      }
    }

    // Place the remaining allocations.
    for (const auto& alloc : allocs) {
      if (alloc.alloc_expr == nullptr) {
        continue;
      }
      if (alloc.buffer->memoryType() == MemoryType::Shared &&
          !kir::ExpressionEvaluator::isConst(alloc.alloc_expr->size())) {
        continue;
      }
      if (alloc.for_loop == nullptr) {
        auto place_before_it = std::find(
            loop_nests_.begin(), loop_nests_.end(), alloc.place_before);
        TORCH_INTERNAL_ASSERT(
            place_before_it != loop_nests_.end(),
            "Could not figure out where to place allocation. ",
            "Use of the buffer, ",
            toString(alloc.buffer),
            ", could not be found.",
            toString(alloc.place_before));
        loop_nests_.insert(place_before_it, alloc.alloc_expr);
      } else {
        alloc.for_loop->body().insert_before(
            alloc.place_before, alloc.alloc_expr);
      }
    }

    // Now that allocations are in place, place the initializations
    for (const auto& alloc : allocs) {
      if (alloc.init_expr == nullptr) {
        continue;
      }
      if (alloc.for_loop == nullptr) {
        auto place_before_it = std::find(
            loop_nests_.begin(), loop_nests_.end(), alloc.place_before);
        // Don't need a check here as if the allocation placement succeeded
        // this will too
        loop_nests_.insert(place_before_it, alloc.init_expr);
      } else {
        alloc.for_loop->body().insert_before(
            alloc.place_before, alloc.init_expr);
      }
    }
  }

 private:
  std::deque<AllocationInformation> allocs;

  std::vector<kir::ForLoop*> for_loops;

  std::vector<kir::Expr*> loop_nests_;

  GpuLower* gpu_lower;

  kir::IrBuilder ir_builder;

 public:
  static std::vector<kir::Expr*> insert(
      const std::vector<kir::Expr*>& loop_nests) {
    AllocationInserter inserter(loop_nests);
    return inserter.loop_nests_;
  }
};

} // namespace

std::vector<kir::Expr*> insertAllocations(
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertAllocations");
  return AllocationInserter::insert(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
