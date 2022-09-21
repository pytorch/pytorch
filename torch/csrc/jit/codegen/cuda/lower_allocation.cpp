#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_allocation.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class AllocationInserter : public kir::ExprMutator {
 private:
  using kir::ExprMutator::handle;

  // Expanded version of BasicAllocInfo in lower_utils.h helps to track
  // additional information
  struct AllocationInformation {
    // The for loop that the initialization of this allocation must be
    // placed in, nullptr if not within a loop
    kir::ForLoop* init_for_loop = nullptr;

    // The expression that the initialization of this allocation must
    // be placed before
    Expr* init_place_before = nullptr;

    // Keep track of the actual allocation loop. This can be different
    // from init_for_loop only with unswitched shared memory allocations,
    // which are moved outer loops to avoid duplicated allocations
    // (see issue #1133).
    kir::ForLoop* alloc_for_loop = nullptr;

    // The expression that this allocation must be placed
    // before. Similar to alloc_for_loop, this is different from
    // init_place_before only with unswitched shared memory allocations.
    Expr* alloc_place_before = nullptr;

    // The allocation position relative to buffer
    size_t alloc_pos = 0;

    // The buffer this allocation is for
    TensorView* buffer = nullptr;

    // Info to transfer to GPU lower
    bool has_halo = false;

    // Local Iterdomains that this allocation covers
    std::unique_ptr<std::vector<IterDomain*>> allocation_domains;
  };

  // Find allocation point
  // Fills info.buffer, info.alloc_pos, info.init_for_loop,
  // info.init_place_before, info.alloc_for_loop, info.alloc_place_before
  void fillAllocationInformation(AllocationInformation& info, Expr* expr) {
    auto loop_alloc_info =
        loop_utils::getAllocInformation(info.buffer, for_loops_);

    info.init_for_loop = loop_alloc_info.init_for_loop;
    info.alloc_for_loop = loop_alloc_info.alloc_for_loop;
    info.alloc_pos = loop_alloc_info.alloc_pos;

    auto next_fl = [](kir::ForLoop* fl, const std::vector<kir::ForLoop*> fls) {
      for (auto i : c10::irange(fls.size())) {
        if (fl == fls[i]) {
          if (i + 1 < fls.size()) {
            return fls[i + 1];
          }
        }
      }
      TORCH_INTERNAL_ASSERT(false, "Could not find desired loop.");
    };

    if (info.init_for_loop == nullptr) {
      info.init_place_before = for_loops_.size() > 0 ? for_loops_[0] : expr;
    } else {
      if (info.init_for_loop == for_loops_.back()) {
        // Inline allocation, place before expr
        info.init_place_before = expr;
      } else {
        // Place allocation after the last computeAt axis
        // TODO: may be more efficient to place before the first non-computeAt
        // axis
        info.init_place_before = next_fl(info.init_for_loop, for_loops_);
      }
    }

    // Set the allocation loop and the place_before expression in the
    // same way as the initialization loop and place_before expression
    if (info.alloc_for_loop == info.init_for_loop) {
      info.alloc_for_loop = info.init_for_loop;
      info.alloc_place_before = info.init_place_before;
    } else {
      if (info.alloc_for_loop == nullptr) {
        info.alloc_place_before = for_loops_.size() > 0 ? for_loops_[0] : expr;
      } else {
        // Since there must be an inner unswitched domain,
        // alloc_for_loop should never be the inner-most loop.
        TORCH_INTERNAL_ASSERT(info.alloc_for_loop != for_loops_.back());
        info.alloc_place_before = next_fl(info.alloc_for_loop, for_loops_);
      }
    }
  }

  // Create initialization expression if init_val is non-null.
  Expr* createInitExpr(AllocationInformation& info, Val* init_val) {
    if (init_val == nullptr) {
      return nullptr;
    }

    std::vector<IterDomain*> init_dims;
    for (const auto axis_i :
         c10::irange(info.alloc_pos, info.buffer->nDims())) {
      if (info.buffer->axis(axis_i)->isReduction() ||
          info.buffer->axis(axis_i)->isBroadcast()) {
        continue;
      }
      auto concrete_id = gpu_lower->caMap()->getConcreteMappedID(
          info.buffer->axis(axis_i), IdMappingMode::LOOP);
      init_dims.push_back(concrete_id);
    }
    Expr* init_expr =
        IrBuilder::create<UnaryOp>(UnaryOpType::Set, info.buffer, init_val);
    for (auto init_loop_it = init_dims.rbegin();
         init_loop_it != init_dims.rend();
         ++init_loop_it) {
      auto id = *init_loop_it;
      kir::ForLoop* new_loop = nullptr;
      auto extent_with_halo = gpu_lower->haloInfo().getExtent(id);
      if (extent_with_halo) {
        new_loop = IrBuilder::create<kir::ForLoop>(
            id,
            IrBuilder::create<Int>(c10::nullopt),
            nullptr,
            extent_with_halo,
            nullptr,
            false,
            nullptr,
            false,
            DoubleBufferLoopStage::NotApplicable);
      } else {
        new_loop = IrBuilder::create<kir::ForLoop>(id);
      }
      new_loop->body().push_back(init_expr);
      init_expr = new_loop;
    }
    return init_expr;
  }

  std::vector<Val*> getGlobalAllocationSizes(AllocationInformation& info) {
    const auto& domain = info.buffer->domain();
    const auto& maybe_rfactor_domain = domain->hasRFactor()
        ? domain->getRFactorDomain()
        : domain->getRootDomain();

    std::vector<Val*> alloc_dims;

    for (const auto id : maybe_rfactor_domain) {
      if (id->isReduction() || id->isStride() || id->isBroadcast()) {
        continue;
      }
      auto extent = id->extent();
      // Use halo-extended extent if found
      auto halo_extent = gpu_lower->haloInfo().getRootAxisInfo(id);
      if (halo_extent.hasHalo()) {
        extent = IrBuilder::addExpr(
            extent, IrBuilder::create<Int>(halo_extent.width()));
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
  std::vector<Val*> getNonGlobalAllocExprWithHalo(
      TensorView* tv,
      const std::vector<IterDomain*>& alloc_domains) {
    std::vector<Val*> start_vals;
    std::transform(
        alloc_domains.begin(),
        alloc_domains.end(),
        std::back_inserter(start_vals),
        [](IterDomain* dom) { return dom->as<Val>(); });

    // Get all exprs involved in generating the allocation IDs
    auto exprs = StmtSort::getExprs(tv->fusion(), start_vals);

    // Get the halo extent if found
    auto getExtent = [this](IterDomain* id) {
      auto extent = gpu_lower->haloInfo().getExtent(id);
      if (extent == nullptr) {
        extent = id->extent();
      }
      return extent;
    };

    std::unordered_map<IterDomain*, Val*> known_extents;

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
                 IrBuilder::mulExpr(outer_it->second, inner_it->second)});
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

    std::vector<Val*> alloc_dims;

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

  std::vector<Val*> getNonGlobalAllocExpr(AllocationInformation& info) {
    const auto memory_type = info.buffer->getMemoryType();
    TORCH_INTERNAL_ASSERT(
        memory_type != MemoryType::Global,
        "Invalid memory type: ",
        memory_type);

    std::vector<Val*> alloc_dims;

    bool has_halo = false;
    std::vector<IterDomain*> alloc_domains;

    info.allocation_domains = std::make_unique<std::vector<IterDomain*>>();

    for (const auto axis_i : c10::irange(info.buffer->nDims())) {
      const auto local_id = info.buffer->axis(axis_i);

      // Don't use reduction/stride/broadcast axis in the allocation
      // computation
      if (local_id->isReduction() || local_id->isStride() ||
          local_id->isBroadcast()) {
        continue;
      }

      auto concrete_id = gpu_lower->caMap()->getConcreteMappedID(
          info.buffer->axis(axis_i), IdMappingMode::LOOP);
      const bool is_block_dim =
          isParallelTypeBlockDim(concrete_id->getParallelType());
      const bool is_thread_dim =
          isParallelTypeThreadDim(concrete_id->getParallelType());
      const bool is_thread =
          isParallelTypeThread(concrete_id->getParallelType());

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
        alloc_domains.push_back(info.buffer->axis(axis_i));
      } else {
        if (
            // If shared memory, don't use any IDs bound to a grid dimension
            (memory_type == MemoryType::Shared && is_block_dim) ||
            // If local memory, don't use any IDs bound to a grid or block
            // dimension
            (memory_type == MemoryType::Local && is_thread)) {
          continue;
        }
        alloc_domains.push_back(info.buffer->axis(axis_i));
      }

      auto extent = concrete_id->extent();

      if (gpu_lower->haloInfo().getExtent(info.buffer->axis(axis_i)) !=
          nullptr) {
        has_halo = true;
      }

      alloc_dims.push_back(extent);
      info.allocation_domains->push_back(local_id);
    }

    // When an axis with halo extension is detected, propagate back
    // the halo extents from leaf IDs to root IDs
    if (has_halo) {
      info.has_halo = true;
      return getNonGlobalAllocExprWithHalo(info.buffer, alloc_domains);
    }

    return alloc_dims;
  }

  kir::Allocate* createAllocExpr(AllocationInformation& info, bool is_output) {
    if (is_output) {
      return nullptr;
    }

    std::vector<Val*> alloc_dims;
    const MemoryType memory_type = info.buffer->getMemoryType();

    if (memory_type == MemoryType::Global) {
      alloc_dims = getGlobalAllocationSizes(info);
    } else {
      alloc_dims = getNonGlobalAllocExpr(info);
    }

    if (alloc_dims.size() == 0 &&
        info.buffer->domain()->noReductions().size() != 0) {
      alloc_dims.push_back(info.buffer->container()->oneVal());
    }

    // Double the allocation size if double-buffered. Record the
    // original size for indexing.
    if (info.buffer->isDoubleBuffered() || info.buffer->isCircularBuffered()) {
      Val* original_alloc_size = nullptr;
      for (auto alloc_dim : alloc_dims) {
        if (original_alloc_size == nullptr) {
          original_alloc_size = alloc_dim;
        } else {
          original_alloc_size =
              IrBuilder::mulExpr(original_alloc_size, alloc_dim);
        }
      }
      GpuLower::current()->doubleBufferInfo().setOriginalAllocSize(
          info.buffer, original_alloc_size);
      int double_buffer_stage = 2;
      if (info.buffer->isCircularBuffered()) {
        double_buffer_stage = info.buffer->circularBufferDepth();
      }
      alloc_dims.push_back(IrBuilder::create<Int>(double_buffer_stage));
    }

    // Create the allocation node
    return IrBuilder::create<kir::Allocate>(
        info.buffer, info.buffer->getMemoryType(), alloc_dims);
  }

  void handle(Expr* expr) override {
    if (!ir_utils::isTvOp(expr) || expr->isA<kir::Allocate>()) {
      ExprMutator::handle(expr);
      return;
    }

    // // Found where the allocation needs to be inserted

    for (const auto i : c10::irange(expr->outputs().size())) {
      auto out = expr->output(i);
      if (!out->isA<TensorView>()) {
        continue;
      }

      auto out_tv = out->as<TensorView>();
      auto default_val = gpu_lower->predicateElimination().getInitValue(out_tv);

      Val* init = nullptr;
      if (expr->isA<ReductionOp>() && out_tv->hasReduction()) {
        TORCH_INTERNAL_ASSERT(
            default_val == nullptr,
            "Reduction should not have a default initialization value for predicate elimination.");
        init = expr->as<ReductionOp>()->init();
      } else if (expr->isA<GroupedReductionOp>() && out_tv->hasReduction()) {
        TORCH_INTERNAL_ASSERT(
            default_val == nullptr,
            "Reduction should not have a default initialization value for predicate elimination.");
        init = expr->as<GroupedReductionOp>()->initVal(i);
      } else if (expr->isA<MmaOp>()) {
        init = expr->as<MmaOp>()->init();
      } else if (expr->isA<WelfordOp>()) {
        TORCH_INTERNAL_ASSERT(
            default_val == nullptr,
            "Welford should not have a default initialization value for predicate elimination.");
        const auto welford = expr->as<WelfordOp>();
        if (out->name() == welford->outVar()->name()) {
          init = welford->initVar() == nullptr ? IrBuilder::create<Double>(0)
                                               : welford->initVar();
        } else if (out->name() == welford->outAvg()->name()) {
          init = welford->initAvg() == nullptr ? IrBuilder::create<Double>(0)
                                               : welford->initAvg();
        } else {
          TORCH_INTERNAL_ASSERT(
              out->name() == welford->outN()->name(), "Unreachable");
          init = welford->initN();
        }
      } else if (expr->isA<GroupedWelfordOp>()) {
        TORCH_INTERNAL_ASSERT(
            default_val == nullptr,
            "Welford should not have a default initialization value for predicate elimination.");
        init = expr->as<GroupedWelfordOp>()->getInitValOfOutput(out);
      } else if (default_val != nullptr) {
        init = default_val;
      }

      const bool is_output = out->isFusionOutput();

      // Don't need to alloc outputs, and if we don't need to initialize we're
      // done.
      if (is_output && init == nullptr) {
        continue;
      }

      AllocationInformation allocation;
      allocation.buffer = out_tv;
      fillAllocationInformation(allocation, expr);

      auto alloc_expr = createAllocExpr(allocation, is_output);
      auto init_expr = createInitExpr(allocation, init);

      // Write information to GPULower
      writeInfoToGPULower(allocation, alloc_expr);

      // Register allocations before initializations to keep them in the right
      // order
      if (alloc_expr != nullptr) {
        if (allocation.buffer->getMemoryType() == MemoryType::Shared) {
          // Shared allocations go at the begining of scope
          TORCH_INTERNAL_ASSERT(!exprs_.empty());
          registerInsertBefore(exprs_[0], alloc_expr, nullptr);
        } else {
          TORCH_INTERNAL_ASSERT(allocation.alloc_place_before != nullptr);
          kir::Scope* scope = allocation.alloc_for_loop == nullptr
              ? nullptr
              : &allocation.alloc_for_loop->body();
          registerInsertBefore(
              allocation.alloc_place_before, alloc_expr, scope);
        }
      }

      if (init_expr != nullptr) {
        TORCH_INTERNAL_ASSERT(allocation.init_place_before != nullptr);
        kir::Scope* scope = allocation.init_for_loop == nullptr
            ? nullptr
            : &allocation.init_for_loop->body();
        registerInsertBefore(allocation.init_place_before, init_expr, scope);
      }
    }
  }

  // Sends alloc_expr, info.has_halo, info.allocation_domains to GpuLower
  void writeInfoToGPULower(
      const AllocationInformation& allocation,
      kir::Allocate* alloc_expr) {
    auto& lower_alloc_info_map = GpuLower::current()->localAllocationInfoMap();
    if (alloc_expr == nullptr) {
      // Skip output allocation.
      return;
    }
    TORCH_INTERNAL_ASSERT(
        !lower_alloc_info_map.count(alloc_expr),
        "duplicated allocation info entry");

    // Create info entry for GPULower
    auto lower_alloc_info_ptr = std::make_unique<LocalAllocationInfo>();
    lower_alloc_info_ptr->alloc_expr = alloc_expr;
    lower_alloc_info_ptr->has_halo = allocation.has_halo;
    if (allocation.allocation_domains) {
      lower_alloc_info_ptr->alloc_domains = *(allocation.allocation_domains);
    }

    // Write entry to the stored map
    lower_alloc_info_map[alloc_expr] = std::move(lower_alloc_info_ptr);
  }

  void handle(kir::IfThenElse*) final {
    TORCH_INTERNAL_ASSERT(
        false,
        "Pass does not support conditional statements, ",
        "this pass should be run before any conditionals are placed in code.");
  }

  AllocationInserter(const std::vector<Expr*>& exprs)
      : gpu_lower(GpuLower::current()) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

 private:
  GpuLower* gpu_lower;

 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    AllocationInserter inserter(exprs);
    return inserter.exprs_;
  }
};

} // namespace

std::vector<Expr*> insertAllocations(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertAllocations");
  return AllocationInserter::insert(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
