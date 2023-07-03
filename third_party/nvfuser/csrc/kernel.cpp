#include <instrumentation.h>
#include <ir_iostream.h>
#include <kernel.h>
#include <kernel_expr_evaluator.h>
#include <kernel_ir_dispatch.h>
#include <lower2device.h>

#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

IrBuilderPasskey::IrBuilderPasskey(IrContainer* ir_container)
    : ir_container_(ir_container) {}

namespace kir {

namespace {

//! Scan all primary expressions in the Kernel IR and build
//! lists of specialized nodes and other interesting information
class KernelIrScanner : private IrVisitor {
 public:
  explicit KernelIrScanner(const Kernel* kernel) {
    IrVisitor::handle(kernel->topLevelExprs());
    const auto gpu_lower = GpuLower::current();
    for (auto split : gpu_lower->nonDivisibleSplitInfo().splitsToValidate()) {
      auto extent = split->in()->extent();
      auto factor = split->factor();
      summary_.splits_to_validate.emplace_back(extent, factor);
    }
  }

  const auto& summary() const {
    return summary_;
  }

 private:
  using IrVisitor::handle;
  void handle(Expr* expr) final {
    IrVisitor::handle(expr);
    for (auto inp : expr->inputs()) {
      handle(inp);
    }
    for (auto out : expr->outputs()) {
      handle(out);
    }
  }
  void handle(BlockSync* sync) final {
    // TODO: Move to a dedicated validation pass
    // which is not on the common execution/compilation path
    if (sync->isWarHazardSync()) {
      ++summary_.war_hazard_syncs_count;
    }
  }

  void handle(GridSync* sync) final {
    summary_.has_cooperative_grid_reduction = true;
  }

  void handle(Allocate* allocate) final {
    switch (allocate->memoryType()) {
      case MemoryType::Global:
        summary_.global_allocations.push_back(allocate);
        break;
      case MemoryType::Shared:
        summary_.dynamic_smem_allocations.push_back(allocate);
        break;
      case MemoryType::Local:
        if (!ExpressionEvaluator::isConst(allocate->size())) {
          summary_.has_dynamic_local_memory_allocations = true;
          summary_.dynamic_lmem_allocations.emplace_back(allocate);
        }
        break;
    }
  }

  void handle(RNGOp* rng_op) final {
    summary_.max_rng_offsets =
        std::max<int>(summary_.max_rng_offsets, rng_op->getRNGOffset());
  }

  void handle(TensorIndex* tensor_index) final {
    const auto tv = tensor_index->view();
    const auto domain = tv->domain();
    // Do we have any reductions?
    summary_.has_block_reductions =
        summary_.has_block_reductions || domain->hasBlockReduction();

    // Update the largest smem data type
    if (domain->hasBlockReduction() || domain->hasGridReduction() ||
        tv->getMemoryType() == MemoryType::Shared) {
      const auto data_type = tv->dtype();
      const size_t type_size = dataTypeSize(data_type);
      if (type_size > max_smem_type_size_) {
        max_smem_type_size_ = type_size;
        summary_.largest_smem_data_type = data_type;
      }
    }
  }

  void handle(WelfordOp* welford_op) final {
    summary_.has_welford = true;
    TORCH_INTERNAL_ASSERT(welford_op->outAvg()->isA<TensorIndex>());
    auto out_dom = welford_op->outAvg()->as<TensorIndex>()->view()->domain();
    summary_.has_block_welford =
        summary_.has_block_welford || out_dom->hasBlockReduction();
  }

  void handle(GridWelford* grid_welford) final {
    summary_.has_welford = true;
    summary_.has_grid_welford = true;
    summary_.has_grid_reductions = true;
    if (grid_welford->welford_op()->isAllreduce()) {
      summary_.has_cooperative_grid_reduction = true;
    }
  }

  void handle(GridReduction* grid_reduction) final {
    summary_.has_grid_reductions = true;
    if (grid_reduction->isAllreduce()) {
      summary_.has_cooperative_grid_reduction = true;
    }
  }

  void handle(GroupedGridReduction* grid_reduction) final {
    summary_.has_grid_reductions = true;
    if (grid_reduction->isAllreduce()) {
      summary_.has_cooperative_grid_reduction = true;
    }
  }

  void handle(GroupedGridWelford* grid_welford) final {
    summary_.has_welford = true;
    summary_.has_grid_welford = true;
    summary_.has_grid_reductions = true;
    if (grid_welford->isAllreduce()) {
      summary_.has_cooperative_grid_reduction = true;
    }
  }

  void handle(GridBroadcast* grid_broadcast) final {
    summary_.has_cooperative_grid_reduction = true;
    handle(grid_broadcast->broadcast_op());
  }

  void handle(BroadcastOp* bop) final {
    const ParallelTypeBitmap parallel_types =
        GpuLower::current()->threadPredMap().getParallelBroadcastDomains(
            bop->out()->as<TensorIndex>()->view());
    summary_.broadcast_parallel_types.emplace(bop, parallel_types);
    // Do we have block broadcasts?
    summary_.has_block_broadcasts =
        summary_.has_block_broadcasts || parallel_types.hasTID();
    // Do we have grid broadcasts?
    summary_.has_grid_broadcasts =
        summary_.has_grid_broadcasts || parallel_types.hasBID();
  }

 private:
  size_t max_smem_type_size_ = 0;
  KernelSummary summary_;
};

//! Make sure tensors have valid allocations even when parallelized
//! loops potentially have larger iteration counts than the number of
//! threads.
//!
//! When an IterDomain of a tensor is parallelized, the IterDomain
//! may not contribute to the allocation of the tensor. For example,
//! it is assumed that an allocation of a local-memory tensor does not
//! need to be accounted for an parallelied IterDomain. This is true
//! when it is guaranteed that each thread only needs to execute the
//! loop body once. However, if not, the allocation is invalid as it
//! only has a space for one value per thread.
//!
//! ValidateAllocation checks all tensor allocations and sees if any
//! tensor may have a parallelized loop whose iteration count may
//! be larger than the number of threads. If so, an error is thrown if
//! the tensor is not allocated on thread-shared memories. Note that
//! when allocated on a shared memory (i.e., MemoryType::Shared or
//! MemoryType::Global for tensors parallelized with threadIdx, or
//! MemoryType::Global for tensors parallelized with blockIdx), it is
//! assumed that allocation is properly extended for the iteration
//! count.
class ValidateAllocation : private OptOutConstDispatch {
 public:
  static void validate(const Kernel* kernel) {
    ValidateAllocation validate_allocation(kernel);
  }

 private:
  explicit ValidateAllocation(const Kernel* kernel) {
    live_allocations_.emplace_back(std::vector<const Allocate*>());
    for (const auto& expr : kernel->topLevelExprs()) {
      OptOutConstDispatch::handle(expr);
    }
    live_allocations_.pop_back();
    TORCH_INTERNAL_ASSERT(live_allocations_.empty());
  }

  void handle(const Allocate* allocate) final {
    TORCH_INTERNAL_ASSERT(!live_allocations_.empty());
    live_allocations_.back().push_back(allocate);
  }

  // for_loop is parallelized and its stop value is not guaranteed to
  // be <= the number of threads, which breaks an assumption made
  // during in the allocation lowering if it's thread-parallel and not
  // allocated on shared or global memories, or if it's block-parallel
  // ando not allocated on global memory.
  void validate(const ForLoop* for_loop) {
    const auto loop_id = for_loop->iter_domain();
    for (const auto& allocations : live_allocations_) {
      for (const auto& allocate : allocations) {
        const auto tv = dynamic_cast<TensorView*>(allocate->buffer());
        if (tv == nullptr) {
          continue;
        }
        for (const auto& axis : tv->domain()->domain()) {
          if (!GpuLower::current()->caMap()->areMapped(
                  loop_id, axis, IdMappingMode::LOOP)) {
            continue;
          }
          if (isParallelTypeThreadDim(loop_id->getParallelType())) {
            TORCH_INTERNAL_ASSERT(
                tv->getMemoryType() == MemoryType::Shared ||
                    tv->getMemoryType() == MemoryType::Global,
                "Tensor t",
                tv->name(),
                " must be allocated on SMEM or GMEM.");
          } else if (isParallelTypeBlockDim(loop_id->getParallelType())) {
            TORCH_INTERNAL_ASSERT(tv->getMemoryType() == MemoryType::Global);
          }
        }
      }
    }
  }

  void handle(const ForLoop* for_loop) final {
    if (for_loop->stop() != for_loop->iter_domain()->extent() &&
        isParallelTypeThread(for_loop->iter_domain()->getParallelType())) {
      validate(for_loop);
    }

    live_allocations_.emplace_back(std::vector<const Allocate*>());
    for (const auto& expr : for_loop->body().exprs()) {
      OptOutConstDispatch::handle(expr);
    }
    live_allocations_.pop_back();
  }

  void handle(const IfThenElse* ite) final {
    for (const auto& expr : ite->thenBody().exprs()) {
      OptOutConstDispatch::handle(expr);
    }
    for (const auto& expr : ite->elseBody().exprs()) {
      OptOutConstDispatch::handle(expr);
    }
  }

 private:
  std::vector<std::vector<const Allocate*>> live_allocations_;
};

} // namespace

// TODO(kir): Kernel IR validation
void Kernel::finalize(std::vector<Expr*> top_level_exprs) {
  TORCH_INTERNAL_ASSERT(top_level_exprs_.empty());
  top_level_exprs_ = std::move(top_level_exprs);
  warp_padded_parallel_info_ = GpuLower::current()->getWarpPaddedParallelInfo();
  profile_ = GpuLower::current()->profile();
  ValidateAllocation::validate(this);
  analyze();
  // Make sure this is after analyze as it sets summary_
  summary_.vectorized_accesses = GpuLower::current()->vectorizedAccesses();
  summary_.vectorized_set_info = GpuLower::current()->vectorizedSetInfo();
  summary_.sync_map = GpuLower::current()->syncMap();
  summary_.parallel_dimension_map_ =
      GpuLower::current()->parallelDimensionMap();
}

void Kernel::analyze() {
  FUSER_PERF_SCOPE("Kernel::analyze");

  const KernelIrScanner ir_scanner(this);
  summary_ = ir_scanner.summary();
}

void Kernel::print() const {
  IrPrinter ir_printer(std::cout);
  ir_printer.handle(this);
}

//! Register the Val with this fusion
void Kernel::registerVal(Val* val) {
  if (inContainer(val)) {
    return;
  }
  if (val->kernel()) {
    TORCH_CHECK(
        val->kernel() == this,
        val->toString(),
        " was not found in the active kernel.");
  }

  Fusion::registerVal(val);
}

//! Register expr with this fusion.
//! When we register an expression, we want to update the dependency tracking
//! of Vals. We add expr to our general expr_set_,
void Kernel::registerExpr(Expr* expr) {
  if (inContainer(expr)) {
    return;
  }

  if (expr->kernel()) {
    TORCH_CHECK(
        expr->kernel() == this,
        expr->toString(),
        " was not found in the active kernel.");
  }

  for (Val* input : expr->inputs()) {
    TORCH_INTERNAL_ASSERT(
        inContainer(input),
        "Input\n",
        input->toString(),
        " to expr,\n",
        expr->toString(),
        ",\n is invalid because it is not in the same kernel.");
  }

  for (Val* output : expr->outputs()) {
    TORCH_INTERNAL_ASSERT(
        inContainer(output),
        "Output\n",
        output->toString(),
        " to expr,\n",
        expr->toString(),
        ",\n is invalid because it is not in the same kernel.");
  }

  // Register expr is explicitly non-SSA when coming from a kernel. This is
  // detected inside Fusion::registerExpr
  Fusion::registerExpr(expr);
}

std::vector<Expr*>& KernelInternalProxy::topLevelExprs() {
  return kernel_->top_level_exprs_;
}

void KernelPerformanceProfile::registerExpr(const Expr* expr) {
  if (expr_entry_map_.find(expr) != expr_entry_map_.end()) {
    return;
  }

  auto slot = getNewIndex();
  expr_entry_map_.emplace(expr, slot);
}

int KernelPerformanceProfile::getNewIndex() {
  return num_profile_entries_++;
}

bool KernelPerformanceProfile::isProfiled(const Expr* expr) const {
  return expr_entry_map_.find(expr) != expr_entry_map_.end();
}

c10::optional<int> KernelPerformanceProfile::getIndex(const Expr* expr) const {
  auto it = expr_entry_map_.find(expr);
  if (it == expr_entry_map_.end()) {
    return c10::optional<int>();
  } else {
    return it->second;
  }
}

std::array<int, 2> KernelPerformanceProfile::getIndicesInProfileBuffer(
    const Expr* expr) const {
  TORCH_INTERNAL_ASSERT(
      isProfiled(expr), "Not a profiled expression: ", expr->toString());

  int cycle_index = getIndex(expr).value() * 2;
  int count_index = cycle_index + 1;

  return {cycle_index, count_index};
}

std::string KernelPerformanceProfile::toString(const at::Tensor& buffer) const {
  std::stringstream ss;
  ss << "Kernel performance profile:\n";
  if (!buffer.defined()) {
    ss << "No profile found\n";
    return ss.str();
  }

  double kilo_freq = at::cuda::getCurrentDeviceProperties()->clockRate;

  ss << std::setprecision(3) << std::fixed;

  for (const auto& kv : expr_entry_map_) {
    auto expr = kv.first;
    auto index = kv.second;
    auto out_tv = ir_utils::getTvOutput(expr);
    double cycles = static_cast<double>(buffer[index][0].item<int64_t>());
    auto count = buffer[index][1].item<int64_t>();
    auto cycles_per_call = count == 0 ? 0.0 : cycles / count;
    auto us_per_call = cycles_per_call / kilo_freq * 1000.0;
    ss << expr->getExprType().value() << ", T" << out_tv->name() << ", "
       << us_per_call << " us, " << count << "\n";
  }

  return ss.str();
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
