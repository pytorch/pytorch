#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower_alias_memory.h>
#include <torch/csrc/jit/codegen/cuda/lower_allocation.h>
#include <torch/csrc/jit/codegen/cuda/lower_expr_sort.h>
#include <torch/csrc/jit/codegen/cuda/lower_index.h>
#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_validation.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO(kir): revisit this
thread_local GpuLower* active_gpu_lower = nullptr; // NOLINT

void GpuLower::replaceSymbolicSizes() {
  FUSER_PERF_SCOPE("replaceSymbolicSizes");

  kir::IrBuilder ir_builder(kernel());

  // Grab inputs and outputs
  // TODO: Only run through inputs for the size map, outputs don't actually set
  // any sizes of the problem.
  std::vector<TensorView*> inputs_and_outputs;
  for (auto val : fusion_->inputs()) {
    if (ir_utils::isTV(val)) {
      inputs_and_outputs.push_back(val->as<TensorView>());
    }
  }
  for (auto val : fusion_->outputs()) {
    if (ir_utils::isTV(val)) {
      inputs_and_outputs.push_back(val->as<TensorView>());
    }
  }

  // Run through inputs and outputs first. Since we're replacing full
  // tensorviews their names are going to change. We need  the new referenc
  // name for the inputs/outputs. This way we won't reference the wrong tensor
  // view. For example T0 may be translated to T9. We don't want our new
  // variable to be T0->size[...] we need it to be T9->size[...]
  for (TensorView* tv : inputs_and_outputs) {
    // Replace the domain with one based on Ti.size[j]
    std::vector<IterDomain*> new_domain_iters;
    const std::vector<IterDomain*>& root_td = tv->getRootDomain();

    size_t dim = 0;
    for (auto id : root_td) {
      const Val* orig_size = id->extent();

      // Output sizes could have reduction axes, which isn't what gets output.
      if (id->isReduction() ||
          (id->getIterType() == IterType::BroadcastWithoutStride)) {
        continue;
      } else if (
          (id->getIterType() == IterType::BroadcastWithStride) ||
          orig_size->isConstScalar()) {
        dim++;
        continue;
      }

      // TODO(kir): consider a different implementation which doesn't
      //  hijack the kir_val_map_
      // Currently turn off this part for inputs of segmented fusion,
      //  since FusionSegmentRuntime will provide these as integer inputs
      if (kir_val_map_.find(orig_size) == kir_val_map_.end() &&
          !orig_size->isFusionInput()) {
        std::stringstream ss;
        ss << "T" << tv->name() << ".size[" << dim++ << "]";
        kir_val_map_[orig_size] = ir_builder.create<kir::NamedScalar>(
            ss.str(), orig_size->getDataType().value());
      }
    }
  }
}

void GpuLower::lower() {
  FUSER_PERF_SCOPE("GpuLower::lower");

  TORCH_INTERNAL_ASSERT(fusion_ != nullptr);
  TORCH_INTERNAL_ASSERT(
      active_gpu_lower == nullptr, "Nested lowering passes are not supported");

  // TODO(kir): revisit this
  struct LowerGuard {
    LowerGuard(GpuLower* gpu_lower) {
      active_gpu_lower = gpu_lower;
    }
    ~LowerGuard() {
      active_gpu_lower = nullptr;
    }
  } lower_guard(this);

  FusionGuard fg(fusion_);

  // Start with a fresh kernel
  kernel_ = std::make_unique<kir::Kernel>();

  // prepare for lowering
  validateIr(fusion_);
  replaceSymbolicSizes();
  trivial_reduction_info_.build(fusion_, this);

  // In the future we may directly use this map, but for now it will propagate
  // and validate (to some extent) the parallelization strategy.
  // This is the first time nodes will be lowered to kir nodes. Since for now we
  // propagate the parallel strategy in some instances, we need to do it before
  // lowering.
  ca_parallel_map_ = ComputeAtMap(ComputeAtMap::MappingMode::PARALLEL);
  ca_parallel_map_.build(fusion_, current());

  // Want to run this after parallel map is created
  validateVectorize(fusion_);

  // Generate mappings to generate indices
  ca_index_map_ = ComputeAtMap(ComputeAtMap::MappingMode::INDEX);
  ca_index_map_.build(fusion_, current());

  // Generate mappings to generate and map to loop nests
  ca_loop_map_ = ComputeAtMap(ComputeAtMap::MappingMode::LOOP);
  ca_loop_map_.build(fusion_, current());

  validateParallelize(fusion_);

  // Compute thread predicates
  ThreadPredicateMap preds(fusion_);

  // Set the kernel inputs & outputs
  for (auto input : fusion_->inputs()) {
    kernel_->addInput(GpuLower::lowerValue(input));
  }

  for (auto output : fusion_->outputs()) {
    kernel_->addOutput(GpuLower::lowerValue(output));
  }

  // Run our passes keeping the lowered expressions and forwarding
  // them

  // Reorder expressions for loop-nest generation respecting computeAt
  // relationships
  auto sorted_exprs = reorderExprsForComputeAt();

  // Generate loop-nests and place each expression at its
  // corresponding loop
  const auto lowered_exprs = LoopNestGenerator::loweredExprs(sorted_exprs);

  // Insert allocations
  const auto alloced_exprs = insertAllocations(lowered_exprs);

  // Insert read after write smem syncs
  const auto raw_sync_exprs = insertRawThreadSynchronization(alloced_exprs);

  const auto unrolled_loops =
      UnrollPass::runPass(fusion_, raw_sync_exprs, preds);

  // Reuse memory locations if:
  // TensorView is dynamic shared memory
  // TensorViews have the same size
  // Output TensorView is modified using Input TensorView
  const auto reuse_mem_exprs = reuseMemoryAllocations(unrolled_loops);

  // Insert SyncThreads at end of for-loop to avoid WAR race condition
  const auto war_sync_exprs = insertWarThreadSynchronization(reuse_mem_exprs);

  const auto indexed_loops =
      IndexLowering::getIndexedExprs(war_sync_exprs, preds);

  // We now have the lowered expressions, finalize the kernel IR
  kernel_->finalize(indexed_loops, preds);
}

kir::Kernel* GpuLower::kernel() const {
  TORCH_CHECK(kernel_);
  return kernel_.get();
}

// Maps Fusion IR nodes to the Kernel IR counterparts
class GpuLower::KernelIrMapper : private OptInConstDispatch {
 public:
  explicit KernelIrMapper(GpuLower* gpu_lower)
      : gpu_lower_(gpu_lower), ir_builder_(gpu_lower->kernel()) {}

  kir::Val* lowerValue(const Val* value) {
    const auto it = gpu_lower_->kir_val_map_.find(value);
    if (it != gpu_lower_->kir_val_map_.end()) {
      return it->second;
    } else {
      handle(value);
      const auto kir_value = gpu_lower_->kir_val_map_[value];
      TORCH_CHECK(kir_value != nullptr);

      // Lower the value definition, if any
      if (value->isScalar()) {
        if (auto def = value->definition()) {
          const auto kir_def = lowerExpr(def);
          TORCH_INTERNAL_ASSERT(kir_value->definition() == kir_def);
        }
      }

      return kir_value;
    }
  }

  kir::Expr* lowerExpr(const Expr* expr) {
    const auto it = gpu_lower_->kir_expr_map_.find(expr);
    if (it != gpu_lower_->kir_expr_map_.end()) {
      return it->second;
    } else {
      handle(expr);
      const auto lowered_node = gpu_lower_->kir_expr_map_[expr];
      TORCH_CHECK(lowered_node != nullptr);
      return lowered_node;
    }
  }

 private:
  void handle(const Statement* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const Val* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const Expr* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const TensorDomain* node) final {
    const auto lowered_node = ir_builder_.create<kir::TensorDomain>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const IterDomain* node) final {
    const auto lowered_node = ir_builder_.create<kir::IterDomain>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const TensorView* node) final {
    const auto lowered_node = ir_builder_.create<kir::TensorView>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const Bool* node) final {
    const auto lowered_node = ir_builder_.create<kir::Bool>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const Double* node) final {
    const auto lowered_node = ir_builder_.create<kir::Double>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const Int* node) final {
    const auto lowered_node = ir_builder_.create<kir::Int>(node, false);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const NamedScalar* node) final {
    const auto lowered_node = ir_builder_.create<kir::NamedScalar>(
        node->name(), node->getDataType().value());
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const UnaryOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
        node->getUnaryOpType(),
        lowerValue(node->out()),
        lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const BinaryOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::BinaryOp>(
        node->getBinaryOpType(),
        lowerValue(node->out()),
        lowerValue(node->lhs()),
        lowerValue(node->rhs()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const TernaryOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::TernaryOp>(
        node->getTernaryOpType(),
        lowerValue(node->out()),
        lowerValue(node->in1()),
        lowerValue(node->in2()),
        lowerValue(node->in3()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const ReductionOp* node) final {
    auto out_tv = node->out()->as<TensorView>();
    // If trivial reduction operation lower to set operation.
    if (std::all_of(
            out_tv->domain()->domain().begin(),
            out_tv->domain()->domain().end(),
            [&](IterDomain* id) {
              // If id is a reduction axis, is it a trivial reduction?
              if (id->isReduction()) {
                return gpu_lower_->trivialReductionInfo().isDerived(id);
              } else {
                return true;
              }
            })) {
      const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
          UnaryOpType::Set, lowerValue(node->out()), lowerValue(node->in()));
      TORCH_CHECK(
          gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
      return;
    }

    const auto lowered_node = ir_builder_.create<kir::ReductionOp>(
        node->getReductionOpType(),
        lowerValue(node->init()),
        lowerValue(node->out()),
        lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const WelfordOp* node) final {
    auto lowerOptional = [&](Val* v) { return v ? lowerValue(v) : nullptr; };
    const auto lowered_node = ir_builder_.create<kir::WelfordOp>(
        lowerValue(node->outVar()),
        lowerValue(node->outAvg()),
        lowerValue(node->outN()),
        lowerValue(node->initVar()),
        lowerValue(node->initAvg()),
        lowerValue(node->initN()),
        lowerOptional(node->inVar()),
        lowerValue(node->inAvg()),
        lowerValue(node->inN()));

    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const BroadcastOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::BroadcastOp>(
        lowerValue(node->out()), lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const TransposeOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
        UnaryOpType::Set, lowerValue(node->out()), lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

 private:
  GpuLower* gpu_lower_ = nullptr;
  kir::IrBuilder ir_builder_;
};

kir::Val* GpuLower::lowerValue(const Val* val) {
  KernelIrMapper kir_mapper(this);
  return kir_mapper.lowerValue(val);
}

kir::Expr* GpuLower::lowerExpr(const Expr* expr) {
  KernelIrMapper kir_mapper(this);
  return kir_mapper.lowerExpr(expr);
}

GpuLower* GpuLower::current() {
  return active_gpu_lower;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
