#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_alias_memory.h>
#include <torch/csrc/jit/codegen/cuda/lower_index.h>
#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_validation.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO(kir): revisit this
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local GpuLower* active_gpu_lower = nullptr;

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
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (id->isReduction()) {
        continue;
      } else if (id->getIterType() == IterType::BroadcastWithoutStride) {
        continue;
        // NOLINTNEXTLINE(bugprone-branch-clone)
      } else if (id->getIterType() == IterType::BroadcastWithStride) {
        dim++;
        continue;
      } else if (orig_size->isConstScalar()) {
        dim++;
        continue;
      }

      // TODO(kir): consider a different implementation which doesn't
      //  hijack the kir_map_
      if (kir_map_.find(orig_size) == kir_map_.end()) {
        std::stringstream ss;
        ss << "T" << tv->name() << ".size[" << dim++ << "]";
        kir_map_[orig_size] = ir_builder.create<kir::NamedScalar>(
            ss.str(), orig_size->getDataType().value());
      }
    }
  }
}

void GpuLower::lower() {
  FUSER_PERF_SCOPE("lower");

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
  kernel_ = std::make_unique<Kernel>();

  // prepare for lowering
  validateIr(fusion_);
  replaceSymbolicSizes();

  // Compute thread predicates
  ThreadPredicateMap preds(fusion_);

  // Run our passes keeping the lowered expressions and forwarding them
  const auto lowered_exprs =
      LoopNestGenerator::loweredExprs(fusion_, preds, fusion_->exprs(true));

  const auto unrolled_loops =
      UnrollPass::runPass(fusion_, lowered_exprs, preds);

  // Reuse memory locations if:
  // TensorView is dynamic shared memory
  // TensorViews have the same size
  // Output TensorView is modified using Input TensorView
  const auto reuse_mem_exprs = reuseMemoryAllocations(fusion_, unrolled_loops);

  // Insert SyncThreads at end of for-loop to avoid WAR race condition
  const auto sync_exprs = insertThreadSynchronization(fusion_, reuse_mem_exprs);

  const auto indexed_loops =
      IndexLowering::getIndexedExprs(fusion_, sync_exprs);

  // We now have the lowered expressions, finalize the kernel IR
  kernel_->finalize(indexed_loops, preds);

  // Set the kernel inputs & outputs
  for (auto input : fusion_->inputs()) {
    kernel_->addInput(GpuLower::lowerValue(input));
  }
  for (auto output : fusion_->outputs()) {
    kernel_->addOutput(GpuLower::lowerValue(output));
  }
}

Kernel* GpuLower::kernel() const {
  TORCH_CHECK(kernel_);
  return kernel_.get();
}

// Maps Fusion IR nodes to the Kernel IR counterparts
//
// TODO(kir): this is a interim solution for easing the Kernel IR splitting
//
class TORCH_CUDA_CU_API GpuLower::KernelIrMapper : private OptInConstDispatch {
 public:
  explicit KernelIrMapper(GpuLower* gpu_lower)
      : gpu_lower_(gpu_lower), ir_builder_(gpu_lower->kernel()) {}

  Val* lower(const Val* value) {
    const auto it = gpu_lower_->kir_map_.find(value);
    if (it != gpu_lower_->kir_map_.end()) {
      return it->second;
    } else {
      handle(value);
      const auto lowered_node = gpu_lower_->kir_map_[value];
      TORCH_CHECK(lowered_node != nullptr);
      TORCH_CHECK(kir::isLoweredVal(lowered_node));

      // Lower the arithmetic expression defining the value, if any
      if (value->isScalar()) {
        if (auto def = value->getOrigin()) {
          lowerDefinition(lowered_node, def);
        }
      }

      return lowered_node;
    }
  }

 private:
  // TODO(kir): rewrite this
  void lowerDefinition(Val* lowered_value, const Expr* def) {
    switch (def->type()) {
      case ExprType::UnaryOp: {
        const auto op = def->as<UnaryOp>();
        ir_builder_.create<kir::UnaryOp>(
            op->getUnaryOpType(), lowered_value, lower(op->in()));
        break;
      }
      case ExprType::BinaryOp: {
        const auto op = def->as<BinaryOp>();
        ir_builder_.create<kir::BinaryOp>(
            op->getBinaryOpType(),
            lowered_value,
            lower(op->lhs()),
            lower(op->rhs()));
        break;
      }
      case ExprType::TernaryOp: {
        const auto op = def->as<TernaryOp>();
        ir_builder_.create<kir::TernaryOp>(
            op->getTernaryOpType(),
            lowered_value,
            lower(op->in1()),
            lower(op->in2()),
            lower(op->in3()));
        break;
      }
      default:
        TORCH_CHECK(false, "Unexpected expression type");
    }
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  }

  void handle(const Statement* node) override {
    OptInConstDispatch::handle(node);
  }

  void handle(const Val* node) override {
    OptInConstDispatch::handle(node);
  }

  void handle(const Expr* node) override {
    OptInConstDispatch::handle(node);
  }

  void handle(const TensorDomain* node) override {
    const auto lowered_node = ir_builder_.create<kir::TensorDomain>(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const IterDomain* node) override {
    const auto lowered_node = ir_builder_.create<kir::IterDomain>(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const TensorView* node) override {
    const auto lowered_node = ir_builder_.create<kir::TensorView>(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const Bool* node) override {
    const auto lowered_node = ir_builder_.create<kir::Bool>(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const Float* node) override {
    const auto lowered_node = ir_builder_.create<kir::Float>(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const Half* node) override {
    const auto lowered_node = ir_builder_.create<kir::Half>(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const Int* node) override {
    const auto lowered_node = ir_builder_.create<kir::Int>(node, false);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const NamedScalar* node) override {
    const auto lowered_node = ir_builder_.create<kir::NamedScalar>(
        node->name(), node->getDataType().value());
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

 private:
  GpuLower* gpu_lower_ = nullptr;
  kir::IrBuilder ir_builder_;
};

Val* GpuLower::lowerValue(const Val* val) {
  TORCH_INTERNAL_ASSERT(!kir::isLoweredVal(val));
  TORCH_INTERNAL_ASSERT(active_gpu_lower != nullptr);
  KernelIrMapper kir_mapper(active_gpu_lower);
  return kir_mapper.lower(val);
}

Val* GpuLower::getLowerValue(const Val* val) {
  KernelIrMapper kir_mapper(this);
  return kir_mapper.lower(val);
}

GpuLower* GpuLower::current() {
  TORCH_INTERNAL_ASSERT(active_gpu_lower != nullptr);
  return active_gpu_lower;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
