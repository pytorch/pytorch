
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_index.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_validation.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

// TODO(kir): revisit this
thread_local GpuLower* active_gpu_lower = nullptr;

class GridReductionBuffers : OptOutDispatch {
 public:
  static std::vector<kir::Allocate*> getGlobalAllocs(
      const std::vector<Expr*>& exprs) {
    GridReductionBuffers fgr;
    for (auto expr : exprs) {
      fgr.handle(expr);
    }
    return fgr.global_allocations_;
  }

  static std::vector<kir::Allocate*> getSyncAllocs(
      const std::vector<Expr*>& exprs) {
    GridReductionBuffers fgr;
    for (auto expr : exprs) {
      fgr.handle(expr);
    }
    return fgr.sync_allocations_;
  }

 private:
  std::vector<kir::Allocate*> global_allocations_;
  std::vector<kir::Allocate*> sync_allocations_;

  GridReductionBuffers() = default;

  void handle(Expr* expr) final {
    OptOutDispatch::handle(expr);
  }

  void handle(kir::ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      OptOutDispatch::handle(expr);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    for (auto expr : ite->body().exprs()) {
      OptOutDispatch::handle(expr);
    }

    for (auto expr : ite->elseBody().exprs()) {
      OptOutDispatch::handle(expr);
    }
  }

  void handle(kir::GridReduction* gr) final {
    global_allocations_.push_back(gr->reduction_buffer());
    sync_allocations_.push_back(gr->sync_buffer());
  }
};

} // namespace

void GpuLower::buildSizesMap() {
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
      // Output sizes could have reduction axes, which isn't what gets output.

      Val* orig_size = id->extent();

      if (id->isReduction()) {
        continue;
      } else if (id->getIterType() == IterType::BroadcastWithoutStride) {
        continue;
      } else if (id->getIterType() == IterType::BroadcastWithStride) {
        dim++;
        continue;
      } else if (orig_size->isConstScalar()) {
        dim++;
        continue;
      }

      if (kir_map_.find(orig_size) == kir_map_.end()) {
        std::stringstream ss;
        ss << "T" << tv->name() << ".size[" << dim++ << "]";
        auto new_size =
            new kir::NamedScalar(ss.str(), orig_size->getDataType().value());
        kir_map_[orig_size] = new_size;
      }
    }
  }
}

void GpuLower::adjustMemoryTypes() {
  for (auto val : fusion_->deterministic_vals()) {
    if (ir_utils::isTV(val)) {
      auto tv = val->as<TensorView>();
      if (fusion_->hasInput(tv) || fusion_->hasOutput(tv)) {
        tv->setMemoryType(MemoryType::Global);
      } else if (tv->getMemoryType() == MemoryType::Global) {
        tv->setMemoryType(MemoryType::Local);
      }
    }
  }
}

void GpuLower::lower() {
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

  // prepare for lowering
  validateIr(fusion_);
  buildSizesMap();
  adjustMemoryTypes();

  // Compute thread predicates
  ThreadPredicateMap preds(fusion_);

  // Run our passes keeping the lowered expressions and forwarding
  // them.
  const auto lowered_exprs =
      LoopNestGenerator::loweredExprs(fusion_, preds, fusion_->exprs(true));

  const auto unrolled_loops =
      UnrollPass::runPass(fusion_, lowered_exprs, preds);

  const auto indexed_loops =
      IndexLowering::getIndexedExprs(fusion_, unrolled_loops);

  // Store the final lowered IR
  lowered_exprs_ = indexed_loops;

  // Get allocations
  global_allocations_ = GridReductionBuffers::getGlobalAllocs(lowered_exprs_);
  sync_allocations_ = GridReductionBuffers::getSyncAllocs(lowered_exprs_);
}

// Traverse through the fusion and print CUDA code associated with it
std::ostream& GpuLower::printKernel(
    std::ostream& os,
    const std::string& kernel_name) {
  FusionGuard fg(fusion_);

  std::vector<kir::Allocate*> allocs;
  allocs.insert(
      allocs.end(), global_allocations_.begin(), global_allocations_.end());
  allocs.insert(
      allocs.end(), sync_allocations_.begin(), sync_allocations_.end());

  std::vector<Val*> global_tensors(allocs.size(), nullptr);
  std::transform(
      allocs.begin(),
      allocs.end(),
      global_tensors.begin(),
      [](kir::Allocate* alloc) { return alloc->buffer(); });

  IRPrinter irp(os);
  irp.printKernel(lowered_exprs_, kernel_name, global_tensors);
  return os;
}

std::string GpuLower::getKernel(const std::string& kernel_name) {
  std::stringstream ss;
  printKernel(ss, kernel_name);
  return ss.str();
}

// Maps Fusion IR nodes to the Kernel IR counterparts
// (this is a interim solution for easing the Kernel IR splitting)
class TORCH_CUDA_API GpuLower::KernelIrMapper : private OptInConstDispatch {
 public:
  explicit KernelIrMapper(GpuLower* gpu_lower) : gpu_lower_(gpu_lower) {}

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
        const auto op = def->as<fuser::UnaryOp>();
        new kir::UnaryOp(op->getUnaryOpType(), lowered_value, lower(op->in()));
        break;
      }
      case ExprType::BinaryOp: {
        const auto op = def->as<fuser::BinaryOp>();
        new kir::BinaryOp(
            op->getBinaryOpType(),
            lowered_value,
            lower(op->lhs()),
            lower(op->rhs()));
        break;
      }
      case ExprType::TernaryOp: {
        const auto op = def->as<fuser::TernaryOp>();
        new kir::TernaryOp(
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
    const auto lowered_node = new kir::TensorDomain(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const IterDomain* node) override {
    const auto lowered_node = new kir::IterDomain(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const TensorView* node) override {
    const auto lowered_node = new kir::TensorView(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const Bool* node) override {
    const auto lowered_node = new kir::Bool(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const Float* node) override {
    const auto lowered_node = new kir::Float(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const Half* node) override {
    const auto lowered_node = new kir::Half(node);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const Int* node) override {
    const auto lowered_node = new kir::Int(node, false);
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

  void handle(const NamedScalar* node) override {
    const auto lowered_node =
        new kir::NamedScalar(node->name(), node->getDataType().value());
    TORCH_CHECK(gpu_lower_->kir_map_.insert({node, lowered_node}).second);
  }

 private:
  GpuLower* gpu_lower_ = nullptr;
};

Val* GpuLower::lowerValue(const Val* val) {
  TORCH_INTERNAL_ASSERT(active_gpu_lower != nullptr);
  KernelIrMapper kir_mapper(active_gpu_lower);
  return kir_mapper.lower(val);
}

} // namespace fuser
} // namespace jit
} // namespace torch
