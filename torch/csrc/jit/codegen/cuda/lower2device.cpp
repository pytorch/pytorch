#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_index.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_validation.h>

#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

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

void GPULower::lower() {
  FusionGuard fg(fusion_);

  // Validate and make some minor modifications in preparation to generate code.
  PrepareForLowering(fusion_);

  ThreadPredicateMap preds(fusion_);

  // Run our passes keeping the lowered expressions and forwarding them.
  auto loop_nests =
      LoopNestGenerator::getLoopNest(fusion_, fusion_->exprs(true), preds);

  auto unrolled_loops = UnrollPass::runPass(fusion_, loop_nests, preds);
  auto indexed_loops = IndexLowering::getIndexedExprs(fusion_, unrolled_loops);
  lowered_exprs_ = indexed_loops;

  // Get allocations:
  global_allocations_ = GridReductionBuffers::getGlobalAllocs(lowered_exprs_);
  sync_allocations_ = GridReductionBuffers::getSyncAllocs(lowered_exprs_);
}

// Traverse through the fusion and print CUDA code associated with it
std::vector<Expr*> GPULower::lowered_exprs() {
  return lowered_exprs_;
}

std::ostream& GPULower::printKernel(
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

std::string GPULower::getKernel(const std::string& kernel_name) {
  std::stringstream ss;
  printKernel(ss, kernel_name);
  return ss.str();
}

} // namespace fuser
} // namespace jit
} // namespace torch
