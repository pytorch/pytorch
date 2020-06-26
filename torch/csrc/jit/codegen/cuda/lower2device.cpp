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

// Traverse through the fusion and print CUDA code associated with it
std::vector<Expr*> GPULower::getLoweredExprs() {
  FusionGuard fg(fusion_);

  // Validate and make some minor modifications in preparation to generate code.
  PrepareForLowering(fusion_);

  auto preds = ThreadPredicates::compute(fusion_);

  // Run our passes keeping the lowered expressions and forwarding them.
  auto loop_nests = LoopNestGenerator::getLoopNest(
      fusion_, fusion_->exprs(true, false, true), preds);

  auto unrolled_loops = UnrollPass::runPass(fusion_, loop_nests, preds);

  auto indexed_loops = IndexLowering::getIndexedExprs(fusion_, unrolled_loops);

  return indexed_loops;
}

std::ostream& GPULower::printKernel(
    std::ostream& os,
    const std::string& kernel_name) {
  FusionGuard fg(fusion_);
  auto exprs = getLoweredExprs();

  IRPrinter irp(os);
  irp.printKernel(exprs, kernel_name);
  return os;
}

} // namespace fuser
} // namespace jit
} // namespace torch
