#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
constexpr int kUnrollFactor = 1;
constexpr int kThreadX = 128;
} // namespace

// This one is a total mess and it should go.
bool scheduleFusion(Fusion* fusion, const at::ArrayRef<c10::IValue> inputs) {
  FUSER_PERF_SCOPE("scheduleFusion");
  return scheduleFusion(fusion);
}

bool scheduleFusion(Fusion* fusion) {
  FusionGuard fg(fusion);
  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  TORCH_INTERNAL_ASSERT(
      !fusion->hasReduction(), "This scheduler only handles pointwise ops.");
  const bool disable_unroll = fusion->isStochastic();

  for (auto out_val : fusion->outputs()) {
    auto out = out_val->as<TensorView>();

    // Merge all dimensions because we're only supporting pointwise
    // Real reductions aren't supposed to reach here
    // This is a workaround to handle trivial reductions, i.e. size-1 reductions
    scheduler_utils::mergeNonReduction(out);
  }

  // Run through outputs, grab all inputs of outputs
  // squeeze with computeAt to set overall structure.
  for (auto output : fusion->outputs()) {
    if (output->getValType() != ValType::TensorView ||
        output->as<TensorView>()->nDims() == 0) {
      continue;
    }
    TensorView* out_tv = output->as<TensorView>();
    const auto domain = out_tv->getRootDomain();
    if (std::any_of(domain.begin(), domain.end(), [](IterDomain* iter_domain) {
          return iter_domain->extent()->isZeroInt();
        })) {
      continue;
    }

    // Split into 128 which will be bockDim.x
    out_tv->split(0, kThreadX);
    // Split by another 4 which will be our unroll factor
    auto ur_factor = disable_unroll ? 1 : kUnrollFactor;
    out_tv->split(0, ur_factor);
  }

  for (auto output : fusion->outputs()) {
    if (output->getValType() != ValType::TensorView) {
      continue;
    }
    TensorView* out_tv = output->as<TensorView>();
    const auto domain = out_tv->getRootDomain();
    if (std::any_of(domain.begin(), domain.end(), [](IterDomain* iter_domain) {
          return iter_domain->extent()->isZeroInt();
        })) {
      continue;
    }
    for (Val* inp : fusion->inputsOf(output)) {
      if (inp->getValType().value() == ValType::TensorView)
        inp->as<TensorView>()->computeAt(out_tv, -1);
    }
    if (output->as<TensorView>()->nDims() == 0) {
      continue;
    }
    out_tv->axis(0)->parallelize(ParallelType::BIDx);
    out_tv->axis(1)->parallelize(ParallelType::Unroll);
    out_tv->axis(2)->parallelize(ParallelType::TIDx);
  }

  return true;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
