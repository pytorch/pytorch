#include <torch/csrc/jit/codegen/cuda/lower_validation.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! A parallel type validation pass to make sure all the outputs of
//!   welford ops are parallelized the same way. Will infer and modify serial
//!   parallel types if other output/s are parallelized, so that
//!   user wouldn't have to specify the same parallelization
//!   3 times. Will throw if conflicts are detected, i.e.
//!   TIDx vs BIDx etc.
class ValidateParallelType : public IterVisitor {
 public:
  static void validate(Fusion* fusion) {
    ValidateParallelType VPT;
    VPT.traverse(fusion);
  }

 private:
  using IterVisitor::handle;
  void convertIterDomain(IterDomain* id0, IterDomain* id1) {
    const auto ptype0 = id0->getParallelType();
    const auto ptype1 = id1->getParallelType();

    if (ptype0 != ptype1) {
      TORCH_CHECK(
          ptype0 == ParallelType::Serial || ptype1 == ParallelType::Serial,
          "Error promoting parallel types");
      if (ptype0 == ParallelType::Serial) {
        id0->parallelize(ptype1);
      }
      if (ptype1 == ParallelType::Serial) {
        id1->parallelize(ptype0);
      }
    }
  }

  void handle(WelfordOp* wop) override {
    auto out_var = wop->outVar()->as<TensorView>();
    auto out_avg = wop->outAvg()->as<TensorView>();
    auto out_n = wop->outN()->as<TensorView>();
    TORCH_INTERNAL_ASSERT(out_var->nDims() == out_avg->nDims());
    TORCH_INTERNAL_ASSERT(out_var->nDims() == out_n->nDims());
    for (size_t i = 0; i < out_var->nDims(); i++) {
      // TODO: can be cleaner.
      convertIterDomain(out_var->axis(i), out_avg->axis(i));
      convertIterDomain(out_avg->axis(i), out_n->axis(i));
      convertIterDomain(out_n->axis(i), out_var->axis(i));
    }
  }
};

} // namespace

void validateIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("validateIr");

  FusionGuard fg(fusion);

  fusion->validateInputs();

  // Convert all output broadcast iterdomains to strided
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isBroadcast()) {
        id->toStridedBroadcast();
      }
    }
  }

  // Validate Parallelization
  ValidateParallelType::validate(fusion);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
