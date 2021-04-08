#include <c10/util/irange.h>

#include <torch/csrc/jit/codegen/cuda/lower_validation.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void validateIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("validateIr");

  FusionGuard fg(fusion);

  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->outputs().begin(), fusion->outputs().end()}, fusion->inputs());

  std::unordered_set<TensorView*> used_tvs;

  for (const auto& val : used_vals) {
    if (ir_utils::isTV(val)) {
      used_tvs.emplace(val->as<TensorView>());
    }
  }

  fusion->validateInputs();

  for (const auto& tv : used_tvs) {
    for (const auto i : c10::irange(tv->nDims())) {
      IterDomain* id = tv->getComputeAtAxis(i).first;

      if (id->isBlockDim()) {
        TORCH_CHECK(
            !id->isBroadcast(),
            "Parallelization across blocks on broadcast axes is not supported, but found on, ",
            tv,
            ".");
      }
      if (tv->hasBroadcast() && tv->getMemoryType() != MemoryType::Global) {
        auto td = tv->domain()->domain();
        auto ca_inputs = ir_utils::iterDomainInputsOf(
            {td.begin(), td.begin() + tv->getThisComputeAtAxis()});
        auto non_ca_inputs = ir_utils::iterDomainInputsOf(
            {td.begin() + tv->getThisComputeAtAxis(), td.end()});

        std::unordered_set<IterDomain*> ca_inputs_set(
            ca_inputs.begin(), ca_inputs.end());
        std::unordered_set<IterDomain*> non_ca_inputs_set(
            non_ca_inputs.begin(), non_ca_inputs.end());

        for (const auto& id : tv->getRootDomain()) {
          if (id->isBroadcast()) {
            // If a broadcast dimension is an input to both an axis within the
            // computeAt point and outside the compute at point we would have to
            // look at consumers to figure out what that axis will be
            // broadcasted to, because we would have to generate everything the
            // consumer could need on that axis. This could be supported but is
            // not at this point.
            TORCH_INTERNAL_ASSERT(
                !(ca_inputs_set.find(id) != ca_inputs_set.end() &&
                  non_ca_inputs_set.find(id) != non_ca_inputs_set.end()),
                "Cannot generate a kernel where a root broadcast dimension is input to both IterDomains outside and within the computeAt point.");
          }
        }
      }
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
