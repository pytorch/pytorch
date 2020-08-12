#include <torch/csrc/jit/codegen/cuda/lower_validation.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

// Some pre-compilation checks
static void IrValidate(Fusion* fusion) {
  FusionGuard fg(fusion);
  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->outputs().begin(), fusion->outputs().end()}, fusion->inputs());

  std::unordered_set<TensorView*> used_tvs;

  for (auto val : used_vals) {
    if (ir_utils::isTV(val)) {
      used_tvs.emplace(val->as<TensorView>());
    }
  }

  fusion->validateInputs();

  for (auto tv : used_tvs) {
    for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
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

        for (auto id : tv->getRootDomain()) {
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

void IrBuildSizesMap(Fusion* fusion) {
  // Sizes of inputs/outputs -> T.size[...]
  std::unordered_map<Val*, Val*> size_map;

  // Grab inputs and outputs
  // TODO: Only run through inputs for the size map, outputs don't actually set
  // any sizes of the problem.
  std::vector<TensorView*> inputs_and_outputs;
  for (auto val : fusion->inputs()) {
    if (ir_utils::isTV(val)) {
      inputs_and_outputs.push_back(val->as<TensorView>());
    }
  }
  for (auto val : fusion->outputs()) {
    if (ir_utils::isTV(val)) {
      inputs_and_outputs.push_back(val->as<TensorView>());
    }
  }

  // Run through inputs and outputs first. Since we're replacing full
  // tensorviews their names are going to change. We need  the new referenc
  // name for the inputs/outputs. This way we won't reference the wrong tensor
  // view. For example T0 may be translated to T9. We don't want our new
  // variable to be T0->size[...] we need it to be T9->size[...]
  //
  // This could be done in a better way but changing split/merge to be a
  // TensorDomain focused operation, then we could simply do this process on
  // domains, instead of tensorviews. This would have the benefit that the
  // TensorView wouldn't change, so users pointers will remain valid. The other
  // option which seems less elegant but would also work is build up the domain
  // on the new tensor, and then simply replace it into the original one.
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

      std::stringstream ss;
      ss << "T" << tv->name() << ".size[" << dim++ << "]";
      Val* new_size =
          new kir::NamedScalar(ss.str(), orig_size->getDataType().value());
      if (!orig_size->sameAs(new_size) ||
          size_map.find(orig_size) == size_map.end()) {
        size_map[orig_size] = new_size;

        // TODO(kir): temporary duplicating the mapping
        //  to make sure we get to the right size from either
        //  the Fusion IR value or the Kernel IR one
        size_map[kir::lowerValue(orig_size)] = new_size;
      }
    }
  }

  fusion->setValuesMap(size_map);
}

void IrAdjustMemoryTypes(Fusion* fusion) {
  for (auto val : fusion->deterministic_vals()) {
    if (ir_utils::isTV(val)) {
      auto tv = val->as<TensorView>();
      if (fusion->hasInput(tv) || fusion->hasOutput(tv)) {
        tv->setMemoryType(MemoryType::Global);
      } else if (tv->getMemoryType() == MemoryType::Global) {
        tv->setMemoryType(MemoryType::Local);
      }
    }
  }
}

void PrepareForLowering(Fusion* fusion) {
  FusionGuard fg(fusion);

  IrValidate(fusion);
  IrBuildSizesMap(fusion);
  IrAdjustMemoryTypes(fusion);
}

} // namespace fuser
} // namespace jit
} // namespace torch
