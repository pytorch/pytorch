#include <torch/csrc/jit/codegen/cuda/lower_validation.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

// Some pre-compilation checks
static void IrValidate(Fusion* fusion) {
  fusion->validateInputs();
  for (Val* val : fusion->vals()) {
    if (ir_utils::isTV(val)) {
      TensorView* tv = ir_utils::asTV(val);
      for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
        IterDomain* id = tv->getComputeAtAxis(i).first;

        if (id->isBlockDim()) {
          TORCH_CHECK(
              !id->isBroadcast(),
              "Parallelization across blocks on broadcast axes is not supported, but found on, ",
              tv,
              ".");
        }
      }
    }
  }
}

void IrBuildSizesMap(Fusion* fusion) {
  // Sizes of inputs/outputs -> T.size[...]
  std::unordered_map<Val*, Val*> size_map;

  // Grab inputs and outputs
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
      if (id->isReduction())
        continue;

      Val* orig_size = id->extent();

      std::stringstream ss;
      ss << "T" << tv->name() << ".size[" << dim++ << "]";
      Val* new_size =
          new NamedScalar(ss.str(), orig_size->getDataType().value());
      if (!orig_size->sameAs(new_size) ||
          size_map.find(orig_size) == size_map.end())
        size_map[orig_size] = new_size;
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
