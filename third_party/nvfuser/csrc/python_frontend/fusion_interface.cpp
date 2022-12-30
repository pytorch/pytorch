#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_interface.h>

namespace nvfuser {

FusionInterface::FusionInterface() : fusion_id_(c10::nullopt) {}
FusionInterface::FusionInterface(size_t fusion_id)
    : fusion_id_(c10::optional<size_t>(fusion_id)) {}

void FusionInterface::define(size_t fusion_id) {
  auto fc = FusionCache::get();
  TORCH_CHECK(fusion_id < fc->fusions_.size(), "Invalid fusion id!");
  fusion_id_ = c10::optional<size_t>(fusion_id);
}

bool FusionInterface::defined() const {
  return fusion_id_.has_value();
}

size_t FusionInterface::id() const {
  TORCH_CHECK(defined(), "Invalid fusion id!");
  return fusion_id_.value();
}

void FusionInterface::addInput(Nvf::Val* input) const {
  fusionPtr()->addInput(input);
}

void FusionInterface::addOutput(Nvf::Val* output) const {
  fusionPtr()->addOutput(output);
}

std::vector<at::Tensor> FusionInterface::execute(
    const at::ArrayRef<c10::IValue>& inputs) const {
  // aliasOutputToInput always adds Tensors as outputs that we don't want
  // to return to the user. We need to remove them.
  auto count_output_aliases = fusionPtr()->getOutputAliasIndices().size();
  auto result = fusionExecutorCachePtr()->runFusionWithInputs(inputs);
  result.erase(result.begin(), result.begin() + count_output_aliases);
  return result;
}

Nvf::FusionGuard FusionInterface::guard() const {
  return Nvf::FusionGuard(fusionPtr());
}

void FusionInterface::print() const {
  fusionExecutorCachePtr()->printFusion();
}

Nvf::FusionExecutorCache* FusionInterface::fusionExecutorCachePtr() const {
  auto fc = FusionCache::get();
  TORCH_CHECK(defined(), "Invalid fusion id!");
  TORCH_CHECK(
      fc->fusions_.at(fusion_id_.value()), "FusionExecutorCache Ptr is Null!");
  return fc->fusions_.at(fusion_id_.value()).get();
}

Nvf::Fusion* FusionInterface::fusionPtr() const {
  auto fusion_ptr = fusionExecutorCachePtr()->fusion();
  TORCH_CHECK(fusion_ptr != nullptr, "Fusion IR pointer is null!");
  return fusion_ptr;
}

} // namespace nvfuser
