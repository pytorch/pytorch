#ifdef USE_CUDA
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_manager.h>

namespace nvfuser {

FusionDefinition::FusionDefinition(FusionManager* fusion_manager)
    : fusion_manager_(fusion_manager),
      end_record_(std::make_shared<RecordFunctor>(new EndRecord())),
      recording_(),
      recording_state_(),
      fusion_state_(),
      ops(this) {}

FusionDefinition* FusionDefinition::enter() {
  fusion_manager_->resetFusionCachePtr();
  return this;
}
void FusionDefinition::exit() {
  auto cache_entry = fusion_manager_->lookupFusionCacheEntry(end_record_.get());
  if (!cache_entry.has_value()) {
    fusion_manager_->createTerminalFusionCacheEntry(end_record_);
    fusion_manager_->traverseFusionCache(end_record_);

    NvfFusionGuard::setCurFusion(fusion_manager_->fusionPtr());
    fusion_state_.resize(recording_state_.size(), nullptr);
    for (auto& record : recording_) {
      auto functor = record.get();
      (*functor)(*this);
    }
    NvfFusionGuard::setCurFusion(nullptr);
  } else {
    fusion_manager_->traverseFusionCache(end_record_);
  }
}

Scalar* FusionDefinition::defineScalar() {
  Scalar* out = new nvfuser::Scalar(recording_state_.size());
  recording_state_.emplace_back(out);
  return out;
}
Tensor* FusionDefinition::defineTensor() {
  Tensor* out = new nvfuser::Tensor(recording_state_.size());
  recording_state_.emplace_back(out);
  return out;
}
void FusionDefinition::defineRecord(RecordFunctor* record) {
  auto cache_entry = fusion_manager_->lookupFusionCacheEntry(record);
  if (cache_entry.has_value()) {
    recording_.emplace_back(cache_entry.value()->record);
  } else {
    recording_.emplace_back(record);
    fusion_manager_->createFusionCacheEntry(recording_.back());
  }
  fusion_manager_->traverseFusionCache(recording_.back());
}

void FusionDefinition::addInput(NvfVal* input) {
  fusion_manager_->fusionPtr()->addInput(input);
}
void FusionDefinition::addOutput(NvfVal* output) {
  fusion_manager_->fusionPtr()->addOutput(output);
}

NvfVal* FusionDefinition::getFusionState(size_t index) const {
  return fusion_state_.at(index);
}
void FusionDefinition::setFusionState(size_t index, NvfVal* val) {
  fusion_state_.at(index) = val;
}

} // namespace nvfuser

#endif // USE_CUDA
