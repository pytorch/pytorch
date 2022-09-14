#ifdef USE_CUDA
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>

namespace nvfuser {

FusionDefinition::FusionDefinition(FusionOwner* fusion_owner)
    : fusion_owner_(fusion_owner),
      prev_fusion_(nullptr),
      recording_(),
      recording_state_(),
      fusion_state_(),
      ops(this) {}

FusionDefinition* FusionDefinition::enter() {
  prev_fusion_ = FusionGuard::getCurFusion();
  FusionGuard::setCurFusion(fusionPtr());
  return this;
}
void FusionDefinition::exit() {
  fusion_state_.resize(recording_state_.size(), nullptr);
  for (auto& record : recording_) {
    auto functor = record.get();
    (*functor)(*this);
  }

  FusionGuard::setCurFusion(prev_fusion_);
  prev_fusion_ = nullptr;
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
  recording_.emplace_back(record);
}

void FusionDefinition::addInput(NvfVal* input) {
  fusionPtr()->addInput(input);
}
void FusionDefinition::addOutput(NvfVal* output) {
  fusionPtr()->addOutput(output);
}

NvfVal* FusionDefinition::getFusionState(size_t index) const {
  return fusion_state_.at(index);
}
void FusionDefinition::setFusionState(size_t index, NvfVal* val) {
  fusion_state_.at(index) = val;
}

Fusion* FusionDefinition::fusionPtr() {
  return fusion_owner_->fusionPtr();
}

} // namespace nvfuser

#endif // USE_CUDA
