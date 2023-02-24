#include <instrumentation.h>
#include <python_frontend/fusion_record.h>
#include <python_frontend/fusion_state.h>
#include <utils.h>

// Require namespace for perf scope instrumentation
using namespace nvfuser::inst;

namespace nvfuser::python_frontend {

FusionState::FusionState()
    : end_record_(new EndRecord()),
      recording_(),
      fusion_(nullptr),
      fusion_state_(),
      num_recording_states_(0) {}

std::unique_ptr<FusionState> FusionState::clone() {
  auto fc = std::make_unique<FusionState>();
  for (auto&& rf : recording_) {
    fc->recording_.emplace_back(rf->clone());
  }
  fc->fusion_ = fusion_;
  fc->fusion_state_.insert(
      fc->fusion_state_.end(), fusion_state_.begin(), fusion_state_.end());
  fc->num_recording_states_ = num_recording_states_;
  return fc;
}

void FusionState::buildFusionIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("FusionContainer::buildFusionIr");
  TORCH_CHECK(fusion != nullptr, "Fusion is undefined.");
  resetFusionState(fusion, num_recording_states_);
  auto fusion_guard = FusionGuard(fusion);
  for (auto& record : recording_) {
    auto functor = record.get();
    (*functor)(*this);
  }
}

void FusionState::addRecord(RecordFunctor* record) {
  FUSER_PERF_SCOPE("FusionContainer::addRecord");
  recording_.emplace_back(record);
  num_recording_states_ += record->numOutputs();
}

Fusion* FusionState::fusion() {
  TORCH_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  return fusion_;
}

void FusionState::printIr() {
  TORCH_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->printMath();
}

void FusionState::resetFusionState(Fusion* fusion, size_t size) {
  TORCH_CHECK(fusion != nullptr, "Fusion is undefined.");
  fusion_ = fusion;
  fusion_state_.clear();
  fusion_state_.resize(size, nullptr);
}

Val* FusionState::getFusionState(size_t index) const {
  return fusion_state_.at(index);
}

void FusionState::setFusionState(size_t index, Val* val) {
  fusion_state_.at(index) = val;
}

void FusionState::addInput(Val* input) {
  TORCH_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->addInput(input);
}

void FusionState::addOutput(Val* output) {
  TORCH_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->addOutput(output);
}

void FusionState::aliasOutputToInput(Val* output, Val* input) {
  TORCH_CHECK(fusion_ != nullptr, "Fusion is undefined.");
  fusion_->aliasOutputToInput(output, input);
}

} // namespace nvfuser::python_frontend
