#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_owner.h>

using NvfDataType = torch::jit::fuser::cuda::DataType;
using NvfTensorView = torch::jit::fuser::cuda::TensorView;
using NvfVal = torch::jit::fuser::cuda::Val;

namespace nvfuser {

struct RecordFunctor;

enum class StateType {
  Tensor,
  Scalar,
  None,
};

struct State {
  State(StateType _stype, size_t _index) : stype(_stype), index(_index) {}

  StateType stype;
  size_t index;
};

struct Tensor : State {
  Tensor(size_t _index) : State(StateType::Tensor, _index) {}
};

struct Scalar : State {
  Scalar(size_t _index) : State(StateType::Scalar, _index) {}
};

// Manually applying the fusion guard via a context manager
class FusionDefinition {
 public:
  FusionDefinition(FusionOwner* fusion_owner)
    : fusion_owner_(fusion_owner),
      prev_fusion_(nullptr),
      recording(),
      recording_state(),
      fusion_state(),
      ops(this) {}

  // The copy/move/assign constructors/operators are being removed
  // because it is not possible to copy the fusion_recording data member
  // because that would require a virtual copy/move/assign of the
  // RecordFunctor that is not supported.
  FusionDefinition(const FusionDefinition& fd) = delete;
  FusionDefinition(FusionDefinition&& fd) = delete;
  FusionDefinition& operator=(const FusionDefinition& fd) = delete;
  FusionDefinition& operator=(FusionDefinition&& fd) = delete;

  // Context Manager Methods
  FusionDefinition* enter() {
    prev_fusion_ = FusionGuard::getCurFusion();
    FusionGuard::setCurFusion(fusionPtr());
    return this;
  }

  void exit() {
    FusionGuard::setCurFusion(prev_fusion_);
    prev_fusion_ = nullptr;
  }

  void addInput(torch::jit::fuser::cuda::Val* input) {
    fusionPtr()->addInput(input);
  }
  void addOutput(torch::jit::fuser::cuda::Val* output) {
    fusionPtr()->addOutput(output);
  }

  Fusion* fusionPtr() {
    return fusion_owner_->fusionPtr();
  }

 private:
  FusionOwner* fusion_owner_;
  Fusion* prev_fusion_;

 public:
  std::vector<std::unique_ptr<RecordFunctor>> recording;
  std::vector<std::unique_ptr<State>> recording_state;
  std::vector<NvfVal*> fusion_state;
  
  struct Operators {
    Operators(FusionDefinition* fd) : fusion_definition(fd) {}

    FusionDefinition* fusion_definition;
  };

  Operators ops;
};

} // nvfuser namespace
