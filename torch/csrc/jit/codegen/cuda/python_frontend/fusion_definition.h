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
  FusionDefinition(FusionOwner* fusion_owner);

  // The copy/move/assign constructors/operators are being removed
  // because it is not possible to copy the fusion_recording data member
  // because that would require a virtual copy/move/assign of the
  // RecordFunctor that is not supported.
  FusionDefinition(const FusionDefinition& fd) = delete;
  FusionDefinition(FusionDefinition&& fd) = delete;
  FusionDefinition& operator=(const FusionDefinition& fd) = delete;
  FusionDefinition& operator=(FusionDefinition&& fd) = delete;

  // Context Manager Methods
  FusionDefinition* enter();
  void exit();

  void addInput(NvfVal* input);
  void addOutput(NvfVal* output);

  Scalar* defineScalar();
  Tensor* defineTensor();
  void defineRecord(RecordFunctor* record);

  NvfVal* getFusionState(size_t index) const;
  void setFusionState(size_t index, NvfVal* val);

  Fusion* fusionPtr();

 private:
  FusionOwner* fusion_owner_;
  Fusion* prev_fusion_;

  std::vector<std::unique_ptr<RecordFunctor>> recording_;
  std::vector<std::unique_ptr<State>> recording_state_;

  std::vector<NvfVal*> fusion_state_;

 public:
  struct Operators {
    Operators(FusionDefinition* fd) : fusion_definition(fd) {}

    FusionDefinition* fusion_definition;
  };

  Operators ops;
};

} // namespace nvfuser
