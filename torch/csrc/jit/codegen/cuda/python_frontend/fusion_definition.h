#pragma once
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_owner.h>

using NvfDataType = torch::jit::fuser::cuda::DataType;
using NvfTensorView = torch::jit::fuser::cuda::TensorView;
using NvfVal = torch::jit::fuser::cuda::Val;

namespace nvfuser {

struct RecordFunctor;

//! The State, child classes Tensor and Scalar, and the StateType enum
//! are used to define state objects to encapsulate the recording of state
//! in the FusionDefinition.

enum class StateType {
  Tensor,
  Scalar,
  None,
};

struct State {
  State(StateType _stype, size_t _index) : stype(_stype), index(_index) {}

  //! StateType is either: Tensor or Scalar
  StateType stype;
  //! A unique index to identifiy each recorded state item.
  size_t index;
};

//! The child classes are used to define separate function signtures in
//! in the FusionDefintion to identify the appropriate Operator function.
//!
//! Example:
//!
//!   add(Tensor* arg1, Tensor* arg2) -> Tensor*
//!   add(Tensor* arg1, Scalar* arg2) -> Tensor*
//!   add(Scalar* arg1, Scalar* arg2) -> Scalar*
struct Tensor : State {
  Tensor(size_t _index) : State(StateType::Tensor, _index) {}
};

struct Scalar : State {
  Scalar(size_t _index) : State(StateType::Scalar, _index) {}
};

//! FusionDefinition defines the C++ side of a Python Context manager to
//! encapsulate the definition of fusion operations.
//!
//! The FusionDefinition records the state definitions and operations prior
//! to exiting the context manager.  Upon exit, the operations are queried
//! in a cache and the recorded records are used to build an nvFuser Fusion
//! object if the definition missed in the cache.
//!
//! \todo Need to implement the cache portion. Currently, the Fusion object
//! is always built.
//!
//! The nested Operators class was designed to allow the user to query all the
//! available Operators in the FusionDefinition via python help.
//!
//! Example:
//!   help(FusionDefinition.Operators)
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
