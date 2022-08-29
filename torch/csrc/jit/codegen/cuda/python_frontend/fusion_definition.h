#pragma once
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_owner.h>

//! nvFuser Fusion IR Types
using NvfDataType = torch::jit::fuser::cuda::DataType;
using NvfFusion = torch::jit::fuser::cuda::Fusion;
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

  //! Enter Python Context Manager
  FusionDefinition* enter();
  //! Exit Python Context Manager -- Triggers cache lookup
  void exit();

  //! These methods are used to record the FusionDefinition for cache lookup

  //! Defines a Scalar State Record
  Scalar* defineScalar();
  //! Defines a Tensor State Record
  Tensor* defineTensor();
  //! Defines a Record that records the operation required to
  //! build the corresponding Fusion IR operation on cache miss.
  void defineRecord(RecordFunctor* record);

  //! These methods are used to replay the operations for building the
  //! nvFuser Fusion IR on a cache miss.

  //! Adds a Tensor/Scalar input to the Fusion object
  void addInput(NvfVal* input);
  //! Adds a Tensor/Scalar output to the Fusion object
  void addOutput(NvfVal* output);
  //! Gets a Fusion IR Tensor/Scalar object
  NvfVal* getFusionState(size_t index) const;
  //! Sets a Fusion IR Tensor/Scalar object
  void setFusionState(size_t index, NvfVal* val);

  //! A pointer to the nvFuser Fusion IR Oject
  NvfFusion* fusionPtr();

 private:
  // \todo These items will be replaced by a FusionManager instead of a cache
  // for an individual fusion object
  FusionOwner* fusion_owner_;
  NvfFusion* prev_fusion_;

  //! A vector of record operations in the FusionDefintion
  std::vector<std::unique_ptr<RecordFunctor>> recording_;
  //! A vector of state (Tensor/Scalar) recorded in the FusionDefinition
  std::vector<std::unique_ptr<State>> recording_state_;

  //! A vector of nvFuser Fusion IR TensorViews/Vals for building the Fusion
  //! IR graph.
  std::vector<NvfVal*> fusion_state_;

 public:
  //! The Operators are not directly defined in this header.  They are defined
  //! in the python bindings through lambda functions so the user only needs to
  //! define new operators in one place.
  struct Operators {
    Operators(FusionDefinition* fd) : fusion_definition(fd) {}

    FusionDefinition* fusion_definition;
  };

  Operators ops;
};

} // namespace nvfuser
