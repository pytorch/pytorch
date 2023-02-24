#pragma once
#include <ir_interface_nodes.h>

namespace nvfuser::python_frontend {

struct RecordFunctor;

//! The State and the StateType enum are used to define state objects to
//! encapsulate the recording of state in the FusionDefinition.

enum class StateType {
  Tensor,
  Scalar,
  None,
};

struct TORCH_CUDA_CU_API State {
  State(size_t _index, StateType _stype) : index(_index), stype(_stype) {}

  bool operator==(const State& other) const;
  bool operator!=(const State& other) const;

  //! A unique index to identifiy each recorded state item.
  size_t index;
  //! StateType is either: Tensor or Scalar
  StateType stype;
};

//! FusionState contains the information used to build a new cpp Fusion object.
//! Unlike FusionDefinition, it does not modify the FusionCache Trie structure.
class TORCH_CUDA_CU_API FusionState {
 public:
  FusionState();

  // The copy/move/assign constructors/operators are removed
  FusionState(const FusionState& other) = delete;
  FusionState(FusionState&& other) noexcept = delete;
  FusionState& operator=(const FusionState& other) = delete;
  FusionState& operator=(FusionState&& other) noexcept = delete;
  virtual ~FusionState() = default;

  //! Get fusion object
  Fusion* fusion();
  //! Prints the Fusion IR representation
  void printIr();

  //! Gets a Fusion IR Tensor/Scalar object
  Val* getFusionState(size_t index) const;
  //! Sets a Fusion IR Tensor/Scalar object
  void setFusionState(size_t index, Val* val);

  //! Adds a Tensor/Scalar input to the Fusion object
  void addInput(Val* input);
  //! Adds a Tensor/Scalar output to the Fusion object
  void addOutput(Val* output);
  //! Alias an Output to Input in the Fusion object
  void aliasOutputToInput(Val* output, Val* input);

  //! Add a Record
  void addRecord(RecordFunctor* record);
  //! Builds an nvFuser Fusion IR object
  void buildFusionIr(Fusion* fusion);

  //! Create clone of FusionState
  std::unique_ptr<FusionState> clone();

 private:
  //! Change the fusion ptr and reset its state
  void resetFusionState(Fusion* fusion, size_t size);

 protected:
  //! Holds an End Record
  std::unique_ptr<RecordFunctor> end_record_;
  //! A vector of record operations in the FusionDefintion
  std::vector<std::unique_ptr<RecordFunctor>> recording_;

 private:
  //! A ptr to the container used when building the Fusion IR from a definition
  Fusion* fusion_;
  //! A vector of nvFuser Fusion IR TensorViews/Vals for building the Fusion
  //! IR graph.
  std::vector<Val*> fusion_state_;
  //! The number of states in Fusion Container
  //! A sum of all outputs for each RecordFunctor
  size_t num_recording_states_;
};

} // namespace nvfuser::python_frontend
