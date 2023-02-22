#pragma once
#include <iostream>

#include <c10/macros/Export.h>
#include <kernel_cache.h>

namespace nvfuser::python_frontend {

class FusionCache;
class FusionInterface;
struct RecordFunctor;
struct UserSchedule;

//! This is helper function used to print a python formated
//! Fusion IR DataType when printing a fusion definition.

TORCH_CUDA_CU_API const char* dtypeToPyString(nvfuser::PrimDataType t);

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

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& os,
    const State& state);

//! The Tensor and Scalar classes are used to define separate function signtures
//! in the FusionDefintion to identify the appropriate Operator function.
//!
//! Example:
//!
//!   add(Tensor* arg1, Tensor* arg2) -> Tensor*
//!   add(Tensor* arg1, Scalar* arg2) -> Tensor*
//!   add(Scalar* arg1, Scalar* arg2) -> Scalar*
struct TORCH_CUDA_CU_API Tensor {
  Tensor(size_t _index, size_t _dims) : index(_index), dims(_dims) {}

  size_t operator()() const {
    return index;
  }

  //! A unique index to identifiy each recorded state item.
  size_t index;
  size_t dims;
};

struct TORCH_CUDA_CU_API Scalar {
  Scalar(size_t _index) : index(_index) {}

  size_t operator()() const {
    return index;
  }

  //! A unique index to identifiy each recorded state item.
  size_t index;
};

//! FusionDefinition defines the C++ side of a Python Context manager to
//! encapsulate the definition of fusion operations.
//!
//! The FusionDefinition records the state definitions and operations prior
//! to exiting the context manager.  Upon exit, the operations are queried
//! in a cache and the recorded records are used to build an nvFuser Fusion
//! object if the definition missed in the cache.
//!
//! The nested Operators class was designed to allow the user to query all the
//! available Operators in the FusionDefinition via python help.
//!
//! Example:
//!   help(FusionDefinition.Operators)
class TORCH_CUDA_CU_API FusionDefinition {
 public:
  FusionDefinition(c10::optional<size_t> id, size_t max_length = 256);

  // The copy/move/assign constructors/operators are being removed
  // because it is not possible to copy the fusion_recording data member
  // because that would require a virtual copy/move/assign of the
  // RecordFunctor that is not supported.
  FusionDefinition(const FusionDefinition& fd) = delete;
  FusionDefinition(FusionDefinition&& fd) = delete;
  FusionDefinition& operator=(const FusionDefinition& fd) = delete;
  FusionDefinition& operator=(FusionDefinition&& fd) = delete;

  //! Enter Python Context Manager -- Reset trie for new cache lookup
  FusionDefinition* setupDefinition();
  //! Exit Python Context Manager -- Triggers Fusion IR build if it is not
  //! cached
  void finalizeDefinition();
  //! Setup user scheduling of a fusion
  //! Copies fusion object and sets up FusionGuard
  void setupSchedule(const at::ArrayRef<c10::IValue>& inputs);
  //! Finalized use scheduling of a fusion
  //! resets FusionGuard, lowers IR to a kernel, compiles kernel
  void finalizeSchedule(const at::ArrayRef<c10::IValue>& inputs);
  //! Prints a python function representing the definition
  void print(std::ostream& os) const;
  //! Prints the Fusion IR representation of an unscheduled fusion
  void printIr();
  //! Executes a fusion if a valid definition or cache lookup occurred prior
  std::vector<at::Tensor> execute(
      const at::ArrayRef<c10::IValue>& inputs,
      bool override_user_schedule) const;
  //! Return fusion id of defined FusionDefinition
  c10::optional<size_t> id() const;

  //! These methods are used to record the FusionDefinition for cache lookup

  //! Defines a Scalar State Record
  Scalar defineScalar();
  //! Defines a Tensor State Record
  Tensor defineTensor(size_t dims);
  //! Defines a Record that records the operation required to
  //! build the corresponding Fusion IR operation on cache miss.
  void defineRecord(RecordFunctor* record);
  //! Adds a Tensor/Scalar input to the Fusion object
  void addInput(nvfuser::Val* input);
  //! Adds a Tensor/Scalar output to the Fusion object
  void addOutput(nvfuser::Val* output);
  //! Alias an Output to Input in the Fusion object
  void aliasOutputToInput(nvfuser::Val* output, nvfuser::Val* input);
  //! Gets a Fusion IR Tensor/Scalar object
  nvfuser::Val* getFusionState(size_t index) const;
  //! Sets a Fusion IR Tensor/Scalar object
  void setFusionState(size_t index, nvfuser::Val* val);
  //! Adds a Fusion IR Tensor/Scalar object
  void addFusionState(size_t index, nvfuser::Val* val);
  //! Gets a Record State object
  State recordingState(size_t index) const;

 private:
  //! Builds an nvFuser Fusion IR object upon exit of a FusionDefintion
  //! when a cache lookup fails.
  void buildFusionIr();
  //! Returns the FusionCache Ptr that holds the cache of Fusions
  FusionCache* fusionCache() const;
  //! Return a prescheduled Fusion object
  nvfuser::Fusion* preschedFusion();

  //! Holds the defined maximum length of a FusionDefinition in order to
  //! prevent a run away error. The user should feel free to increase this
  //! number as appropriate.
  size_t max_length_;
  //! Fusion Cache Id for Scheduled Fusion.
  c10::optional<size_t> fusion_id_;
  //! A pointer to the FusionCache.
  FusionCache* fusion_cache_;
  //! A ptr to the container used when building the Fusion IR from a definition
  nvfuser::Fusion* fusion_;

  //! Holds an End Record
  std::unique_ptr<RecordFunctor> end_record_;

  //! A vector of record operations in the FusionDefintion
  std::vector<std::unique_ptr<RecordFunctor>> recording_;
  //! A vector of state recorded in the FusionDefinition
  std::vector<State> recording_state_;

  //! A vector of nvFuser Fusion IR TensorViews/Vals for building the Fusion
  //! IR graph.
  std::vector<nvfuser::Val*> fusion_state_;

  // Book keeping data members for user created schedules

  //! Data member for holding previous fusion container when manually setting
  //! the fusion guard.
  nvfuser::Fusion* prev_fusion_;
  //! Data member for holding the current user schedule object
  UserSchedule* user_sched_;

 public:
  //! The Operators are not directly defined in this header.  They are defined
  //! in the python bindings through lambda functions so the user only needs to
  //! define new operators in one place.
  //! Operators define what operations are fused.
  struct Operators {
    Operators(FusionDefinition* fd) : fusion_definition(fd) {}
    bool validUse() const {
      return !fusion_definition->id().has_value();
    }

    FusionDefinition* fusion_definition;
  };

  //! The SchedOperators are not directly defined in this header.  They are
  //! defined in the python bindings through lambda functions so the user only
  //! needs to define new operators in one place.
  //! SchedOperators allow the user to define how a fusion should be blocked
  //! for execution.
  struct SchedOperators {
    SchedOperators(FusionDefinition* fd) : fusion_definition(fd) {}
    bool validUse() const {
      return fusion_definition->id().has_value();
    }

    FusionDefinition* fusion_definition;
  };

  Operators ops;
  SchedOperators sched;
};

} // namespace nvfuser::python_frontend
