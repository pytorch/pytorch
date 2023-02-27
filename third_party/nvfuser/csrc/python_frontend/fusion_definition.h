#pragma once
#include <iostream>

#include <c10/macros/Export.h>
#include <kernel_cache.h>
#include <python_frontend/fusion_state.h>

namespace nvfuser::python_frontend {

class FusionCache;
class FusionDefinition;
class FusionInterface;
class FusionState;
struct RecordFunctor;
struct UserSchedule;

//! This is helper function used to print a python formated
//! Fusion IR DataType when printing a fusion definition.

TORCH_CUDA_CU_API const char* dtypeToPyString(PrimDataType t);

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
  Tensor(size_t _index, size_t _dims, FusionDefinition* _fd)
      : index(_index), dims(_dims), fusion_definition(_fd) {}

  size_t operator()() const {
    return index;
  }

  //! A unique index to identifiy each recorded state item.
  size_t index;
  size_t dims;

  //! Pointer to the FusionDefinition used to create this tensor
  FusionDefinition* fusion_definition;
};

struct TORCH_CUDA_CU_API Scalar {
  Scalar(size_t _index, FusionDefinition* _fd)
      : index(_index), fusion_definition(_fd) {}

  size_t operator()() const {
    return index;
  }

  //! A unique index to identifiy each recorded state item.
  size_t index;

  //! Pointer to the FusionDefinition used to create this scalar
  FusionDefinition* fusion_definition;
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
class TORCH_CUDA_CU_API FusionDefinition : public FusionState {
 public:
  FusionDefinition(c10::optional<size_t> id, size_t max_length = 256);

  // The copy/move/assign constructors/operators are removed
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
  //! Executes a fusion if a valid definition or cache lookup occurred prior
  std::vector<at::Tensor> execute(
      const at::ArrayRef<c10::IValue>& inputs,
      bool override_user_schedule) const;
  //! Return fusion id of defined FusionDefinition
  c10::optional<size_t> id() const;
  //! Prints the Prescheduled Fusion IR representation
  void printMathIr();

  bool completed() {
    return id().has_value();
  }

  //! These methods are used to record the FusionDefinition for cache lookup

  //! Defines a Scalar State Record
  Scalar defineScalar();
  //! Defines a Tensor State Record
  Tensor defineTensor(size_t dims);
  //! Defines a Record that records the operation required to
  //! build the corresponding Fusion IR operation on cache miss.
  void defineRecord(RecordFunctor* record);
  //! Gets a Record State object
  State recordingState(size_t index) const;

 private:
  //! Returns the FusionCache Ptr that holds the cache of Fusions
  FusionCache* fusionCache() const;
  //! Return a prescheduled Fusion object
  Fusion* preschedFusion();

  //! Holds the defined maximum length of a FusionDefinition in order to
  //! prevent a run away error. The user should feel free to increase this
  //! number as appropriate.
  size_t max_length_;
  //! Fusion Cache Id for Scheduled Fusion.
  c10::optional<size_t> fusion_id_;
  //! A pointer to the FusionCache.
  FusionCache* fusion_cache_;

  //! A vector of state recorded in the FusionDefinition
  std::vector<State> recording_state_;

  // Book keeping data members for user created schedules

  //! Data member for holding previous fusion container when manually setting
  //! the fusion guard.
  Fusion* prev_fusion_;
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
      return !fusion_definition->completed();
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
      return fusion_definition->completed();
    }

    FusionDefinition* fusion_definition;
  };

  Operators ops;
  SchedOperators sched;
};

} // namespace nvfuser::python_frontend
