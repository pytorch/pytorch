#pragma once
#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

//! nvFuser Fusion IR namespace abbreviation
namespace Nvf = torch::jit::fuser::cuda;

namespace nvfuser {

class FusionCache;
class FusionInterface;
struct RecordFunctor;

//! This is helper function used to print a python formated
//! Fusion IR DataType when printing a fusion definition.

TORCH_CUDA_CU_API const char* dtypeToPyString(Nvf::DataType t);

//! The State and the StateType enum are used to define state objects to
//! encapsulate the recording of state in the FusionDefinition.

enum class StateType {
  Tensor,
  Scalar,
  None,
};

struct State {
  State(size_t _index, StateType _stype) : index(_index), stype(_stype) {}

  //! A unique index to identifiy each recorded state item.
  size_t index;
  //! StateType is either: Tensor or Scalar
  StateType stype;
};

//! The Tensor and Scalar classes are used to define separate function signtures
//! in the FusionDefintion to identify the appropriate Operator function.
//!
//! Example:
//!
//!   add(Tensor* arg1, Tensor* arg2) -> Tensor*
//!   add(Tensor* arg1, Scalar* arg2) -> Tensor*
//!   add(Scalar* arg1, Scalar* arg2) -> Scalar*
struct Tensor {
  Tensor(size_t _index) : index(_index) {}

  size_t operator()() const {
    return index;
  }

  //! A unique index to identifiy each recorded state item.
  size_t index;
};

struct Scalar {
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
  FusionDefinition(FusionInterface* fusion, size_t max_length = 256);

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
  //! Prints a python function representing the definition
  void print(std::ostream& os) const;

  //! These methods are used to record the FusionDefinition for cache lookup

  //! Defines a Scalar State Record
  Scalar defineScalar();
  //! Defines a Tensor State Record
  Tensor defineTensor();
  //! Defines a Record that records the operation required to
  //! build the corresponding Fusion IR operation on cache miss.
  void defineRecord(RecordFunctor* record);
  //! Adds a Tensor/Scalar input to the Fusion object
  void addInput(Nvf::Val* input);
  //! Adds a Tensor/Scalar output to the Fusion object
  void addOutput(Nvf::Val* output);
  //! Gets a Fusion IR Tensor/Scalar object
  Nvf::Val* getFusionState(size_t index) const;
  //! Sets a Fusion IR Tensor/Scalar object
  void setFusionState(size_t index, Nvf::Val* val);
  //! Gets a Record State object
  State recordingState(size_t index) const;

 private:
  //! Builds an nvFuser Fusion IR object upon exit of a FusionDefintion
  //! when a cache lookup fails.
  void buildFusionIr();
  //! Returns the FusionCache Ptr that holds the cache of Fusions
  FusionCache* fusionCachePtr() const;
  //! Returns the FusionInterface Ptr that represents the corresponding
  //! Fusion IR object.
  FusionInterface* fusionInterfacePtr() const;

  //! Holds the defined maximum length of a FusionDefinition in order to
  //! prevent a run away error. The user should feel free to increase this
  //! number as appropriate.
  size_t max_length_;

  //! A pointer to an interface for an nvFusion Fusion IR object.
  FusionInterface* fusion_;
  //! A pointer to the FusionCache.
  FusionCache* fusion_cache_;

  //! Holds an End Record
  std::unique_ptr<RecordFunctor> end_record_;

  //! A vector of record operations in the FusionDefintion
  std::vector<std::unique_ptr<RecordFunctor>> recording_;
  //! A vector of state recorded in the FusionDefinition
  std::vector<State> recording_state_;

  //! A vector of nvFuser Fusion IR TensorViews/Vals for building the Fusion
  //! IR graph.
  std::vector<Nvf::Val*> fusion_state_;

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
