#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! namespace for hosting catalog of possible compile time
//!  info that can be cached. Each possible entry type has
//!  a value in `CompileTimeEntryType` and an entry type class
//!  definition like `VectorizableInputsAndOutputs`. The corresponnding
//!  classes contain their entry type, data type and maybe more
//!  later depending on use cases.
namespace HeuristicCompileTime {

//! Each entry type under this category represent some information
//!  that can be inferred compile-time, i.e. without any runtime input
//!  meta data. They will be stored in `HeuristicSummary` and will
//!  be re-used each time the same fusion is visited.

//! Enum for all possible types of cached entries of compile-time info.
enum class CompileTimeEntryType {
  DOMAIN_MAP,
  TRANSPOSE_DOMAIN_MAP,
  REFERENCE_TENSORS,
  REFERENCE_TENSORS_FOR_GROUPS,
  VECTORIZABLE_INPUTS_AND_OUTPUTS,
  INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS,
  UNROLLABLE_INPUTS_AND_OUTPUTS,
  REDUCTION_TVS,
  PERSISTENT_BUFFER_INFO,
  SCOPE_PERSISTENT_FACTOR_INFO,
  BROADCAST_BYTE_MULTIPLES,
  INNER_MOST_DIMS_INFO,
  CAN_SCHEDULE_TRANSPOSE,
};

//! Entry type definition class for `DOMAIN_MAP`,
//!  stores the domain map of a fusion.
class DomainMap {
 public:
  using DataType = pointwise_utils::DomainMap;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::DOMAIN_MAP;
};

//! Entry type definition class for `DOMAIN_MAP`,
//!  stores the domain map of a fusion, used by transpose scheduler.
class TransposeDomainMap {
 public:
  using DataType = pointwise_utils::DomainMap;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::TRANSPOSE_DOMAIN_MAP;
};

//! Entry type definition class for `REFERENCE_TENSORS`,
//!  stores the the reference TensorViews used to schedule a fusion.
class ReferenceTensors {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::REFERENCE_TENSORS;
};

//! Entry type definition class for `REFERENCE_TENSORS`,
//!  stores the the reference TensorViews used to schedule a fusion, used by
//!  transpose scheduler.
class ReferenceTensorsForGroups {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::REFERENCE_TENSORS_FOR_GROUPS;
};

//! Entry type definition class for `VECTORIZABLE_INPUTS_AND_OUTPUTS`,
//!  stores the vectorizable TensorViews on a fusion's inputs and outputs.
class VectorizableInputsAndOutputs {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS;
};

//! Entry type definition class for `INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS`,
//!  stores the fusion's inputs and outputs grouped by inner most dimension.
class InputsOutputsInnerDimGroups {
 public:
  using DataType = std::vector<std::vector<TensorView*>>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS;
};

//! Entry type definition class for `UNROLLABLE_INPUTS_AND_OUTPUTS`,
//!  stores the unrollable TensorViews on a fusion's inputs and outputs.
class UnrollableInputsAndOutputs {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::UNROLLABLE_INPUTS_AND_OUTPUTS;
};

//! Entry type definition class for `REDUCTION_TVS`,
//!  stores the all tvs with reduction axes in a fusion.
class ReductionTVs {
 public:
  using DataType = std::vector<TensorView*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::REDUCTION_TVS;
};

//! Entry type definition class for `PERSISTENT_BUFFER_INFO`,
//!  stores persistent buffers inferred from topology and scheduling of fusion.
class PersistentBufferInfo {
 public:
  using DataType = scheduler_utils::PersistentBufferInfo;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::PERSISTENT_BUFFER_INFO;
};

//! Entry type definition class for `INNER_MOST_DIMS_INFO`,
//!  Used in the transpose scheduler to store inner most IterDomains and their
//!  position in reference1 of group 1 and group 2
class InnerMostDimInfo {
 public:
  using DataType = std::vector<int64_t>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::INNER_MOST_DIMS_INFO;
};

//! Auxiliary data types for `SCOPE_PERSISTENT_FACTOR_INFO` entry type.
using ScopedPersistenceBufferMap = std::unordered_map<Val*, std::vector<bool>>;

//! Entry type definition class for `SCOPE_PERSISTENT_FACTOR_INFO`,
// Tracks which buffers are active at a given Val*, order of bool vector is
// based on persistence buffer order from persistence buffer info, this is then
// appended by the projectable persistent buffers' inputs. True in the bool
// vector means the persistent buffer is active at the generation of the key.
class ScopePersistentFactorInfo {
 public:
  using DataType = ScopedPersistenceBufferMap;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::SCOPE_PERSISTENT_FACTOR_INFO;
};

//! Entry type definition class for `BROADCAST_BYTE_MULTIPLES`,
//!  stores "byte multiples" information. This information can be used to figure
//!  out if using a 2D scheduler how many bytes have to be transferred with
//!  varying split locations. See BroadcastMultiple definition for more
//!  information.
class BroadcastMultiples {
 public:
  using DataType = scheduler_utils::BroadcastMultipleInformation;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::BROADCAST_BYTE_MULTIPLES;
};

//! Entry type definition class for `CAN_SCHEDULE_TRANSPOSE`,
//!  stores if the transpose scheduler can scheduler this fusion
class CanScheduleTranspose {
 public:
  using DataType = bool;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::CAN_SCHEDULE_TRANSPOSE;
};

//! Base abstract class for unified storage in `HeuristicSummary`,
//!  each entry in `HeuristicSummary` will be a subclass.
class CompileTimeInfoBase : public PolymorphicBase {
 public:
  CompileTimeInfoBase(CompileTimeEntryType entry_type)
      : entry_type_(entry_type) {}
  CompileTimeEntryType type() {
    return entry_type_;
  }

 private:
  CompileTimeEntryType entry_type_;
};

} // namespace HeuristicCompileTime

// Note: Do NOT export this class. MSVC issue with exported class that contains
// std::vector<unique_ptr<xxx>>: https://godbolt.org/z/3E4e8T1P1
//! Compile-time information cache for `canSchedule` and
//!  `getHeuristics` interfaces. Each cache instance
//!  stores information that could be inferred at compile
//!  time in a fusion and therefore corresponds to an
//!   instance of FusionExecutor.
//!  Since each instance of FusionExecutor has a unique
//!   heuristic type, this cache also has a heuristic
//!   type to simplify data validation.
//!  HeuristicSummary has two modes of operation:
//!  - when in `recording` mode, the information is not available
//!     in the cache and entries can be written and stored.
//!  - when not in `recording` mode, compiled-time data has
//!     been stored in this cache and the entries can be accessed
//!!    but new entries can no longer be inserted.
class HeuristicSummary {
  using Entry = HeuristicCompileTime::CompileTimeInfoBase;
  using EntryOwningPtr = std::unique_ptr<Entry>;
  using EntryPtr = Entry*;
  using EntryType = HeuristicCompileTime::CompileTimeEntryType;

 public:
  HeuristicSummary(
      Fusion* fusion,
      ScheduleHeuristic heuristic,
      SchedulerRuntimeInfo& runtime_info);

  bool isRecording() {
    return recording_;
  }

  void insert(EntryOwningPtr new_entry);

  EntryPtr at(EntryType entry_type) {
    return entry_type_map_.at(entry_type);
  }

 private:
  void validate() const;

 private:
  std::vector<EntryOwningPtr> entries_;
  std::unordered_map<EntryType, EntryPtr> entry_type_map_;
  ScheduleHeuristic heuristic_;
  bool recording_ = true;
};

//! A utility class to facilitate accessing HeuristicSummary.
//!  This utility is needed because the information to be stored
//!    in HeuristicSummary is used in several different scenarios
//!    and we want to support all these use cases in one interface.
//!  The current use examples are:
//!   1. During fusion segmentation process, all the fusions
//!     given to canSchedule are temporary and therefore the
//!     compile time info do not need to be cached, and in fact
//!     a cache wouldn't be instantiated by that time.
//!
//!   2. When the compiled kernel is launched the first time, the
//!     cache will be in `recording` phase and all the computed information
//!     should be captured and written into the cache.
//!
//!   3. When we check a compiled fusion for heuristic hit,
//!      we want to use the cached info to save runtime latency.
//!
//! The designed interface is used as:
//!   auto entry = HeuristicSummaryEntry<EntryClass>(data_cache, maker_fn);
//!   auto& data = entry.get();
//!
//!  `maker_fn` will be called to compute the information when no cached data
//!   exists and `entry` will own the computed data when no data cache is
//!   supplied.
template <typename EntryClass>
class HeuristicSummaryEntry {
  using EntryDataType = typename EntryClass::DataType;
  using EntryDataTypeOwnPtr = std::unique_ptr<EntryDataType>;
  using MakerFnType = std::function<EntryDataTypeOwnPtr()>;

 public:
  //! Creates a data entry with type defined in EntryClass,
  //!  eg. EntryClass = VectorizableInputsAndOutputs;
  //!
  //! @param data_cache, a pointer to an instantiated compile-time
  //!  info cache. The info data will be
  //!    1. read from data cache if data cache is not recording.
  //!    2. written into  data cache if data cache is recording.
  //!    3. managed by owned_data_ if data cache is nullptr
  //! @param fn:
  //!   The factory function that needs to return a owning pointer
  //!  i.e. std::unique_ptr<EntryClass::DataType>. It will only
  //!  be called either when data cache is recording or when no data
  //!  cache is given.
  HeuristicSummaryEntry(HeuristicSummary* data_cache, MakerFnType fn);

  //! Unified interface to get actual data, either from cache
  //!  or from factory function.
  EntryDataType& get() {
    return *data_ptr_;
  }

 private:
  //! Internal data owing pointer that will manage the computed
  //!  data where there is no data cache.
  EntryDataTypeOwnPtr owned_data_ = nullptr;

  //! Pointer to the valid data entry that could be accessed.
  EntryDataType* data_ptr_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
