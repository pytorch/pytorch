#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Virtual base class for schedule heuristics
//!   heuristic implementations derive from this
//!   class and implement a schedule(Fusion*)
//!   and a bool canSchedule(Fusion*) interface
class TORCH_CUDA_CU_API SchedulerEntry {
 public:
  //! Fusion runtime facing API,
  //!   builds a new entry with the given heuristics
  //!   corresponding to the given fusion
  static std::unique_ptr<SchedulerEntry> makeEntry(
      ScheduleHeuristic sh,
      Fusion* fusion,
      ExpressionEvaluator& ee);

  virtual ~SchedulerEntry() = default;

  //! Fusion segmenter facing API,
  //!   returns a schedule that applies in the given fusion, returns a nullopt
  //!   if no schedule in the registry can handle.
  static c10::optional<ScheduleHeuristic> proposeHeuristics(Fusion* fusion);

  //! Fusion runtime facing API,
  //!   schedule the given fusion with heuristics owned
  //!   by this entry, for actual heuristics to override
  virtual void schedule(Fusion* fusion) = 0;

  //! Heuristic comparison
  bool sameAs(const SchedulerEntry* other);

  bool hasParam() const {
    return has_param_;
  }

  ScheduleHeuristic heuristc() const {
    return heuristc_;
  }

  const ReductionParams& params() const {
    return rparams_;
  }

 protected:
  explicit SchedulerEntry(ScheduleHeuristic heuristic, bool has_param)
      : heuristc_(heuristic), has_param_(has_param) {}

  //! What kind of heuristics does this entry have?
  const ScheduleHeuristic heuristc_;

  //! Does this entry have any parameter?
  const bool has_param_;

  //! What are the schedule parameters, if any?
  ReductionParams rparams_;
};

//! Hash function for a scheduler entry
class TORCH_CUDA_CU_API SchedulerEntryHash {
 public:
  size_t operator()(const SchedulerEntry& se) const;
};

//! Debug print function for heuristics
std::string toString(ScheduleHeuristic sh);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
