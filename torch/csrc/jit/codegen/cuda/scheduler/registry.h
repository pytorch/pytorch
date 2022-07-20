#pragma once
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/compile_time_info.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SegmentedGroup;
class ExpressionEvaluator;

//!  SchedulerRuntimeInfo is the abstraction introduced in
//! this PR for passing runtime input dependent information
//! to the schedulers and kernel caches.
//!
//! Note:
//!  if any additional info needed,  or maybe just the inputs themselves it
//!    could just be added to this class, and they will be distributed to the
//!    segmenter and schedulers.
//!  It is important that input id encoding should be up to date with any change
//!   of this class to avoid launching compiled kernels with illegal inputs.
class TORCH_CUDA_CU_API SchedulerRuntimeInfo : public NonCopyable {
 public:
  // Max vector size we will consider, in bytes,
  //  currently set to 16B = 128b
  static constexpr size_t max_alignment_size_in_byte = 16;

  //! Create runtime info for given fusion and input. Creating and binding
  //! evaluator is optional. The evaluator is used to manage intermediate
  //!  integers in the fusion. We need them for segmenter and schedulers,
  //!  but we don't need them when we are just using this class to provide
  //!  additional encoding for kernel cache lookup.
  SchedulerRuntimeInfo(
      Fusion* complete_fusion,
      const at::ArrayRef<at::IValue>& inputs,
      bool create_expr_evaluator = false);

  //! Lookup for the alignment sizes of the given tv. Currently only returns
  //!  actual alignment info for input tensors to the complete fusion,
  //!  and for other intermediate/fuser-allocated tensors will
  //!  return max_alignment_size_in_byte.
  size_t getAlignmentSize(TensorView* tv);

  // Gets maximum vectorizable width of tv, assumes we can merge across all
  // iteration domains if contiguous. Cannot permute the dimensions to fix
  // contiguity. Ignores dimensions that are broadcast or reduction.
  size_t getMaxVectorizableWidth(TensorView* tv);

  // Gets the vectorizable width of the inner most dimension of tv if it's
  // contiguous. Ignores inner most dimensions that are broadcast or reduction.
  size_t getInnerDimVectorizableWidth(TensorView* tv);

  // Computes alignment size in bytes for provided ptr address
  static size_t computeAlignmentSize(size_t ptr_address);

  // Return the runtime pointer value for provided tensor view
  size_t ptrOf(TensorView* tv);

  KernelIndexMode getIndexMode() {
    return index_mode_;
  }

  Fusion* fusion() {
    return complete_fusion_;
  }

  ExpressionEvaluator& expressionEvaluator() {
    TORCH_INTERNAL_ASSERT(expression_evaluator_ != nullptr);
    return *expression_evaluator_;
  }

 private:
  // Bind full fusion inputs to the internal expression evaluator
  void initializeExpressionEvaluator(const at::ArrayRef<at::IValue>& inputs);

  // check if input is compatible with 32b index mode
  void collectIndexModeInfo(const at::ArrayRef<at::IValue>& inputs);

 private:
  bool isInputTv(TensorView* tv) {
    return std::find(
               complete_fusion_->inputs().begin(),
               complete_fusion_->inputs().end(),
               tv) != complete_fusion_->inputs().end();
  }

  // Returns the offset of tv in the inputs ignoring non tensor views. Used to
  // access input_sizes, input_strides, input_ptr
  int offsetTensorPos(TensorView* tv);

  // Expression evaluator used to probe sizes in the fusion IR
  std::unique_ptr<ExpressionEvaluator> expression_evaluator_ = nullptr;

  // Fusion reference that this runtime info is associated with
  Fusion* complete_fusion_ = nullptr;

  // Copy of aten input pointer addresses
  // TODO: Support output tensor pointers
  std::unordered_map<Val*, size_t> input_ptrs_;

  // Cache for getAlignmentSize
  std::unordered_map<TensorView*, size_t> alignment_map_;
  // Cache for getMaxVectorizableWidth
  std::unordered_map<TensorView*, size_t> max_vectorword_map_;
  // Cache for getInnerDimVectorizableWidth
  std::unordered_map<TensorView*, size_t> inner_vectorword_map_;

  // Found index mode kernel needs to be run in
  KernelIndexMode index_mode_ = KernelIndexMode::INT64;

  // TODO: Remove
  std::unordered_map<TensorView*, size_t> vectorword_map_;
};

class HeuristicSummary;

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
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  virtual ~SchedulerEntry() = default;

  //! External access for canSchedule utilities through SchedulerEntry
  //!  to avoid exposing a single function to the namespace
  static bool canSchedule(
      ScheduleHeuristic sh,
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  //! Fusion segmenter facing API,
  //!   returns a schedule that applies in the given fusion, returns a nullopt
  //!   if no schedule in the registry can handle.
  static c10::optional<ScheduleHeuristic> proposeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info);

  //! Fusion runtime facing API,
  //!   schedule the given fusion with heuristics owned
  //!   by this entry, for actual heuristics to override
  virtual void schedule(Fusion* fusion) = 0;

  //! Heuristic comparison
  bool sameAs(const SchedulerEntry* other);

  bool hasReductionParam() const {
    return has_reduction_param_;
  }

  ScheduleHeuristic heuristc() const {
    return heuristc_;
  }

  KernelIndexMode indexMode() const {
    return index_mode_;
  }

  const ReductionParams& reductionParams() const {
    TORCH_INTERNAL_ASSERT(
        has_reduction_param_, "This schedule heuristic is not reduction.");
    return rparams_;
  }

  const PointwiseParams& pointwiseParams() const {
    TORCH_INTERNAL_ASSERT(
        !has_reduction_param_, "This schedule heuristic is not pointwise.");
    return pparams_;
  }

  void updateLaunchConstraint(const LaunchParams& launch_params) {
    if (hasReductionParam()) {
      rparams_.lparams = launch_params;
    } else {
      pparams_.lparams = launch_params;
    }
  }

 protected:
  explicit SchedulerEntry(ScheduleHeuristic heuristic, bool has_reduction_param)
      : heuristc_(heuristic), has_reduction_param_(has_reduction_param) {}

  ReductionParams& rparams() {
    return rparams_;
  }

  PointwiseParams& pparams() {
    return pparams_;
  }

 private:
  //! What kind of heuristics does this entry have?
  const ScheduleHeuristic heuristc_;

  //! Has reduction params if true, else has pointwise params
  const bool has_reduction_param_;

  //! Reduction parameters if applicable
  ReductionParams rparams_;

  //! Pointwise parameters if applicable
  PointwiseParams pparams_;

  //! Kernel Index Mode
  KernelIndexMode index_mode_ = KernelIndexMode::INT64;
};

//! Hash function for a scheduler entry
class TORCH_CUDA_CU_API SchedulerEntryHash {
 public:
  size_t operator()(const SchedulerEntry& se) const;
};

//! Debug print function for heuristics
std::string toString(ScheduleHeuristic sh);

//! Debug print function for heuristics
std::ostream& operator<<(std::ostream& os, ScheduleHeuristic sh);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
