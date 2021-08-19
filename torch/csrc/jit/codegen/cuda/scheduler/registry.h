#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
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
class TORCH_CUDA_CU_API SchedulerRuntimeInfo {
 public:
  // Max vector size we will consider, in bytes,
  //  currently set to 16B = 128b
  const size_t max_alignment_size_in_byte = 16;

  //! Create runtime info for given fusion and input. Creating and binding
  //! evaluator is optional. The evaluator is used to manage intermediate
  //!  integers in the fusion. We need them for segmenter and schedulers,
  //!  but we don't need them when we are just using this class to provide
  //!  additional encoding for kernel cache lookup.
  SchedulerRuntimeInfo(
      Fusion* complete_fusion,
      const at::ArrayRef<at::IValue>& inputs,
      bool create_expr_evaluator = false);

  //! Create runtime info by copying all the global
  //! input meta data (i.e. alignment), but not the
  //! expression evaluator.
  SchedulerRuntimeInfo(const SchedulerRuntimeInfo& global_runtime_info);

  //! Lookup for the alignment sizes of the given tv. Currently only returns
  //!  actual alignment info for input tensors to the complete fusion,
  //!  and for other intermediate/fuser-allocated tensors will
  //!  return max_alignment_size_in_byte.
  size_t getAlignmentSize(TensorView* tv);

  //! Take the minimum of input tv alignment sizes. This is both information for
  //! vectorization and
  //!  a signature for kernel cache id lookup. May need to be updated with
  //!  vectorization logic.
  size_t getCommonAlignmentSize() const {
    return common_alignment_size_;
  }

  //! Returns the max width the given tensor view can be vectorized,
  //!  for input tensors will use the pre-computed value based on
  //!  the given tensor alignment and strides. For intermediate tensors
  //!  will assume it is contiguous and aligned to 128bit/16Byte
  size_t getVectorizableWidth(TensorView* tv);

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

  // Compute alignment data for all input tensors of full fusion
  void collectVectorizationInfo(const at::ArrayRef<at::IValue>& inputs);

  // Compute alignment data for given tensor
  size_t collectAlignmentSize(const at::Tensor& tensor) const;

  // Compute max vectorization word size for each an input tensor
  size_t collectMaxVectorizeSize(
      const at::Tensor& tensor,
      size_t max_word_size_in_byte);

  // check if input is compatible with 32b index mode
  void collectIndexModeInfo(const at::ArrayRef<at::IValue>& inputs);

 private:
  std::unique_ptr<ExpressionEvaluator> expression_evaluator_ = nullptr;
  Fusion* complete_fusion_;
  std::unordered_map<TensorView*, size_t> alignment_map_;
  std::unordered_map<TensorView*, size_t> vectorword_map_;
  size_t common_alignment_size_;
  KernelIndexMode index_mode_ = KernelIndexMode::INT64;
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

  //! What kind of heuristics does this entry have?
  const ScheduleHeuristic heuristc_;

  //! Has reduction params if true, else has pointwise params
  const bool has_reduction_param_;

  //! Reduction parameters if applicable
  ReductionParams rparams_;

  //! Pointwise parameters if applicable
  PointwiseParams pparams_;

  //! Kernel Index Mode
  KernelIndexMode index_mode_;
};

//! Hash function for a scheduler entry
class TORCH_CUDA_CU_API SchedulerEntryHash {
 public:
  size_t operator()(const SchedulerEntry& se) const;
};

//! Debug print function for heuristics
std::string toString(ScheduleHeuristic sh);

class TORCH_CUDA_CU_API HeuristicSummary {
  using ValToFactorMap = std::unordered_map<Val*, int>;
  using ValToFactorMapPtr = std::unique_ptr<ValToFactorMap>;
  using ScopedPersistenceFactorMap =
      std::unordered_map<Val*, ValToFactorMapPtr>;

 public:
  HeuristicSummary(
      Fusion* fusion,
      ScheduleHeuristic heuristic,
      SchedulerRuntimeInfo& runtime_info);
  // Recording scheme:
  bool isRecording() {
    return recording_;
  }

  // Validate post recording:
  //  make sure we have collected all the needed fields
  void validate() {
    switch (heuristic_) {
      case ScheduleHeuristic::PointWise:
        TORCH_INTERNAL_ASSERT(vectorizable_inputs_outputs_);
        TORCH_INTERNAL_ASSERT(mapped_input_output_dims_);
        break;
      case ScheduleHeuristic::Reduction:
        TORCH_INTERNAL_ASSERT(reduction_tvs_);
        break;
      case ScheduleHeuristic::Normalization:
        TORCH_INTERNAL_ASSERT(vectorizable_inputs_outputs_);
        TORCH_INTERNAL_ASSERT(reduction_tvs_);
        TORCH_INTERNAL_ASSERT(persistent_buffer_info_);
        TORCH_INTERNAL_ASSERT(has_post_reduction_bcast_);
        TORCH_INTERNAL_ASSERT(supported_post_reduction_fusion_);
        break;
    }
  }

  // Accessors (un-protected for now)
  void setVectorizableInputsOutputs(const std::vector<TensorView*>& input) {
    TORCH_INTERNAL_ASSERT(recording_);

    if (!vectorizable_inputs_outputs_) {
      vectorizable_inputs_outputs_ =
          std::make_unique<std::vector<TensorView*>>(input);
    }
  }

  auto* getVectorizableInputsOutputs() {
    return vectorizable_inputs_outputs_.get();
  }

  void setReductionTVs(const std::vector<TensorView*>& input) {
    TORCH_INTERNAL_ASSERT(recording_);

    if (!reduction_tvs_) {
      reduction_tvs_ = std::make_unique<std::vector<TensorView*>>(input);
    }
  }

  auto* getReductionTVs() {
    return reduction_tvs_.get();
  }

  void setPersistentBufferInfo(
      const scheduler_utils::PersistentBufferInfo& input) {
    TORCH_INTERNAL_ASSERT(recording_);

    if (!persistent_buffer_info_) {
      persistent_buffer_info_ =
          std::make_unique<scheduler_utils::PersistentBufferInfo>(input);
    }
  }

  auto* getPersistentBufferInfo() {
    return persistent_buffer_info_.get();
  }

  void setSupportedPostReductionFusion(bool input) {
    TORCH_INTERNAL_ASSERT(recording_);

    if (!supported_post_reduction_fusion_) {
      supported_post_reduction_fusion_ = std::make_unique<bool>(input);
    }
  }

  auto* getSupportedPostReductionFusion() {
    return supported_post_reduction_fusion_.get();
  }

  void setHasPostReductionBCast(bool input) {
    TORCH_INTERNAL_ASSERT(recording_);

    if (!has_post_reduction_bcast_) {
      has_post_reduction_bcast_ = std::make_unique<bool>(input);
    }
  }

  auto* getHasPostReductionBCast() {
    return has_post_reduction_bcast_.get();
  }

  void setScopedPersistenceFactorMap(const ScopedPersistenceFactorMap& input) {
    TORCH_INTERNAL_ASSERT(recording_);

    scope_persistence_factor_map_ =
        std::make_unique<ScopedPersistenceFactorMap>();
    for (const auto& it : input) {
      ValToFactorMap& to_copy = *(it.second);
      scope_persistence_factor_map_->operator[](it.first) =
          std::make_unique<ValToFactorMap>(to_copy);
    }
  }

  auto* getScopedPersistenceFactorMap() {
    return scope_persistence_factor_map_.get();
  }

  void setMappedInputOutputDims(const std::vector<int64_t>& input) {
    TORCH_INTERNAL_ASSERT(recording_);

    if (!mapped_input_output_dims_) {
      mapped_input_output_dims_ = std::make_unique<std::vector<int64_t>>(input);
    }
  }

  auto* getMappedInputOutputDims() {
    return mapped_input_output_dims_.get();
  }

 private:
  ScheduleHeuristic heuristic_;
  bool recording_ = true;

  // Actual data payload, could be folded into subclasses later.
  std::unique_ptr<std::vector<TensorView*>> vectorizable_inputs_outputs_;
  std::unique_ptr<std::vector<TensorView*>> reduction_tvs_;
  std::unique_ptr<scheduler_utils::PersistentBufferInfo>
      persistent_buffer_info_;
  std::unique_ptr<bool> has_post_reduction_bcast_;
  std::unique_ptr<bool> supported_post_reduction_fusion_;
  std::unique_ptr<ScopedPersistenceFactorMap> scope_persistence_factor_map_;
  std::unique_ptr<std::vector<int64_t>> mapped_input_output_dims_;
};

// A temporary utility class to save some boilerplate code when
//  using HeuristicSummary. Can be significantly improved in a follow up.
template <typename T>
class HeuristicCacheAccessor {
 public:
  HeuristicCacheAccessor() = default;

  T& read() {
    if (temporary_data_) {
      return *temporary_data_;
    } else {
      return *owned_data_;
    }
  }

  void writeNew(T data) {
    owned_data_ = std::make_unique<T>(std::move(data));
  }

  void takeNew(std::unique_ptr<T>& data) {
    owned_data_ = std::move(data);
  }

  void writeTemporary(T* data) {
    temporary_data_ = data;
  }

 private:
  std::unique_ptr<T> owned_data_ = nullptr;
  T* temporary_data_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
