#pragma once

#include <torch/csrc/jit/codegen/cuda/evaluator_common.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>

#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>

#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SegmentedGroup;
class FusionHeuristics;
class SchedulerRuntimeInfo;

// Utilities for benchmarking and profiling
struct ExecutorLog {
  std::shared_ptr<HeuristicParams> params = nullptr;
  FusionExecutor* fusion_executor = nullptr;
};

//! FusionKernelRuntime is the unified interface from fusion graphs into
//!  caching, compilation into kernels, and kernel launches.
//!
//! Each instance is also a cache entry tracked by FusionKernelRuntimeCache.
//!
//! Two types of instance can be created, one for complete/single-kernel fusion
//!  and one for segmented/multi-kernel fusion.
//! Conceptually this is a generalization of FusionExecutor that supports both
//!  single-kernel and multi-kernel caching/compiling/launching
class TORCH_CUDA_CU_API FusionKernelRuntime {
 public:
  explicit FusionKernelRuntime(
      Fusion* fusion,
      const KernelArgumentHolder& inputs);

  //! Type notations within FusionKernelRuntime Context
  using HashType = size_t;
  using SchedulerEntryPtr = std::unique_ptr<SchedulerEntry>;

  //! Evicts internally cached parameters based on input sizes.
  //!  An interface used by runtime caches.
  void evictCache(size_t input_id) {
    for (auto& fe : executors_) {
      fe.evictCache(input_id);
    }
  }

  //! query if we already have a compiled kernel for execution
  bool isCompiled() {
    std::unique_lock<std::mutex> lock0(mutex_, std::try_to_lock);
    std::unique_lock<std::mutex> lock1(compiling_, std::try_to_lock);
    if (!lock0.owns_lock() || !lock1.owns_lock()) {
      // compilation in progress
      return false;
    }

    return std::all_of(
        executors_.begin(), executors_.end(), [](const auto& executor) {
          return executor.compiled();
        });
  }

  //! starts compilation async
  void startAsyncCompile(KernelArgumentHolder& inputs);

  //! maps entries in `args` to fusion inputs.
  //! Note that this function also pushes extra bits like dimension extent into
  //! `args` for expression evaluator binding. So consider your `args` polluted
  //! after this function and use it with caution.
  void mapFusionInputsToArgs(
      std::unordered_map<Val*, const ArgAbstract*>& tensor_map,
      KernelArgumentHolder& args);

  //! Unified interface to run the managed kernels with given input
  std::vector<at::Tensor> runWithInput(KernelArgumentHolder& args);

  //! Turn On/Off profiling
  void profile(bool to_profile = true) {
    profiling_ = to_profile;
  }

  //! Internal knob for profiling shape inference
  void disableLaunchParamCache() {
    for (auto& executor : executors_) {
      executor.disableLaunchParamCache();
    }
  }

  //! Internal knob for profiling shape inference
  void disableKernelLaunch() {
    for (auto& executor : executors_) {
      executor.setExecuteKernelFlag(false);
    }
  }

  //! Returns if this runtime is segmented
  bool isSegmented() {
    return is_segmented_;
  }

  //! Returns the fusion segments if applicable
  SegmentedFusion* fusionSegments() {
    return segmented_fusion_.get();
  }

  //! Returns the list of heuristics in this runtime
  FusionHeuristics* schedulerHeuristics() {
    return heuristics_.get();
  }

  //! Return the most recently used executor, corresponding to the
  //!  most recent kernel launch.
  //! TODO: have a interface for grabbing all recent logs. Need to put a buffer
  //! space for recent logs
  ExecutorLog getMostRecentExecutorLog() {
    TORCH_INTERNAL_ASSERT(
        profiling_, "Executor log is only produced in profiling mode");
    return most_recent_executor_log_;
  }

  // Try to compute heuristics based on the SegmentedFusion managed
  //  in this kernel runtime, and will return a nullopt if either
  //  any segment cannot be scheduled or the parameters don't match
  using HeuristicsPtr = std::unique_ptr<FusionHeuristics>;
  c10::optional<HeuristicsPtr> getMaybeHeuristicsFor(
      const KernelArgumentHolder& args);

  //! Copy the launch params given in the parameter heuristics to prepare
  //!  for kernel launch for a new input dimension but same heuristics
  void updateHeuristicsLaunchParams(FusionHeuristics* update_heuristics);

 private:
  //! Interface to run a single kernel, either one kernel for single-kernel
  //! fusions, or a kernel for a segmentedGrouup in a segmented fusion. Returns
  //! the kernel outputs.
  std::vector<at::Tensor> runKernelWithInput(
      KernelArgumentHolder& args,
      SegmentedGroup* sg);

  //! Interface to compile a single kernel, either one kernel for single-kernel
  //! fusions, or a kernel for a segmentedGrouup in a segmented fusion. Returns
  //! the kernel outputs with tensor that doesn't own memory.
  KernelArgumentHolder compileKernel(
      const KernelArgumentHolder& args,
      SegmentedGroup* sg);

  //! Interface to run a the whole graph in a segmented fusion and return the
  //! complete
  //!  fusion outputs.
  std::vector<at::Tensor> runMultiKernelWithInput(
      const at::ArrayRef<IValue>& inputs,
      size_t input_id);

  //! Access the list of schedulers maintained in this runtime instance
  const std::vector<SchedulerEntryPtr>& schedulers();

  void prepareRuntimeOrder();

 private:
  //! Entries indexed by groupID:
  //! Executors holding compiled kernels
  std::vector<FusionExecutor> executors_;

  //! Heuristics object holding scheduler entries for all segments
  std::unique_ptr<FusionHeuristics> heuristics_;

  // Checks if this runtime instance is for a single-kernel fusion (false) or a
  //  segmented fusion (true).
  bool is_segmented_ = true;

  //! Multi-Kernel fusion segment when applies
  std::unique_ptr<SegmentedFusion> segmented_fusion_ = nullptr;

  //! Pre-allocated runtime workspace to speed up kernel launch preparation.
  struct RuntimeWorkSpace {
    //! Pre-determined order to run the segmented groups
    std::vector<SegmentedGroup*> group_run_order;

    //! Pre-determined order to bind tensor input meta data
    std::vector<Val*> group_extent_binding_order;
  } runtime_workspace_;

  //! Utility to speed up value evaluation at runtime
  std::unique_ptr<FusionPrecomputedValues> precomputed_values_;

  // States for profiling support
  bool profiling_ = false;

  std::mutex mutex_;
  // TODO: remove `compiling_` mutex and rely on `mutex_` only.
  // we don't need the second mutex, if only I could figure out how to pass
  // unique_lock into lambda
  std::mutex compiling_;

  // The heuristics and executor for most recent kernel launch
  ExecutorLog most_recent_executor_log_;
};

//! Encoding an input set to unique id, which is used to short-cut cache entry
//! selection in our nested cache implementation to cut off overhead.
//!
//! We have implemented naive LRU cache eviction policy here, since each entry
//! in `InputsIdLookup` is attached to a static input shape/stride, and could
//! grow gigantic when we have input shapes that does not stabalize to a finite
//! set.
//!
//! \note the uniqueness of the ide generated for a given input set is only
//!   local to the instance of `InputsIdLookup`.
//!
class TORCH_CUDA_CU_API InputsIdLookup : public NonCopyable {
 public:
  //! constructor where maximum cache size is fixed during init
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  explicit InputsIdLookup(size_t max_cache_size = 100)
      : max_cache_size_(max_cache_size){};

  //! struct to hold return value for lookupId.
  struct IdLookupReturn {
    size_t id = 0;
    size_t evict_id = 0;
    bool eviction = false;
  };

  //! encode each input sets to with an unique id;
  //! Returned data structure also indicates whether eviction has happened
  //! within the lookup cache. This is needed because lookup shortcut is also
  //! cached in nested `GraphCache`, `FusionExecutorCache` and `FusionExecutor`.
  //! see [ Note -- 2 level cache implementation ]
  IdLookupReturn lookupId(const at::ArrayRef<IValue>& inputs);

  //! debugging API that returns the size of lookup table
  size_t size() const {
    return encoding_lookup_.size();
  }

 private:
  // string to store encoded input meta information. Reuse the buffer instead of
  // stringtream gives few us perf gain.
  std::string encoding_; // Note: shared state, guarded by mutex_

  // mutex_ used to guard reused encoding_
  std::mutex mutex_;

  //! entry stored in `encoding_lookup_` to implement LRU
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct EncodingEntry {
    size_t id = 0;
    std::list<std::string>::iterator lru_iter;
  };

  //! maximum cache size for LRU
  const size_t max_cache_size_;

  //! next available unique id, we monotonically increase `current_id_` avoid
  //! conflicts
  size_t current_id_ = 1;

  //! entry in the cache, This is used to implement LRU cache, where entries in
  //! the list is ordered by their recent usage (freshly used entry is placed at
  //! the beginning)
  std::list<std::string> used_entry_;

  //! map from `std::string` to a unique id `size_t` (packaged in
  //! `EncodingEntry`
  //! ). We store an iterator to `used_entry_` to implement LRU
  std::unordered_map<std::string, EncodingEntry> encoding_lookup_;
};

//! [ Note -- 2 level cache implementation ]
//!
//! We have 2 level cache for a separation in function to keep them simpler.
//!
//! 2 level hierarchically nested cache is to handle the code generation and
//! execution of a given PyTorch IR graph that is unique in its computational
//! graph (see note on unique computational graph down).
//!
//! The nested cache structures are:
//!     a. GraphCache
//!        - GraphCache translates PyTorch IR into Fusion IR and pass it to a
//!          `FusionExecutorCache`;
//!        - GraphCache assumes all inputs to comply with profiling information,
//!          mostly tensor size & contiguity (see note on unique computational
//!          graph). The assumption is assured at runtime by
//!          `prim::CudaFusionGuard`;
//!     b. FusionExecutorCache
//!        - has a single `Fusion`, FusionExecutorCache handles kernel schedule
//!          and passed scheduled tensor to `FusionExecutor` to generate code;
//!        - create `FusionExecutor` instances to handle heuristics from dynamic
//!          shape (varying tensor sizes);
//!        - create `FusionExecutor` instances to handle different devices;
//!        - holds input cache `InputsIdLookup`, which allow cache on heuristics
//!          and launch parameters to reduce latency.
//!
//! * note on unique computational graph
//! In theory, computational graph should refer to only the computational nodes
//! in a subgraph and should remain agnostic to input meta info, like
//! shape, strides, type e.t.c.. However, the contract right here is fuzzy.
//! Different executor applies their own protocol of what is a unique
//! computational graph. e.g. Legacy Executor embeds tensor type &
//! dimensionality in the graph, while Profiling Executor keeps symbolic shape
//! as well as stride order in the graph as well.
//!
//! Our definition of a "unique" computational graph is aligned with `Fusion`
//! IR, hence the requirement extends to meta information on input tensors.
//! Which means, for each input tensor, following properties are fixed:
//!     a) stride order;
//!     b) contiguity information;
//!     c) broadcasting semantics (size-1 or not);
//!     d) rank;
//!     e) scalar type;
//!
//!
//! [ Note -- Segmented Fusion Tentative Design ]
//! Segmentation adds an extra dimension in caching. Initial implementation,
//! assumed graph partition strategy is independent of input pattern, which we
//! can revisit once we have more advanced graph segmentation logic Each
//! FusionExecutorCache corresponds to one graph and one graph segmentation.
//!
//!
class TORCH_CUDA_CU_API FusionExecutorCache {
 public:
  //! create new fusion executor cache at a given device to handle kernel
  //! generation of dynamic sizes
  //! fusion executor is taking the ownership of `fusion`
  explicit FusionExecutorCache(std::unique_ptr<Fusion> fusion);

  //! Execute fusion graph with given inputs, create `FusionExecutor` as needed
  //! Note this function also handles permutation & input update outside of
  //! codegen.
  std::vector<at::Tensor> runFusionWithInputs(
      const at::ArrayRef<IValue>& inputs);

  Fusion* fusion() {
    return fusion_.get();
  }

  void printFusion() {
    fusion_->printMath();
  }

  FusionKernelRuntime* getMostRecentKernelRuntime() {
    return most_recent_runtime_;
  }

  // TODO: in a follow up we need a global logging structure
  //  to capture runtime profiling info. We also need to define
  //  a suitable profiling window / buffer size.
  ExecutorLog getMostRecentExecutorInfo() {
    TORCH_INTERNAL_ASSERT(most_recent_runtime_ != nullptr);
    return most_recent_runtime_->getMostRecentExecutorLog();
  }

  void profile(bool to_profile) {
    profiling_ = to_profile;
    for (auto& it : kernel_runtimes_) {
      for (auto& kernel_runtime : it.second) {
        kernel_runtime->profile(to_profile);
      }
    }
  }

  //! Internal knob for profiling shape inference
  void disableLaunchParamCache() {
    for (auto& it : kernel_runtimes_) {
      for (auto& kernel_runtime : it.second) {
        kernel_runtime->disableLaunchParamCache();
      }
    }
  }

  //! Internal knob for profiling shape inference
  void disableKernelLaunch() {
    for (auto& it : kernel_runtimes_) {
      for (auto& kernel_runtime : it.second) {
        kernel_runtime->disableKernelLaunch();
      }
    }
  }

  //! converts inputs from IValue to KernelArgumentHolder, also handles cache
  //! lookup
  KernelArgumentHolder prepareInputs(const at::ArrayRef<IValue>& inputs);

  //! query if there's a kernel ready to go for given inputs
  bool isCompiled(const at::ArrayRef<IValue>& inputs);

  //! compile a kernel executor for given inputs. Note: the compilation is
  //! async, there's some restriction on the user side. e.g. don't overlap
  //! compilation and execution for the same FusionExecutor entry. This is
  //! experimental at this moment, please use with extra caution.
  void compileFusionAsync(const at::ArrayRef<IValue>& inputs);

 private:
  //! evict cached short cut entry in `code_to_fe_lookup_` as well as cached
  //! entry in `FusionExecutor`
  void evictCache(size_t cache_id);

  FusionKernelRuntime* getKernelRuntimeFor(const KernelArgumentHolder& inputs);

 private:
  //! original un-scheduled `Fusion`;
  std::unique_ptr<Fusion> fusion_;

  //! inputs to unique_id lookup table;
  InputsIdLookup inputs_id_lookup_;

  //! Graphs after input dependent transfoms
  std::unordered_map<size_t, std::vector<std::unique_ptr<FusionKernelRuntime>>>
      kernel_runtimes_;

  //! Logging state for most recent compilation
  bool profiling_ = false;

  //! Logging state for most recent compilation
  ExecutorLog most_recent_executor_log_;

  //! short-cut for cache hit
  std::unordered_map<size_t, FusionKernelRuntime*> id_to_kernel_runtime_;

  //! Profiling info:
  //! TODO: this can be largely expanded to look at complete
  //!   caching profiles. Currently it just makes it easier to test
  FusionKernelRuntime* most_recent_runtime_ = nullptr;

  //! indices of fusion outputs that are aliased to inputs. These are used only
  //! to support in-place update and should have been dropped before pushing
  //! outputs to stack.
  std::set<int> aliased_output_indices_;
};

class GraphCache {
 public:
  //! TODO: we should probably change shared_ptr to unique_ptr, as we want to
  //!       claim the ownership of the computational graph.
  //! create GraphCache on a given graph;
  //! We extract global stride index order and translate PyTorch JIT IR to
  //! Fusion IR.
  explicit GraphCache(const std::shared_ptr<Graph>& graph);

  //! execute graph with given inputs
  std::vector<at::Tensor> runGraphWithInputs(
      const at::ArrayRef<IValue>& inputs);

 private:
  //! construct FusionExecutorCache
  void createFusion(const std::shared_ptr<Graph>& graph);

 private:
  //! FusionExecutorCache that performs schedule and kernel execution;
  std::unique_ptr<FusionExecutorCache> fusion_executor_cache_;

  //! num of outputs
  size_t num_of_outputs_ = 0;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
