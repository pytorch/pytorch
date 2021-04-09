#pragma once

#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>

#include <c10/util/ArrayRef.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SegmentedGroup;
class FusionHeuristics;

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
  //! Type notations within FusionKernelRuntime Context
  using HashType = size_t;
  using SchedulerEntryPtr = std::unique_ptr<SchedulerEntry>;

  //! Create a runtime instance for segmented fusion
  explicit FusionKernelRuntime(
      SegmentedFusion* segmented_fusion,
      std::unique_ptr<FusionHeuristics>& heuristics,
      size_t input_id);

  //! Create a runtime instance for complete/single-kernel fusion
  explicit FusionKernelRuntime(
      Fusion* fusion,
      std::unique_ptr<FusionHeuristics>& heuristics,
      size_t input_id);

  //! Evicts internally cached parameters based on input sizes.
  //!  An interface used by runtime caches.
  void evictCache(size_t input_id) {
    for (auto& fe : executors_) {
      fe.evictCache(input_id);
    }
  }

  //! Unified interface to run the managed kernels with given input
  std::vector<at::Tensor> runWithInput(
      const at::ArrayRef<IValue>& inputs,
      size_t input_id) {
    if (is_segmented_) {
      return runMultiKernelWithInput(inputs, input_id);
    } else {
      return runKernelWithInput(inputs, input_id);
    }
  }

  //! Cache Interface: Common utility for computing hash of scheduler entires
  static HashType getHash(FusionHeuristics* sh);

  //! Cache Interface: trivially copied and easily compared
  //!   descriptor for a FusionKernelRuntime instance
  class HeuristicTag {
   public:
    //! Computes hash upon creation
    explicit HeuristicTag(FusionHeuristics*);

    //! Tag equal abstracts the heuristics equivalence
    bool operator==(const HeuristicTag& other) const;

    //! Returns computed hash value
    HashType hash() const {
      return hash_;
    }

   private:
    HashType hash_;
    FusionHeuristics* heuristics_;
  };

  class HeuristicTagHash {
   public:
    HashType operator()(const HeuristicTag& et) const {
      return et.hash();
    }
  };

 private:
  //! Interface to run a single kernel, either one kernel for single-kernel
  //! fusions,
  //!  or a kernel for a segmentedGrouup in a segmented fusion. Returns the
  //!  kernel outputs.
  std::vector<at::Tensor> runKernelWithInput(
      const at::ArrayRef<IValue>& inputs,
      size_t input_id,
      SegmentedGroup* sg = nullptr);

  //! Interface to run a the whole graph in a segmented fusion and return the
  //! complete
  //!  fusion outputs.
  std::vector<at::Tensor> runMultiKernelWithInput(
      const at::ArrayRef<IValue>& inputs,
      size_t input_id);

  //! Access the list of schedulers maintained in this runtime instance
  const std::vector<SchedulerEntryPtr>& schedulers();

 private:
  friend class HeuristicTag;
  //! Entries indexed by groupID:
  //! Executors holding compiled kernels
  std::vector<FusionExecutor> executors_;

  //! Heuristics object holding scheduler entries for all segments
  std::unique_ptr<FusionHeuristics> heuristics_;

  // Checks if this runtime instance is for a single-kernel fusion (false) or a
  //  segmented fusion (true).
  bool is_segmented_ = true;

  // Maintain the original segmented fusion that this runtime is maintaining
  //  heuristics for. Applies only in the segmented fusion case, i.e.
  //  is_segmented==true
  SegmentedFusion* segmented_fusion_ = nullptr;

  // Maintain the original fusion that this runtime is maintaining
  //  heuristics for. Applies only in the single-kernel fusion case, i.e.
  //  is_segmented==false
  Fusion* complete_fusion_ = nullptr;
};

//! Object holding cache entries for segmented fusion
class TORCH_CUDA_CU_API FusionKernelRuntimeCache {
 public:
  explicit FusionKernelRuntimeCache() = default;

  //! Evict the cacheEntry by id.
  //!  removes ID to RT lookup and corresponding
  //!  input entries. Doesn't actually release any compiled
  //!  kernel because compiling is expensive
  void evictId(size_t input_id);

  //! Interface for registering segmented fusion for caching heuristics
  void initSegmentCache(SegmentedFusion* sf);

  //! Interface for registering complete fusion for caching single kernel
  //! heuristics
  void initSingleKernelCache(
      Fusion* fusion,
      ScheduleHeuristic schedule_heuristic);

  //! API for collecting FusionKernelRuntime entry from cache,
  //!  contains a two level lookup,
  //!  if input_id is hit -> returns cached
  //!  if input_id miss -> lookup with heuristics -> return cached if found
  //!  if heuristics miss -> create a new entry and return created
  FusionKernelRuntime* getRt(
      const at::ArrayRef<IValue>& inputs,
      size_t input_id);

 private:
  using HeuristicTag = FusionKernelRuntime::HeuristicTag;
  using HeuristicTagHash = FusionKernelRuntime::HeuristicTagHash;
  //! FusionKernelRuntime cache based on HeuristicTag lookup
  using SegRuntimePtr = std::unique_ptr<FusionKernelRuntime>;
  using SegRuntimeCache =
      std::unordered_map<HeuristicTag, SegRuntimePtr, HeuristicTagHash>;
  //! One cache per device id
  using SegRuntimeCacheGroup =
      std::unordered_map<int, std::unique_ptr<SegRuntimeCache>>;

  //! internal maintenance functions
  //!  Currently don't have releasing entry at this level since
  //!  we would not release compiled kernels at this point
  void insertEntry(int dev_id, HeuristicTag tag, SegRuntimePtr&& rt);
  FusionKernelRuntime* at(int dev_id, HeuristicTag tag);

 private:
  //! Checks if this cache is for segmented fusion or not
  bool is_segmented_ = false;

  //! Store the heuristic corresponding to the complete fusion if any
  ScheduleHeuristic complete_fusion_heuristic_ = ScheduleHeuristic::PointWise;

  //! Contains the complete fusion
  Fusion* complete_fusion_ = nullptr;

  //! Data structure hosting the actual caches
  SegRuntimeCacheGroup seg_runtime_cache_group_;

  //! Input_id to runtime shortcut
  std::unordered_map<size_t, FusionKernelRuntime*> id_to_rt_;

  //! Reference to the segmented fusion held in FusionExecutorCache
  SegmentedFusion* segmented_fusion_ = nullptr;

  //! In case of cache hit by input id, return pointer to that entry,
  //!  returns nullptr if input_id miss
  FusionKernelRuntime* getRtById(size_t input_id);

  //! In case of input id miss, evaluate heuristics and find a hit by heuristics
  //!   in case of heuristics miss, create a new entry
  FusionKernelRuntime* getRtByHeuristics(
      const at::ArrayRef<IValue>& inputs,
      size_t input_id);
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
//!        - GraphCache handles permutation for I/O tensors, when they share
//!          global stride order. This permutation facilitates dimension
//!          collapsing, which gives simpler indexing.
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
  //! generation of dynamic sizes;
  //! fusion executor is taking the ownership of `fusion`;
  explicit FusionExecutorCache(std::unique_ptr<Fusion>&& fusion);

  //! Execute fusion graph with given inputs, create `FusionExecutor` as needed;
  std::vector<at::Tensor> runFusionWithInputs(
      const at::ArrayRef<IValue>& inputs);

  Fusion* fusion() {
    return fusion_.get();
  }

  void printFusion() {
    fusion_->printMath();
  }

  SegmentedFusion* fusionSegments() {
    TORCH_INTERNAL_ASSERT(isSegmented());
    return fusion_segments_.get();
  }

  bool isSegmented() {
    return fusion_segments_ != nullptr;
  }

 private:
  //! evict cached short cut entry in `code_to_fe_lookup_` as well as cached
  //! entry in `FusionExecutor`
  void evictCache(size_t cache_id) {
    fusion_kernel_runtime_cache_.evictId(cache_id);
  };

 private:
  //! original un-scheduled `Fusion`;
  std::unique_ptr<Fusion> fusion_;

  //! inputs to unique_id lookup table;
  InputsIdLookup inputs_id_lookup_;

  //! Multi-Kernel fusion segment when applies
  std::unique_ptr<SegmentedFusion> fusion_segments_ = nullptr;

  //! Caching for segmented fusions
  FusionKernelRuntimeCache fusion_kernel_runtime_cache_;
};

class GraphCache {
 public:
  //! TODO: we should probably change shared_ptr to unique_ptr, as we want to
  //!       claim the ownership of the computational graph.
  //! create GraphCache on a given graph;
  //! We extract global stride index order and translate PyTorch JIT IR to
  //! Fusion IR.
  explicit GraphCache(const std::shared_ptr<Graph>& graph);

  //! execute graph with given inputs, permutation on I/O tensors are performed.
  std::vector<at::Tensor> runGraphWithInputs(
      const at::ArrayRef<IValue>& inputs);

 private:
  //! Computation graph;
  std::shared_ptr<Graph> graph_;
  //! TODO: poor name, we should use `eliminated_axes_` instead;
  at::DimVector reduction_axes_;
  bool support_permutation_;

  //! helper function used at run-time to check whether a common permutation is
  //! present, this is used to take the short-cut to skip permutation logic.
  bool requiresPermutation();

  //! construct FusionExecutorCache
  void createFusion(const std::shared_ptr<Graph>& graph);

  //! extract permutation for I/O tensor from accumulcated tensor type pointer
  //! on all inputs;
  void extractPermutation(const TensorTypePtr& acc_type);

 private:
  // common permutation order used to facilitate dimension coalescing;
  at::DimVector input_permutation_;
  at::DimVector pw_output_permutation_;
  at::DimVector reduction_output_permutation_;

  //! FusionExecutorCache that performs schedule and kernel execution;
  std::unique_ptr<FusionExecutorCache> fusion_executor_cache_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
