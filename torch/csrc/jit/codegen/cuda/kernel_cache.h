#pragma once

#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <c10/util/ArrayRef.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <type_traits>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// [ Note -- 2 level cache implementation ]
//
// 2 level hierarchically nested cache is to handle the code generation and
// execution of a given PyTorch IR graph that is unique in its computational
// graph (see note computational graph down).
//
// The nested cache structures are:
//     a. GraphCache
//        - holds a vector of `InputsRequirement` & `FusionExecutorCache`, where
//          each entry is constructed to handle a set of inputs with unique
//          contiguity info, stride order & broadcasting semantics, on a given
//          device;
//        - `InputsRequirement::complyWith` demonstrates the meta information
//          that remains unchanged for a given `FusionExecutorCache`
//        - At run-time (or compile-time with Profiling Executor), we extract
//          `InputsRequirement` from given inputs to the fused operation. We
//          iterate through existing entries within GraphCache (that is the
//          `input_stacks_`) looking for a suitable entry to execute the
//          computation.
//        - In the case of cache miss, we generate a new entry and put it in
//          the GraphCache instance (We push back to both `input_stacks_` and
//          `fe_cache_`, fusion executor cache.
//     b. FusionExecutorCache
//        - holds a group of `FusionExecutor` to handle dynamic shape (varying
//          tensor sizes)
//        - currently this is a dummy implementation and has branching to handle
//          different scheduler for point-wise fusion and reduction fusion;
//
// * note computational graph
// In theory, computational graph should refer to only the computational nodes
// in a subgraph and should remain agnostic to input meta info, like
// shape, strides, type e.t.c.. However, the contract right here is fuzzy.
// Different executor applies their own protocol of what is a unique
// computational graph. e.g. Legacy Executor embeds tensor type & dimensionality
// in the graph, while Profiling Executor keeps symbolic shape as well as stride
// order in the graph as well.
// Our definition of computational graph is relaxed to support Legacy Executor,
// so the `GraphCache` could handle varying memory layout of strided tensor
// (different stride order & contiguity information). We utilize the profiling
// information now by generating an entry in GraphCache with the given profiling
// record.

class FusionExecutorCache {
 public:
  // create new fusion executor cache at a given device to handle kernel
  // generation of dynamic sizes;
  // fusion executor is taking the ownership of `fusion`;
  FusionExecutorCache(std::unique_ptr<Fusion>&& fusion, at::Device device);

  // Execute fusion graph with given inputs, create `FusionExecutor` as needed;
  std::vector<at::Tensor> runFusionWithInputs(
      const at::ArrayRef<IValue>& inputs);

 private:
  // device_ where compiled binaries are loaded on & inputs are expected to
  // reside;
  at::Device device_;

  // original un-scheduled `Fusion`;
  std::unique_ptr<Fusion> fusion_;

  // TODO: ugly logic for now. We should integrate the hashing of cache for
  //       different kernels. (alternatively we could do so in scheduler).
  // ugly bits now:
  // The fact that we have heuristics only for reduction, but use a general
  // kernel for all point-wise fusion ended up with this:
  // 1. For point-wise fusion, we have a single `FusionExecutor` in
  //    `pw_fusion_executor_cache_`
  // 2. For reduction fusion we have a hash table with ReductionParams as entry
  //    pointing to the actual `FusionExecutor` in `red_fusion_executor_cache_`
  //
  // Unfortunately, at run-time in order to search compatible `FusionExecutor`,
  // we have to call `scheduleReduction` in order to get an instance of
  // `ReductionParams` for indexing. This is not very efficient. Hence the TODO:
  // add a direct cache from inputs shapes to `FusionExecutor` entries.
  std::unique_ptr<FusionExecutor> pw_fusion_executor_cache_;
  std::unordered_map<ReductionParams, FusionExecutor, ReductionParamsHash>
      red_fusion_executor_cache_;
};

class GraphCache {
 public:
  // TODO: we should probably change shared_ptr to unique_ptr, as we want to
  //       claim the ownership of the computational graph.
  // create GraphCache on a given graph;
  // Note: if run with profiling executor, we'll try to generete a kernel with
  // profiling information at this moment.
  GraphCache(std::shared_ptr<Graph> graph);

  // execute graph with given inputs.
  std::vector<at::Tensor> runGraphWithInputs(
      const at::ArrayRef<IValue>& inputs);

 private:
  // TODO: place holder with naive implementation for now.
  // structure use to mark the compatibility of each FusionExecutorCache;
  // We also have `input_permutation_` & `output_permutation_` used to
  // facilitate dimension coalescing per stride order.
  struct InputsRequirement {
    // target device
    c10::optional<at::Device> device_;
    // TODO: TensorTypePtr is not very easy to work with.
    // c10::nullopt to take place of non-tensor type;
    std::vector<c10::optional<at::TensorTypePtr>> vec_optional_ttp;

    // common permutation order used for dimension coalescing;
    at::DimVector input_permutation_;
    at::DimVector output_permutation_;

    // construct InputsRequirement from `Graph`, this is used for constructing
    // `GraphCache` entry using profiling record
    InputsRequirement(
        const std::shared_ptr<Graph>& graph,
        const std::vector<size_t>& reduction_axes);

    // construct InputsRequirement from live input feeds, this is used to handle
    // run-time inputs to: 1. search for compatible entry; 2. insert new entry
    // in case of a cache miss.
    InputsRequirement(
        const at::ArrayRef<IValue>& inputs,
        const std::vector<size_t>& reduction_axes);

    // bool operator==(const InputsRequirement& other);
    bool complyWith(const InputsRequirement& expect);

    // helper function used at run-time to check whether a common permutation is
    // present, this is used to take the short-cut to skip permutation logic.
    bool requiresPermutation();
  };

  // construct FusionExecutorCache per InputsRequirement.
  // This function makes sure that we properly insert both `input_stacks_` and
  // `fe_cache_` at the same time.
  FusionExecutorCache* createFusionExecutorCache(
      const InputsRequirement& input_stack);

 private:
  // Computation graph;
  std::shared_ptr<Graph> graph_;
  // TODO: poor name, we should use `eliminated_axes_` instead;
  at::DimVector reduction_axes_;

  // TODO: we should really hash instead of iterative check. Optimize later...
  //       unordered_map<InputsRequirement, FusionExecutorCache>;
  std::vector<InputsRequirement> input_stacks_;
  std::vector<std::unique_ptr<FusionExecutorCache>> fe_cache_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
