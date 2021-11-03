#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SchedulerRuntimeInfo;

namespace scheduler_utils {

constexpr int64_t register_file_size = 256 * 1024;
constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
constexpr int64_t y_grid_limit = 65535;

// Largest Power of 2 less-than n
constexpr int64_t lastPow2(int64_t n) {
  TORCH_INTERNAL_ASSERT(n >= 0);
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 16); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 32); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  return std::max((int64_t)1, n - (n >> 1));
}

// Merge all reduction to the right side and returns total number of
// reduction axes. Don't merge is typically used for trivial reductions.
size_t mergeReduction(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge = {});

// merge all non-reduction axes to the left side and returns total number of
// iteration axes. Don't merge is typically used for trivial reductions.
size_t mergeNonReduction(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& dont_merge = {});

TORCH_CUDA_CU_API void parallelizeAllLike(
    TensorView* reference_tv,
    const std::vector<TensorView*>& all_tvs);

TORCH_CUDA_CU_API void computeAtInputs(
    TensorView* consumer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

TORCH_CUDA_CU_API void computeWithOutputs(
    TensorView* producer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

struct PersistentBufferInfo {
  std::vector<TensorView*> buffers;
  std::unordered_set<IterDomain*> unmappable_dims;
};

// Buffers whos roots can't map to all producer roots based on compute at. These
// are the buffers we would make persistent in a persistent kerenl or would have
// to recompute if we can't make a persistent kernel. This function will also
// return inputs as being marked persistent if they follow this pattern. It is
// important to note however inputs don't strictly have to be persistent as they
// can simply be read multiple times from GMEM in the same kernel.
PersistentBufferInfo persistentBuffers(Fusion* fusion);

struct TvProperties {
  // How many elements in tensor view are there to reduce
  int64_t reduction_numel = 1;
  // How many reductions do we need to perform, i.e. how many iter dimension
  // elements are there
  int64_t iteration_numel = 1;
  // Do we reduce the fastest dimension, if no reduction mark true
  bool fastest_dim_reduction = true;
  // What's the iter numel to the left of the reduction (if there is one)
  int64_t iter_outside_red = 1;
  // What's the iter numel to the right of the reduction (if this is or isn't
  // one)
  int64_t iter_inside_red = 1;
};

// Fill TvProperties structure about tv
TvProperties getProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* tv);

// Will call computeAt once on each producer, with the first consumer found that
// is a consumer of the individual producer
void computeAtBetween(
    const std::vector<TensorView*>& producers,
    const std::vector<TensorView*>& consumers,
    int pos,
    ComputeAtMode mode,
    std::unordered_set<IterDomain*> mapped_to_trivial_reduction = {});

// Compute the amount of register space would be needed to perform this kernel
// persistently, only based on buffers that must be persistent, and based on the
// maximum of all minimum size requirement. i.e. if must be persistent, only
// hold persistent dimension.
int64_t persistentBufferSize(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    PersistentBufferInfo& persistent_buffers,
    HeuristicSummary* data_cache = nullptr);

// Returns a set of all iteration domains (in roots of tensors) that map to a
// trivial reduction
std::unordered_set<IterDomain*> getTrivialReductionMap(Fusion* fusion);

// Merges tensor view to the form:
// [IterationDomain, ReductionDomain, TrivialReductionDim0,
// TrivialReductionDim1, ...] Returns if <iteration dimensions, reduction
// dimensions>
std::pair<bool, bool> canonicalDimReduction(Fusion* fusion, TensorView* tv);

// Return a list of tensor views that are outputs of reduction operations. If
// multiple outputs of an expression are found, only include one in the list
// (WelfordOp)
std::vector<TensorView*> getReductionTvs(Fusion* fusion);

// Consistent parallelization based on provided reduction parameters. Provided
// tensor is expected to be reduced by canonicalDimReduction before sending
// here. reduction_tv should be provided as the tensorview to reduce.
// RFactor of reduction_tv will be returned if applicable otherwise reduction_tv
// is returned
TensorView* scheduleReductionTV(
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    bool has_iter_axis);

// Reset inputs and outputs to global memory, everything else to local.
void clearMemorySpace(Fusion* fusion);

// Returns cached after tensors of the fusion inputs if unrolled. Otherwise
// return empty vector.
std::vector<TensorView*> cacheInputs(Fusion* fusion, bool unroll);

// Returns the pairs of <cache of each fusion output, corresponding output> for
// all outputs.
std::vector<std::pair<TensorView*, TensorView*>> cacheAndForkOutputs(
    Fusion* fusion,
    bool unroll);

// Inlining function intended for single or multi reduction fusions.
void multiReductionInliner(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* reduction_tv,
    TensorView* reference_tv,
    std::vector<TensorView*> reduction_tvs,
    std::vector<TensorView*> cached_inputs,
    std::vector<std::pair<TensorView*, TensorView*>> cached_outputs);

// Uses a lot of logic from TransformPropagator in the implementation
class FindAllMappedDims {
 private:
  FindAllMappedDims(TensorView* from, IterDomain* starting_id);

 private:
  std::unordered_map<TensorView*, IterDomain*> mapped_ids;
  TensorView* starting_tv = nullptr;
  IterDomain* starting_id = nullptr;

 public:
  // Looks through fusion and finds all dims that match to the one provided in
  // the tensorview provided. Iter domain must be a root domain.
  static std::unordered_set<IterDomain*> from(TensorView* tv, IterDomain* id);
};

// Checks if tensor view has an iteration domain in vector dims in its inner
// most root position (excluding broadcast and reduction), and checks if it is a
// contiguous dimension
bool shouldVectorize(
    TensorView* tv,
    std::unordered_set<IterDomain*> vector_dims);

// Returns all inputs and outputs that share the inner most dimension of the
// provided reference. If reference is an input it ignores reduction axes, will
// ignore all broadcast axes.
std::vector<TensorView*> getVectorizableInputsOutputs(TensorView* reference_tv);

// Returns a vector of counts, size = reference_tv->getRootDomain().size(), each
// entry [i] is the number of inputs/outputs that have a non-broadcast dimension
// mapped to the corresponding dimension in reference_tv. Count includes
// reference_tv if reference_tv is an input or output. Count is multiplied by
// data type size.
std::vector<int64_t> mappedInputsOutputs(TensorView* reference_tv);

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
