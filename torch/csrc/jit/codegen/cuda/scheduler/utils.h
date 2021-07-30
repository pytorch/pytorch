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

void computeAtInputs(
    TensorView* consumer,
    int pos,
    ComputeAtMode mode = ComputeAtMode::Standard);

void computeWithOutputs(
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
    ComputeAtMode mode);

// Compute the amount of register space would be needed to perform this kernel
// persistently, only based on buffers that must be persistent, and based on the
// maximum of all minimum size requirement. i.e. if must be persistent, only
// hold persistent dimension.
int64_t persistentBufferSize(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info);

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
