#pragma once

#include <ATen/core/ivalue.h>

#include <fusion.h>
#include <scheduler/transpose_heuristic.h>

#define SUPPORT_SPLITTING_INNERMOST_DIM 0

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Note [Transpose scheduling]
//
// The target of transpose scheduling is to get coalesced global memory access
// to as much input and output tensors as possible. For a DAG with only pure
// pointwise operators, the scheduling is very simple because the inner most
// dimension of all input and output tensors are all mapped together in the
// ComputeAtMap, i.e., there is essentially only one inner most dimension. In
// such case, we just vectorize that inner most dimension and bind it to
// threadIdx.x identically for all input and output tensors. In the case where
// transposes are present in the DAG, the inner most dimensions of different
// inputs and outputs might not match. And there is no fixed pattern on which
// input/output tensors should share the same inner most dimension with which.
// Consider the following example DAGs ([T] represents transpose, all tensors
// are 2D):
//
//   t0    t1      t0    t1      t0    t1        t0    t1         t0
//    \    |        \    /        \    |          \    |          |
//     \  [T]       [T] [T]        \  [T]          t2 [T]        [T]
//      \ /           \ /           \ / \         / \ / \         |
//      t2             t2           t2   t3      t3  t4 t5       [T]
//                                                                |
//                                                                t1
//
// In order to support all these cases in a general way, the following
// perspective is very important: What we are looking for is to bind threadIdx.x
// differently for different inputs and outputs, so there has to be some tensor
// somewhere in the DAG that we write and read with different threadIdx.x
// bindings. The tensor of binding swap can be any tensor on the path that
// connects inputs/outputs with different inner most dimension, especially, it
// does not necessarily have to be the tensor of the transpose operator. In
// other words, thanks to our indexing system who is already taking care of the
// correctness of transpose, the scheduler can freely choose where to realize
// these transposes as different threadIdx.x bindings. This observation greatly
// simplifies our scheduling.
//
// Our scheduling strategy is as follows: We first split the inputs and outputs
// of the fusion into two groups according to their inner most dimension. The
// inner most dimensions of tensors in the same group are mapped to each other,
// and they are not mapped to the inner most dimesion of tensors in a different
// group. Depending on the transpose pattern, there can be more than two groups,
// if this is the case, we only consider the two largest groups, and the tensors
// in the remaining groups will just be accessed unvectorized and uncoalesced.
// We call the largest group as `group1` and the second largest group as
// `group2`. When we have the groups, we will make a 2D tiling [I1, I2] ->
// [I1/tile1, tile1, I2/tile2, tile2] on the inner most dimensions of group1 and
// group2. If I1 and I2 are too small to make a 32x32 tile, such as in the
// fusion of tanspose(T1[1024, 2, 1024, 2], {1, 3}), we merge in other
// dimensions to make a virtual I1 and I2. The details of how we create virtual
// I1 and I2 are described in note [Supporting small transpose dimensions].
//
// Each tile [tile1, tile2] will be handled by a block, and the tensors that
// have mismatched threadIdx.x bindings will use shared memory. The outer IDs of
// the tiling split will be merged with non-tiled IDs and then binded to
// blockIdx.x for the entire DAG, regardless of which group a tensor belongs to.
// For the inner tile IDs [tile1, tile2], we need to transform and parallelize
// group 1 and group 2 differently. The intermediate tensors can be transformed
// and parallelized consistently either with group 1 or group 2. Here, since
// group 1 is larger than group 2, we decide to only transform and parallelize
// the cached inputs of group 2 together with group 2, and keep the rest of the
// DAG consistent with group 1.
//
// If you would like to see an example of how to manually schedule a complicated
// DAG using this idea, refer to:
//   FusionManualScheduleTransposeComplexDAG1_CUDA

class SchedulerRuntimeInfo;
class HeuristicSummary;

TORCH_CUDA_CU_API std::shared_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

TORCH_CUDA_CU_API std::shared_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

TORCH_CUDA_CU_API void scheduleTranspose(
    Fusion* fusion,
    TransposeParams params);

TORCH_CUDA_CU_API LaunchParams scheduleTranspose(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs);

//! Utility for canSchedule interface to check if this fusion has at least two
//! groups, each with a fully broadcasted reference tensor.
TORCH_CUDA_CU_API bool hasAtLeastTwoValidGroups(Fusion* fusion);

// If can schedule at runtime, returns empty string, otherwise returns the
// reason why we should not schedule at runtime.
TORCH_CUDA_CU_API std::string getTransposeRuntimeRejectReason(
    Fusion* fusion,
    HeuristicSummary* data_cache,
    SchedulerRuntimeInfo& runtime_info);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
