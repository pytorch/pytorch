#pragma once

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/maxinfo_propagator.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SchedulerRuntimeInfo;
class HeuristicSummary;

namespace scheduler_utils {

// Assume any only half of the register file is available to spend on buffers,
// this is because when we allocate a buffer in register is has to be accesed
// with a compile time coonstant index. Unfortunately nvcc seems to be using
// many registers for indexing. This is a bad estimation of extra register use,
// but it's hard to get a better one.
constexpr int64_t register_file_size = 256 * 1024 / 2;
constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
constexpr int64_t y_grid_limit = 65535;
constexpr int64_t z_grid_limit = 65535;
constexpr int64_t z_block_limit = 64;

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

// Div x by y, but min at 1
inline int64_t safeDiv(const int64_t x, const int64_t y) {
  return std::max(x / y, (int64_t)1);
}

// Split the given dimensions in `to_split`. Also update the dimensions in
// `to_update` to the positions in the splitted tensor. Splitting one dimension
// multiple times is supported, and if this is the case, then the order of
// `to_split` matters. All given dimensions are numbers before any split.
TORCH_CUDA_CU_API void splitDims(
    TensorView* tv,
    std::vector<std::pair<size_t, size_t>> to_split, // (dim, size)
    std::vector<size_t>& to_update);

TORCH_CUDA_CU_API inline void splitDims(
    TensorView* tv,
    std::vector<std::pair<size_t, size_t>> to_split) { // (dim, size)
  std::vector<size_t> unused;
  splitDims(tv, std::move(to_split), unused);
}

// Merge all the given dimensions in `to_merge` into a single dimension. Also
// update the dimensions in `to_update` to the positions in the merged tensor.
// Returns the merged dimension. All given dimensions are numbers before any
// merge.
TORCH_CUDA_CU_API c10::optional<size_t> mergeDims(
    TensorView* tv,
    std::vector<size_t> to_merge,
    std::vector<size_t>& to_update);

TORCH_CUDA_CU_API inline c10::optional<size_t> mergeDims(
    TensorView* tv,
    std::vector<size_t> to_merge) {
  std::vector<size_t> unused;
  return mergeDims(tv, std::move(to_merge), unused);
}

// Merge all reduction to the right side and returns total number of
// reduction axes.
size_t mergeReduction(TensorView* tv);

// merge all non-reduction axes to the left side and returns total number of
// iteration axes.
size_t mergeNonReduction(TensorView* tv);

// Propagate the parallelization from the selected dimensions of the reference
// tensor to their corresponding dimensions in all selected tensors in the DAG.
// Position `pos` means selecting all the dimensions [0, 1, ..., pos - 1]. pos =
// -1 means selecting all dimensions. `selected_tvs` are selected tensors in the
// DAG. Empty `selected_tvs` means selecting all tensors in the fusion of
// `reference_tv`. `selected_parallel_types` are the selected parallel types.
// Empty `selected_parallel_types` means selecting all parallel types.
TORCH_CUDA_CU_API void parallelizeAllLike(
    TensorView* reference_tv,
    int64_t pos = -1,
    std::vector<TensorView*> selected_tvs = {},
    const std::unordered_set<ParallelType>& selected_parallel_types = {},
    bool propagate_padding = true);

TORCH_CUDA_CU_API inline void parallelizeAllLike(
    TensorView* reference_tv,
    std::vector<TensorView*> selected_tvs,
    const std::unordered_set<ParallelType>& selected_parallel_types = {},
    bool propagate_padding = true) {
  parallelizeAllLike(
      reference_tv,
      -1,
      std::move(selected_tvs),
      selected_parallel_types,
      propagate_padding);
}

struct PersistentBufferInfo {
  std::vector<TensorView*> persistent_buffers;
  std::unordered_set<IterDomain*> unmappable_dims;

  // Persistent buffers are needed until the path through the reduction -
  // broadcast chain is resolved by any other chain using the persistent buffer
  // that is not going through a reduction. This assumes all reduction paths
  // have the same reduction pattern. Order is the same as persistent_buffers
  std::vector<std::vector<TensorView*>> persistent_buffer_resolution_points;

  // Not all persistent buffers can be projected to inputs, if a buffer can be
  // projected to the inputs which may reduce the persistent buffer size (BN
  // Backwards specifically) then keep track of it here. Persistent buffers that
  // have a persistent buffer/reduction before them should not be projected
  // through that.
  std::vector<TensorView*> projectable_persistent_buffers;

  // Track inputs of input projectable buffers
  std::vector<TensorView*> projectable_buffer_inputs;

  // Map unmappable dims to projectable_buffer_inputs
  std::unordered_set<IterDomain*> unamppable_dims_projected_to_inputs;
};

// Buffers whos roots can't map to all producer roots based on compute at. These
// are the buffers we would make persistent in a persistent kerenl or would have
// to recompute if we can't make a persistent kernel. This function will also
// return inputs as being marked persistent if they follow this pattern. It is
// important to note however inputs don't strictly have to be persistent as they
// can simply be read multiple times from GMEM in the same kernel.
TORCH_CUDA_CU_API PersistentBufferInfo persistentBuffers(Fusion* fusion);

struct TvProperties {
  // How many elements in tensor view are there to reduce.
  int64_t total_reduction_numel = 1;

  // How many reductions do we need to perform, i.e. how many iter dimension.
  // elements are there
  int64_t total_iteration_numel = 1;

  // Is the inner most dimension a reduction, if no reductions mark true.
  bool fastest_dim_reduction = true;

  // How many elements in the inner most dimension merging surrounding domains
  // that match in type. This is used for 3D schedulers in
  // reduction/normalization.
  int64_t inner_most_dimension_numel = 1;

  // Same thing as above, but the number of dimensions instead of the numel.
  int64_t inner_most_dimension_ndims = 1;

  // Merging neighboring iteration domains, and reduction domains, what's the
  // resulting dimensionality of the problem.
  int64_t dimensionality = 1;
};

// Fill TvProperties structure about tv
TvProperties getProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* tv);

// Struct to store persistent buffer sizes. also holds the persistent buffer
// size of the buffers are projected to the inputs.
struct PersistentBufferSizeReturn {
  int64_t persistent_buffer_size = 0;
  int64_t projected_persistent_buffer_size = 0;
};

// Compute the amount of register space would be needed to perform this kernel
// persistently, only based on buffers that must be persistent, and based on the
// maximum of all minimum size requirement. i.e. if must be persistent, only
// hold persistent dimension.
TORCH_CUDA_CU_API PersistentBufferSizeReturn persistentBufferSize(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    PersistentBufferInfo& persistent_buffers,
    HeuristicSummary* data_cache = nullptr);

// Merges tensor view to the form:
// [IterationDomain, ReductionDomain] Returns if <iteration dimensions,
// reduction dimensions>
std::pair<bool, bool> canonicalDimReduction(
    Fusion* fusion,
    TensorView* tv,
    bool schedule_3D = false);

// Return a list of tensor views that are outputs of reduction operations. If
// multiple outputs of an expression are found, only include one in the list
TORCH_CUDA_CU_API std::vector<TensorView*> getReductionTvs(Fusion* fusion);

// Returns a list of TensorViews that are the consumer tv for a view operation.
std::vector<TensorView*> getViewTVs(Fusion* fusion);

// Reset inputs and outputs to global memory, everything else to local.
void clearMemorySpace(Fusion* fusion);

// Returns cached after tensors of the fusion inputs if unrolled. Otherwise
// return empty vector.
TORCH_CUDA_CU_API std::vector<TensorView*> cacheInputs(
    Fusion* fusion,
    bool unroll);

// Returns the pairs of <cache of each fusion output, corresponding output> for
// all outputs.
TORCH_CUDA_CU_API std::vector<std::pair<TensorView*, TensorView*>>
cacheAndForkOutputs(Fusion* fusion, bool unroll);

// Ignores broadcast and reduction, returns iter domain in root domain that's
// "inner most". If this is an rfactored reduction domain, actually check the
// root domain, this is because the rfactored reduction tensorview has the
// vectorized dimension, but that means the rfactor domain could have reordered
// what we consider the "inner most" allocated position on it if we consider the
// rfactor dimension.
//
// If reduction tv and has rfactor return root domain, otherwise return rfactor
// domain.
IterDomain* innerMostRootDim(TensorView* tv);

// Looks through fusion and finds all dims that match to the one provided in
// the tensorview provided. Iter domain must be a root domain. If inner_only,
// will only map dimensions if they're the inner most position. This is
// important when projecting a dimension between an rfactor position and its
// root position when mapping from consumer to producer. If inner_only=true,
// takes the rfactor/root dimensions that maps, projects it to the root/rfactor
// domain, but only following the inner most pass when encounting split/merge.
// When propagating backward, for split it will only propagate backwards if the
// mapped dimension is the inner portion of the split. For merge, inner_only
// doesn't make a dimension and will propagate through the inner portion of the
// merge. When propagating forward, the logic is symmetric with the backward
// case.
class FindAllMappedDims : public MaxInfoSpanningTree::Propagator {
  std::unordered_map<TensorView*, IterDomain*> mapped_root_ids_;
  std::unordered_map<TensorView*, IterDomain*> mapped_rfactor_ids_;
  TensorView* starting_tv_ = nullptr;
  IterDomain* starting_id_ = nullptr;
  bool inner_only_;

 public:
  FindAllMappedDims(TensorView* from, IterDomain* starting_id, bool inner_only);
  virtual void setUp() override;
  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;
  std::unordered_set<IterDomain*> get() const;
};

// Checks if tensor view has an iteration domain in vector dims in its inner
// most root position (excluding broadcast and reduction), and checks if it is a
// contiguous dimension
bool hasInnerDim(
    TensorView* tv,
    std::unordered_set<IterDomain*> vector_dims,
    bool should_vectorize);

// Returns all inputs and outputs that share the inner most dimension of the
// provided reference. If reference is an input it ignores reduction axes, will
// ignore all broadcast axes. If inner_only, will require inner->inner mapping
// in view, otherwise, it allows all inner->any mapping. If vectorize_pass, will
// check contiguity for vectorization, otherwise it just checks it has that
// inner dim.
std::vector<TensorView*> getInputsOutputsWithInnerDim(
    TensorView* reference_tv,
    bool inner_only,
    bool vectorize_pass);

// Holder return struct for the below function.
struct DisjointViewSetInfo {
  // const* to the disjoint set in disjoint_view_set passed in to
  // getDisjointViewSetsOf each iterdomain in the rfactor of ref is mapped to.
  //
  // WARNING: these pointers are relative to the disjoint_view_set reference
  // passed into getDisjointViewSetsOf it's the user's responsibility to
  // maintain the lifetime of that reference to match this vector.
  std::vector<const VectorOfUniqueEntries<IterDomain*>*> disjoint_sets_of_ref;

  // Unique ID associated to the disjoint view group the rfactor id belongs to
  // in disjoint_sets_of_ref. It's straight forward to map from
  // disjoint_sets_of_ref to the vector, but not the other way around.
  std::vector<int> disjoint_set_ids;

  // TensorView reference the above vectors are relative to.
  TensorView* ref;
};

// Returns disjoint view sets mapped onto the given reference. Returns a pair
// of vectors of size rfactorDomain of reference. Vector of
// VectorOfUniqueEntries returns a const* to the disjoint set in
// disjoint_view_set the iterdomain is mapped to. Integer vector represents
// which disjoint view group the rfactor id belongs to. It's straight forward
// to map from the former to the latter, but not the latter to former.
//
// Since we return a const* to entries in disjoint_view_set, it must be passed
// in as a reference. Algorithm is N^2 based on number of dims in reference,
// but generating the disjoint view set is likely the limiter on perf of this
// function.
DisjointViewSetInfo getDisjointViewSetsOf(
    Fusion* fusion,
    TensorView* of,
    DisjointSets<IterDomain*>& disjoint_view_set);

// Structure to hold byte multiples for break points. I.e. if we have the
// tensors:
// T0[I0, I1] float
// T1[I0, I1] bool
// T2[I0]     half
// T3    [I1] double
// and a break point of 1 the multiples would be:
// lhs_multiple = 4 + 1 + 2 = 7
// rhs_multiple = 4 + 1 + 8 = 13
struct BroadcastMultiple {
  int64_t rhs_multiple = 0;
  int64_t lhs_multiple = 0;
};

struct BroadcastMultipleInformation {
  std::vector<int> view_disjoint_set_ids;
  std::vector<BroadcastMultiple> broadcast_multiples;
};

// Returns a vector of size reference_tv->getMaybeRFactorDomain().size() which
// is a view disjoint set id of each of those iter domains. If entries share the
// same value, they undergo view transformations in the fusion together.
// Broadcast multiples are also of size
// reference_tv->getMaybeRFactorDomain().size(), each entry [i] is the number of
// inputs/outputs that have a non-broadcast dimension mapped to the
// corresponding dimension in reference_tv. Broadcast multiples includes
// reference_tv if reference_tv is an input or output. Broadcast multiples is
// multiplied by data type size. In the case of view operations the broadcast
// multiple is the full multiple size if any domain in the group maps to a
// non-broadcast dimension in the given input/output. Otherwise if all
// dimensions are broadcast that input/output will not contribute to the
// multiple.
TORCH_CUDA_CU_API BroadcastMultipleInformation
getBroadcastMultiples(TensorView* reference_tv, DataType index_type);

namespace matmul_utils {
//! Utilities in this namespace facilitates scheduling matmul kernels with
//!  hierarchichal tiling specified in MatMulTileOptions.

//! Schedule utility for matmul prolog:
//!   Use all the threads on a CTA tile to load matmul operands
//!  into shared memory with the given vectorization word.
//! TODO:
//!  will need to add bank conflict removal swizzle in a follow up.
TORCH_CUDA_CU_API void scheduleContiguousVectorLoad(
    TensorView* tv,
    MatMulTileOptions tile,
    int vector_word,
    bool vectorize = true);

//! Schedule utility for mma output in matmul main loop:
//!  Realize the hierarchical tiling based on the given tiling options.
//! TODO: rewrite this one with makeTile
TORCH_CUDA_CU_API void scheduleWarpTileWithReduction(
    TensorView* tv,
    MatMulTileOptions tile);

//! Schedule utility for mma output in matmul main loop:
//!  Realize the hierarchical tiling based on the given tiling options
//! on consumers of mma ops in epilog.
//! TODO: remove this one eventually.
TORCH_CUDA_CU_API void scheduleWarpTileWithNoReduction(
    TensorView* tv,
    MatMulTileOptions tile);

//! Lower level primitive spliting inner iterdomains into tiles:
//! Eg.
//!  A[B,I0,I1,I2] -> makeTile({1,2,3})
//! Gives A[B, I0o, I1o, I2o, I0i(1), I1i(2), I2i(3)]
TORCH_CUDA_CU_API void makeTile(TensorView* tv, std::vector<int> tile_sizes);

//! Order the inner tile dimensions as the original order in
//!  root domain. Also putting broadcast domains on the left.
//! Eg. A[I0o,I1o,B2o,I0i,I1i,B2i] (root domain: I1,B,I0)
//! -> A[I0o, I1o, B2o, B2i, I1i, I0i]
//! This is used to facilitate data layout swizzling and
//!  defining vectorized loads.
TORCH_CUDA_CU_API void orderTiledConcreteIdAsRoot(TensorView* tv);

//! Orders the root id ordering of the given tv as
//! [Batch, Previous Reduction, M, N, K]
//!  for easier processing of later scheduling steps.
//!
//! This matching works on root domain only, and
//!  will throw if the tv has a leaf iterdomain that is
//!  not a root id.
TORCH_CUDA_CU_API void canonicalizeMmaTvOrdering(TensorView* tv);

} // namespace matmul_utils

//! Propagate current transformations on from_tv up to the given
//!  position, to all tensorviews on the owning fusion that has
//!  a connection with `from_tv` on the fusion graph.
TORCH_CUDA_CU_API void transformPropagateToAllFrom(
    TensorView* from_tv,
    int pos);

//! A type of custom transform propagator that propagates iterdomain
//!  transforms from a source tv to all tvs that are selected
//!  using a "direction" and a "boundary".
//!
//! The propagation model always assumes a `from_tv`, a `direction` and a
//! `boundary`.
//!
//! This propagator will only transform producers and consumers
//! of `from_tv`, and all propagation modes **require** a boundary to be
//! specified to signify where the propagation should stop.
//!
//! There are currently three modes of propagation: forward, backward and
//! both-way, see comment on the interface functions for details.
struct TORCH_CUDA_CU_API BoundedDirectionalTransformPropagator {
  //! Custom option container for configuring
  //!  the transform propagation actions.
  //! All option values default to false unless
  //!  the corresponding setter is called.
  struct Options {
    //! If true, the transform propagator will
    //!   also propagate parallel types from
    //!   `from_tv` to all selected tvs.
    bool propagate_parallel_type = false;

    //! If true, the specified boundary tvs
    //!  will also be replayed as `from_tv`.
    //!  If false, they will not be affected
    //!  by the propagation pass.
    bool transform_boundary = false;

    //! Sets the position boundary in parallel
    //!  type propagation, see comment on
    //!  scheduler_utils::parallelizeAllLike.
    //! Only used if propagate_parallel_type==true.
    int parallel_propagation_pos = -1;

    //! Setter for enabling parallel type
    //!  propagation. see comment on the variable.
    //!
    //! \param up_to_pos, sets the parallel type
    //!  propagation boundary. see comment on
    //!  scheduler_utils::parallelizeAllLike.
    Options propagateParallelType(int up_to_pos = -1) {
      propagate_parallel_type = true;
      parallel_propagation_pos = up_to_pos;
      return *this;
    }

    //! Setter for enabling propagation to
    //!  boundary tvs. see comment on the variable
    Options propagateToBoundary() {
      transform_boundary = true;
      return *this;
    }
  };

  //! Replay transforms from tensorview `from`
  //!  to the tensorviews that are consumers
  //!  of boundary tensorviews in `to` and producers of `from`.
  static void backward(
      TensorView* from,
      int pos,
      std::vector<TensorView*> to,
      c10::optional<Options> options = c10::nullopt);

  //! Replay transforms from tensorview `from`
  //! to the tensorviews that are producers
  //!  of boundary tensorviews in `to` and consumers of `from`.
  static void forward(
      TensorView* from,
      int pos,
      std::vector<TensorView*> to,
      c10::optional<Options> options = c10::nullopt);

  //! Replay transforms from tensorview `from`
  //!  to all the tensorviews that are consumers
  //!  of tensorviews in `backward_to` and producers
  //!  of tensorviews in `forward_to` while being
  //!  either a producer or a consumer of tensorview `from`.
  static void bothWays(
      TensorView* from,
      int pos,
      std::vector<TensorView*> backward_to,
      std::vector<TensorView*> forward_to,
      c10::optional<Options> options = c10::nullopt);

 private:
  //! Utility function:
  //!  Will realize the transform propagation to the
  //! tensorview's in `included_tvs`.
  //!  Assumes that all tvs in included_tvs are either
  //! a producer or a consumer of from_tv.
  static void propagate(
      TensorView* from_tv,
      int pos,
      std::unordered_set<TensorView*> included_tvs,
      Options options);
};

// Schedulers typically start by merging some axes together then splitting,
// and propagating those transformations through the dag. What we want to
// understand is if these merges can be supported through view operations.
// For example it could be problematic to support a reduction fusion:
//
// tv0[2, 3, 4]
// tv1 = sum(tv0, {1, 2})
// tv2 = view(tv0, {6, 4})
//
// Since the first step of the reduction scheduler would be tv1->merge(1, 2).
// If we tried to propagate this transformation through the view it would make
// the view invalid. If we tried to propagate the view through the reduction,
// it would attempt to merge a reduction and non-reduction dimension. So for
// these types of fusions we would like to understand that the view considers
// axis 1 and 2 of tv1 as "non-separable" axes.
//
// If IterDomains are disjoint in the returned set, then they are considered
// "separable".
// Warning: This pass generates the IdGraphs, not intended for use at runtime.
TORCH_CUDA_CU_API DisjointSets<IterDomain*> disjointViewSets(Fusion* fusion);

// Makes sure that there are no group id's left of pos that match right of pos.
// e.g.
// [1, 0, 0] pos 2 would return false
// [1, 0, 0] pos 1 would return true
TORCH_CUDA_CU_API bool breakIsDisjoint(std::vector<int> group_ids, int pos);

// Generates an old to new map to reorder tv's domain as the rfactor order.
// Priority is given to inner most dimensions for example:
// rfactor [i0, i1, i2]
// domain [i0*i2, i1]
// will produce the map {{0, 1}, {1, 0}}
// This is somewhat similar to orderTiledConcreteIdAsRoot
TORCH_CUDA_CU_API std::unordered_map<int, int> domainReorderAsRfactorMap(
    TensorView* tv);

// Assumes view's are consistent as detected by
// registery.cpp::requiresForwardViewReplay returning false
void propagateViewTransforms(Fusion* fusion, const ComputeAtMap& ca_map);

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
