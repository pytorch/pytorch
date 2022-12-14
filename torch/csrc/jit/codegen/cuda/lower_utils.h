
#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/parallel_type_bitmap.h>

#include <bitset>
#include <map>

// Provides utilities for dealing with nested ForLoop and IfThenElse scopes

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ThreadPredicateMap;

using IterDomainMap = std::unordered_map<IterDomain*, IterDomain*>;

namespace scope_utils {

//! Create an **empty** Forloop and copy the metadata.
kir::ForLoop* cloneForLoop(kir::ForLoop* for_loop);

//! Create an **empty** IfThenElse and copy the metadata.
kir::IfThenElse* cloneIfThenElse(kir::IfThenElse* ite);

} // namespace scope_utils

namespace ir_utils {

// Somtimes we want to temporarily view a tensorview with another tensordomain.
// This isn't a permanent transformation, but in indexing we want to index
// producers with a consumer set of indices, so we need to view the producer
// transformed like consumer while we index. This will set the tv with td for
// the life of this context guard.
class TORCH_CUDA_CU_API TVDomainGuard {
 private:
  TensorView* tv_;
  TensorDomain* prev_domain_;

 public:
  explicit TVDomainGuard(TensorView* tv, TensorDomain* td);
  TVDomainGuard(const TVDomainGuard&) = delete;
  TVDomainGuard(TVDomainGuard&&);

  //! An utility to access the tensordomain before the temporary
  //!  view. This is used to retrieve information, like swizzle
  //!  information that can only be reliably kept at the original domain.
  const TensorDomain* prevDomain() const {
    return prev_domain_;
  }

  ~TVDomainGuard();
};

// Create a TVDomainGuard that temporarily view a tensorview with specified
// all-true or all-false contiguity.
TORCH_CUDA_CU_API ir_utils::TVDomainGuard overrideContiguityGuard(
    TensorView* tv,
    bool contiguity);

//! Return inputs of provided IterDomains that are IterDomains. A list
//! of input IterDomain can be optionally given. Otherwise,
//! IterDomains with no defining expression are returned.
std::vector<IterDomain*> iterDomainInputsOf(
    const std::vector<IterDomain*>& input_ids,
    const std::vector<IterDomain*>& all_inputs = {});

// Return inputs of provided IterDomains that are IterDomains, order as the
// second provided vector.
std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order);

// Returns if Val is a TensorView or TensorIndex
TORCH_CUDA_CU_API bool isTV(const Val* const);

// Returns if Expr is a TensorView or TensorIndex Expr.
TORCH_CUDA_CU_API bool isTvOp(const Expr*);

// Returns the first output of Expr that is a TensorView
TORCH_CUDA_CU_API TensorView* getTvOutput(const Expr*);

// Returns the first input of Expr that is a TensorView
TORCH_CUDA_CU_API TensorView* getTvInput(const Expr*);

//! Returns the iterdomain that maps to the thread dimension grouped
//!  to warps. Returns nullopt if the reduction is not to be lowered to
//!  a warp reduction.
c10::optional<IterDomain*> getMaybeWarpReductionDim(
    const Val* output,
    const Val* input);

bool isScalarOp(const Expr*);

//! Get TensorView potentially via kir::TensorIndex. Returns nullptr if
//! cast fails.
TensorView* getTv(Val*);
const TensorView* getTv(const Val*);

//! Get only TensorView potentially via kir::TensorIndex.
std::vector<TensorView*> getTvs(const std::vector<Val*>& vals);

//! Return true if axis is derived from a root axis that is an input
//! to a CA leaf axis.
bool derivedFromRootCAAxes(const TensorView* tv, IterDomain* axis);

std::unordered_map<ParallelType, IterDomain*, TypeHash> getParallelDomains(
    const Val* val);

//! Returns true if the expression will be lowered to
//!  a ldmatrix intrinsic.
bool isLdMatrixOp(const Expr* expr);

//! Returns true if the expression will be lowered to
//!  a cp.async intrinsic.
bool isCpAsyncOp(const Expr* expr);

//! Short-cut for detecting initialization for cpAsync op.
bool isCpAsyncInit(const Expr* expr);

//! Short-cut for matching a singleton expr in a if statement,
//!  which likely becomes a predicated instruction in ptx, eg.:
//!  if(...) {expr;}
//! Returns the expr if it is this pattern.
//! Returns nullptr if the pattern doesn't match.
c10::optional<Expr*> getMaybePredicatedSingleton(Expr* expr);

//! Short-cut for checking if the expression loads from global memory.
bool isGlobalLoad(const Expr* expr);

//! Short-cut for checking if the given expression initializes buffers
//!  for global memory load.
bool isGlobalLoadInit(const Expr* expr);

//! Returns true if the given expression fills the output
//!  tensor with a single scalar.
bool isTensorScalarFillOp(const Expr* expr);

//! Flattens all the scoped exprs, i.e. ForLoop and IfThenElse,
//!  and returns all the exprs in all scopes in the original
//!  linear textural order.
TORCH_CUDA_CU_API std::vector<Expr*> flattenScopedExprs(
    const std::vector<Expr*>& loop_nests);

//! Returns all swizzle ops between the set of iterdomains
//!  in `from` and `to`.
std::vector<Expr*> getAllSwizzlesBetween(
    std::vector<IterDomain*> from,
    std::vector<IterDomain*> to);

// Replace value pass on Kernel IR.
//  Replace each use of any Val* that apears in the given `replacement_map`
//  Keeps the predicate carried by each expr
//
// Warning: Blindly replaces all use based on pointer
// Warning: May invalidate indexing if replacing uses of allocated values
std::vector<Expr*> replaceInputsInExpr(
    const std::vector<Expr*>& exprs,
    const std::unordered_map<Val*, Val*>& replacement_map);

// Go through all expressions and compute a local ordering of loops. operator<
// is implemented based on the concrete_id_dependencies analysis done. If
// there's no dependency between two IDs then order doesn't mater, otherwise we
// can tell which is inner most by checking if there's any dependency
// relationships.
//
// Dependency relationships in concrete_id_dependencies has a "global" view in
// the fusion, so it can resolve ordering by only looking at id's and the
// dependency map.
//
// For example two expressions may have domains: [I0], [I1] Yet we
// won't know the ordering unless we see a domain with: [I0, I1]. This happened
// in advancedIndexing9 (also see AdvancedLowering6) test when merging T5 with
// the group containing T10 (cache of T5, which is post broadcasted output) and
// T6(pre broadcasted output).
// T5 had the domain [0, 1, 2, 3, 4] produce at 3
// T6 had the domain [0, 3, 4] compute at 3
// Merging [0, 1, 2] and [0, 3, 4] resulted in the domain [0, 3, 4, 1, 2]
//
// If ID's are not in filter, we don't care about their ordering and ignore
// them. This is because we're only focused on loops we will have to merge
// across groups. If the domain is not in a produce at position in the producer
// edges, or a compute at position in the consumer edges, the expressions we
// look at may not have a unique ordering.

struct TORCH_CUDA_CU_API IterDomainDependencySorter {
  IterDomainDependencySorter(
      const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
          concrete_id_dependencies,
      std::shared_ptr<const ComputeAtMap> compute_at_map)
      : concrete_id_dependencies_(concrete_id_dependencies),
        compute_at_map_(compute_at_map) {}

  // Return true if id0 should be before id1
  // Orders such that if x maps to {y}, x comes before y in final ordering.
  inline bool operator()(IterDomain* id0, IterDomain* id1) {
    auto concrete_id_0 =
        compute_at_map_->getConcreteMappedID(id0, IdMappingMode::LOOP);
    auto concrete_id_1 =
        compute_at_map_->getConcreteMappedID(id1, IdMappingMode::LOOP);

    if (concrete_id_dependencies_.find(concrete_id_0) !=
        concrete_id_dependencies_.end()) {
      const auto& dependencies_0 = concrete_id_dependencies_.at(concrete_id_0);
      // if id0 depends on id1 it means id1 is inside id0, so id0 < id1
      if (dependencies_0.count(concrete_id_1)) {
        return true;
      }
    }

    return false;
  }

  const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
      concrete_id_dependencies_;
  const std::shared_ptr<const ComputeAtMap> compute_at_map_;
};

} // namespace ir_utils

namespace lower_utils {

bool hasBlockSync(const Expr* expr, const ThreadPredicateMap& pred_map);

// Allocate global buffer for a grid communication calls, i.e. grid reduce, grid
// welford reduce, grid broadcast.
kir::Allocate* allocGlobalBufferForGridComm(
    Val* buffer_size,
    DataType dtype,
    bool zero_init);

struct BasicAllocInfo {
  // The for loop that the initialization of this allocation must be
  // placed in, nullptr if not within a loop
  kir::ForLoop* init_for_loop = nullptr;

  // Keep track of the actual allocation loop. This can be different
  // from init_for_loop only with unswitched shared memory allocations,
  // which are moved outer loops to avoid duplicated allocations. This means
  // that the alloc position may be outside what's expected. Most applications
  // outside lower_allocation is likely looking for init_for_loop which is
  // more directly related to how large an allocation is and how it's used.
  // (see issue #1133).
  kir::ForLoop* alloc_for_loop = nullptr;

  // The allocation position relative to buffer IDs, it could be outside the
  // compute at position if it's shared memory with a compute at inside an
  // unswitch
  size_t alloc_pos = 0;
};

// Fill the above allocation struct based on provided information. id_map is
// used if we're looking at a producer tensor but loops on a consumer tensor.
BasicAllocInfo getAllocInformation(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map = {},
    bool use_id_map = false);

//! Returns true if the expression has a variant that takes a predicate
//!  as an inline argument.
bool supportInlinePredicate(Expr* expr);

} // namespace lower_utils

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
