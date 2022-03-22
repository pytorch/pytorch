
#pragma once

#include <c10/macros/Export.h>

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
class TVDomainGuard {
 private:
  TensorView* tv_;
  TensorDomain* prev_domain;

 public:
  explicit TVDomainGuard(TensorView* _tv, TensorDomain* td);

  ~TVDomainGuard();
};

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
bool isTV(const Val* const);

// Returns is Expr is a TensorView or TensorIndex Expr.
TORCH_CUDA_CU_API bool isTvOp(const Expr*);

// Returns the first output of Expr that is a TensorView
TensorView* getTvOutput(const Expr*);

bool hasBlockSync(const Expr* expr, const ThreadPredicateMap& pred_map);

//! Returns the Fuser iterdomain that maps to the thread dimension grouped
//!  to warps. Returns nullopt if the reduction is not to be lowered to
//!  a warp reduction.
c10::optional<IterDomain*> getMaybeWarpReductionDim(const ReductionOp* node);

bool isScalarOp(const Expr*);

//! Get TensorView potentially via kir::TensorIndex. Returns nullptr if
//! cast fails.
TensorView* getTv(Val*);

//! Get only TensorView potentially via kir::TensorIndex.
std::vector<TensorView*> getTvs(const std::vector<Val*>& vals);

//! Return true if axis is derived from a root axis that is an input
//! to a CA leaf axis.
bool derivedFromRootCAAxes(const TensorView* tv, IterDomain* axis);

std::unordered_map<ParallelType, IterDomain*, TypeHash> getParallelDomains(
    Val* val);

} // namespace ir_utils

namespace loop_utils {

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
} // namespace loop_utils

// Replace value pass on Kernel IR.
//  Replace each use of any Val* that apears in the given `replacement_map`
//  Keeps the predicate carried by each expr
//
// Warning: Blindly replaces all use based on pointer
// Warning: May invalidate indexing if replacing uses of allocated values
std::vector<Expr*> replaceInputsInExpr(
    const std::vector<Expr*>& exprs,
    const std::unordered_map<Val*, Val*>& replacement_map);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
