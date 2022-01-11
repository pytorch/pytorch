
#pragma once

#include <torch/csrc/Export.h>

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

using IterDomainMap = std::unordered_map<kir::IterDomain*, kir::IterDomain*>;

namespace scope_utils {

//! Returns the list of nesting loops starting at `scope`
// Primarily used in indexing, maybe could be moved there
std::vector<kir::ForLoop*> getLoops(kir::Expr* scope);

//! Insert expr in scope before ref
//!
//! \warning for kir::IfThenElse we implicitly insert in the "then" branch!
//!
void insertBefore(kir::Expr* scope, kir::Expr* ref, kir::Expr* expr);

//! Create an **empty** Forloop and copy the metadata.
kir::ForLoop* cloneForLoop(kir::IrBuilder& ir_builder, kir::ForLoop* for_loop);

//! Create an **empty** IfThenElse and copy the metadata.
kir::IfThenElse* cloneIfThenElse(
    kir::IrBuilder& ir_builder,
    kir::IfThenElse* ite);

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

bool isTV(const Val* const);

TORCH_CUDA_CU_API bool isTVOp(const Expr*);

bool isTVOp(const kir::Expr* expr);

TensorView* getTVOutput(const Expr*);
kir::TensorView* getTVOutput(const kir::Expr*);

bool isScalarOp(const Expr*);

// TODO(kir): remove
Expr* asExpr(Statement*);

// TODO(kir): Remove in favor of ->as<TensorView>()
TensorView* asTV(Val*);

//! Get kir::TensorView potentially via kir::TensorIndex. Returns nullptr if
//! cast fails.
kir::TensorView* getTv(kir::Val*);

//! Get only kir::TensorView potentially via kir::TensorIndex.
std::vector<kir::TensorView*> getTvs(const std::vector<kir::Val*>& vals);

//! Get kir::TensorView potentially via kir::TensorIndex. Error if cast fails.
kir::TensorView* asTv(kir::Val*);

//! Get kir::TensorView potentially via kir::TensorIndex. Error if cast fails.
std::vector<kir::TensorView*> asTvs(const std::vector<kir::Val*>& vals);

bool hasBlockSync(const Expr* expr, const ThreadPredicateMap& pred_map);
bool hasBlockSync(const kir::Expr* expr, const ThreadPredicateMap& pred_map);

// expr_replacement_map maps an expression to its replacement.
//
// The applyReplacement function serves two purposes.
//
// 1. If expr is found in expr_replacement_map, return the value for expr key.
// Otherwise, return the original expression.
//
// 2. If a replacement is not found and the expression is a ForLoop or an
// IfThenElse, it modifies the expressions in its scope by running the
// handle_scope function
//
// The handle_scope function iterates over the expressions in the scope.
// For each expression, it updates the expression the value returned by
// applyReplacement.
kir::Expr* applyReplacements(
    const std::unordered_map<kir::Expr*, kir::Expr*>& expr_replacement_map,
    kir::Expr* expr);

//! Returns the Fuser iterdomain that maps to the thread dimension grouped
//!  to warps. Returns nullopt if the reduction is not to be lowered to
//!  a warp reduction.
c10::optional<IterDomain*> getMaybeWarpReductionDim(
    const kir::ReductionOp* node);

c10::optional<IterDomain*> getMaybeWarpReductionDim(const ReductionOp* node);

//! Return true if axis is derived from a root axis that is an input
//! to a CA leaf axis.
bool derivedFromRootCAAxes(const TensorView* tv, IterDomain* axis);

std::unordered_map<ParallelType, kir::IterDomain*, TypeHash> getParallelDomains(
    kir::Val* val);

} // namespace ir_utils

namespace loop_utils {

// I wanted to make the tv's in these util functions constant, but that started
// a long const-ness project going into TensorView (making functions const
// there) then into lower_loops where we sort exprs.
// TODO: We should fix this when we have some time.

// Figure out which loop the allocation needs to be in. Returns nullptr if
// outside the first loop in loops. Also find out which index in tv the
// first dimension that needs to be allocated is. Meaning we need to allocate
// that local axis and above.
// TODO: Only remaining use of this is in index compute, remove use from there,
// or refactor and use in lower_allocation
std::pair<kir::ForLoop*, int64_t> getAllocPoint(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    bool use_id_map);

std::pair<kir::ForLoop*, int64_t> getAllocPoint(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops);
} // namespace loop_utils

// Replace value pass on Kernel IR.
//  Replace each use of any kir::Val* that apears in the given `replacement_map`
//  Keeps the predicate carried by each expr
//
// Warning: Blindly replaces all use based on pointer
// Warning: May invalidate indexing if replacing uses of allocated values
std::vector<kir::Expr*> replaceInputsInExpr(
    const std::vector<kir::Expr*>& exprs,
    const std::unordered_map<kir::Val*, kir::Val*>& replacement_map);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
