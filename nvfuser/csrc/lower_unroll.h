#pragma once
#include <c10/macros/Export.h>

#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <lower_thread_predicate.h>
#include <lower_utils.h>
#include <root_domain_map.h>

#include <bitset>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Unroll pass
//!
//! A bit deceptively: UnrollPass adds all predicates, so it needs to be run
//! even if we don't unroll any loops.
//!
//! Unrolling pass will get IR that looks something like:
//! for( i : I0o{ceil(I0/4)} ) {
//!   for( j : I1o{ceil(I1/128)} ) {
//!     for( k : I0i{4} )
//!       for( l : I1i{128} )
//!         T0[I0o{ceil(I0/4)}, I1o{ceil(I1/128)}, I0iU{4}, I1i{128}] = ...
//!
//! And it will return the following:
//! for( i : I0o{ceil(I0/4)} ) {
//!   for( j : I1o{ceil(I1/128)} ) {
//!
//!     if( i * 4 + 3 < I && j * 128 + 127 < J ){
//!       for( k : I0i{4} )
//!         for( l : I1i{128} )
//!           T0[ ( i * 4 + k ) * J + j * 128 + l ] = ...
//!     } else {
//!       for( k : I0i{4} )
//!         for( l : I1i{128} )
//!           if( i * 4 + k < I && j * 128 + l < J)
//!              T0[ ( i * 4 + k ) * J + j * 128 + l ] = ...
//!     }
//!
//!   }
//! }
//!
//! As can be seen it generates two sets of loops for I0i{4} and I1i{128}. The
//! first set is protected by a predicate that makes sure there's a full
//! internal tile we can iterate over. This way we remove the predicate nested
//! in the inner most loop. There's of course a second set of loops, which has a
//! predicate still in the inner most loop, making sure that we cover edges and
//! corners.
//!
class TORCH_CUDA_CU_API UnrollPass : kir::ExprMutator {
 public:
  // Take the incoming exprs and run loop unrolling, returning the new IR
  static std::vector<Expr*> runPass(
      Fusion* fusion,
      const std::vector<Expr*>& exprs);

  static bool canOmitElseClause(kir::ForLoop* fl);

 private:
  void registerReplace(Expr* reference, Expr* new_expr, kir::Scope* scope);

  // Generate the for Expr replacement map
  UnrollPass(const std::vector<Expr*>& exprs);

  const std::unordered_map<Expr*, Expr*>& replacementMap() const {
    return expr_replacement_map_;
  }

  using OptOutDispatch::handle;

  void handle(kir::ForLoop* fl) final;

  void handle(Expr* expr) final;

 private:
  // We will track which loops in the incoming IR will be replaced and by what
  std::unordered_map<Expr*, Expr*> expr_replacement_map_;

  // keep track if we're within an unrolled loop
  bool look_for_unroll_ = true;

  // Indicates if the currently visited expression is inside a
  // unswitched path
  bool unswitched_loop_ = false;

  // As we generate inline predicates check if we actually generated a
  // non-trivial one.
  bool non_trivial_pred_found_ = false;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
