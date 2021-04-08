#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

#include <bitset>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/*
 * A bit deceptively: UnrollPass adds all predicates, so it needs to be run even
 * if we don't unroll any loops.
 *
 * Unrolling pass will get IR that looks something like:
 * for( i : I0o{ceil(I0/4)} ) {
 *   for( j : I1o{ceil(I1/128)} ) {
 *     for( k : I0i{4} )
 *       for( l : I1i{128} )
 *         T0[I0o{ceil(I0/4)}, I1o{ceil(I1/128)}, I0iU{4}, I1i{128}] = ...
 *
 * And it will return the following:
 * for( i : I0o{ceil(I0/4)} ) {
 *   for( j : I1o{ceil(I1/128)} ) {
 *
 *     if( i * 4 + 3 < I && j * 128 + 127 < J ){
 *       for( k : I0i{4} )
 *         for( l : I1i{128} )
 *           T0[ ( i * 4 + k ) * J + j * 128 + l ] = ...
 *     } else {
 *       for( k : I0i{4} )
 *         for( l : I1i{128} )
 *           if( i * 4 + k < I && j * 128 + l < J)
 *              T0[ ( i * 4 + k ) * J + j * 128 + l ] = ...
 *     }
 *
 *   }
 * }
 *
 * As can be seen it generates two sets of loops for I0i{4} and I1i{128}. The
 * first set is protected by a predicate that makes sure there's a full internal
 * tile we can iterate over. This way we remove the predicate nested in the
 * inner most loop. There's of course a second set of loops, which has a
 * predicate still in the inner most loop, making sure that we cover edges and
 * corners.
 */

class TORCH_CUDA_CU_API UnrollPass : public OptOutDispatch {
 private:
  // Wrapper to access thread_predicates_ based on an output TV
  kir::Bool* getThreadPredicate(TensorView*);

  // We will track which loops in the incomming IR will be replaced and by what
  std::unordered_map<Expr*, Expr*> loop_replacement_map;

  // Hold on to a reference to the fusion for convenience
  Fusion* fusion_;

  // Hold on to the incoming exprs, but don't modify them. We don't set the
  // Expr* to be const as Exprs' are const by virtue of their interface design
  const std::vector<Expr*>& incoming_exprs_;

  // Keep all for loops conveniently to make unrolling easier
  std::vector<kir::ForLoop*> for_loops;

  // Map from TensorView
  const ThreadPredicateMap& thread_predicates_;

  std::unordered_map<IterDomain*, IterDomain*> p2c_root_map;

  // keep track if we're within an unrolled loop
  bool look_for_unroll = true;

  // As we generate inline predicates check if we actually generated a
  // non-trivial one.
  bool non_trivial_pred_found = false;

  // Custom dispatch for Expr, want to find out of it's a TV op
  void handle(Expr*) final;

  // Open the for loop.
  void handle(kir::ForLoop*) final;

  // Constructor
  UnrollPass(
      Fusion* _fusion,
      const std::vector<Expr*>& _incoming_exprs,
      const ThreadPredicateMap& _thread_predicates)
      : fusion_(_fusion),
        incoming_exprs_(_incoming_exprs),
        thread_predicates_(_thread_predicates) {
    p2c_root_map = loop_utils::p2cRootMap(_fusion->exprs(true));
  }

  // Generate the for Expr replacement map
  void computeMap();

 public:
  // Take the incoming fusion and exprs and run loop unrolling, returning the
  // new IR.
  static std::vector<Expr*> runPass(
      Fusion* fusion,
      const std::vector<Expr*>& exprs,
      const ThreadPredicateMap& thread_predicates);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
