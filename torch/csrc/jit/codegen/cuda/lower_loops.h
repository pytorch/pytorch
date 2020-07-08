#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Loop nest generator pass will get IR that looks something like:
 * T0[I0o{ceil(I0/4)}, I1o{ceil(I1/128)}, I0iU{4}, I1i{128}] = ...* for( i :
 * I0o{ceil(I0/4)} ) { and will generate the loop nest structure for these exprs
 * like:
 *
 * for( i : I0o{ceil(I0/4)} ) {
 *   for( j : I1o{ceil(I1/128)} ) {
 *     for( k : I0i{4} )
 *       for( l : I1i{128} )
 *         T0[I0o{ceil(I0/4)}, I1o{ceil(I1/128)}, I0iU{4}, I1i{128}] = ...
 *
 * It does not generate predicates, but it will generate allocations, and loop
 * nests to initialize reduction buffers.
 *
 */
class TORCH_CUDA_API LoopNestGenerator : public OptOutDispatch {
 private:
  // Lowered exprs to return
  std::vector<Expr*> lowered_exprs;
  // Fusion pointer for convenience
  Fusion* fusion_;

  // Keep all for loops conveniently to make unrolling easier, basically just a
  // stack of the active for_loops
  std::vector<ForLoop*> for_loops;

  // Track the active computeAt scope, and what view we're "computeAt-ing" into
  std::vector<std::pair<IterDomain*, TensorView*>> compute_at_scope;

  // Predicates from ThreadPredicates that we will extend to reduction buffer
  // initialization
  ThreadPredicateMap& thread_predicates_;

  // Create, place, and return the allocation for tv
  Expr* pushAlloc(TensorView*);

  // Open a new inner most for loop, track which TV it was constructed from
  // according to the computeAt chain.
  void openFor(std::pair<IterDomain*, TensorView*>);

  // Close the inner most for loop
  void popFor();

  // Wrap pushBack in lower_utils if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Update for loop structure based on this TensorView, see implementation for
  // more details
  void updateLoopNest(TensorView*);

  // Initialize a buffer to init_val. If this buffer is in smem or registers,
  // pass in its allocation statement so we can make sure that we insert this
  // initialization comes after the allocation.
  void initReduction(TensorView* tv, Val* init_val, Expr* alloc_expr = nullptr);

  // Check if expr is a TV op and handle accordingly.
  void handle(Expr*) final;

  // Run the pass and accumulate output in lowered_exprs
  void generate(const std::vector<Expr*>& exprs);

  LoopNestGenerator(Fusion* _fusion, ThreadPredicateMap& _thread_predicates)
      : fusion_(_fusion), thread_predicates_(_thread_predicates) {}

 public:
  static std::vector<Expr*> getLoopNest(
      Fusion* fusion,
      std::vector<Expr*> exprs,
      ThreadPredicateMap& thread_predicates) {
    FusionGuard fg(fusion);
    LoopNestGenerator lng(fusion, thread_predicates);
    lng.generate(exprs);
    return lng.lowered_exprs;
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
