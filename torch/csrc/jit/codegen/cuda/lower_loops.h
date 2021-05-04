#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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
class TORCH_CUDA_CU_API LoopNestGenerator : public OptOutDispatch {
 public:
  static std::vector<Expr*> loweredExprs(
      Fusion* fusion,
      ThreadPredicateMap& thread_predicates,
      const std::vector<Expr*>& exprs) {
    FUSER_PERF_SCOPE("LoopNestGenerator::loweredExprs");
    LoopNestGenerator generator(fusion, thread_predicates, exprs);
    return generator.lowered_exprs;
  }

 private:
  LoopNestGenerator(
      Fusion* fusion,
      ThreadPredicateMap& thread_predicates,
      const std::vector<Expr*>& exprs);

  // Create the allocation for tv, place it inside the loop associated with
  // alloc_id, return the node
  Expr* pushAlloc(TensorView*);

  // Fusion shared_memory values
  // Tracks if shared memory is modified
  std::unordered_map<Val*, bool> smem_;

  // Track dynamic shared memory buffers
  // Insert allocation at the beginning of the kernel
  std::deque<kir::Allocate*> dynamic_smem_;

  // Clear the modify status for all shared memory buffers
  void cleanSharedMemory();

  // Toggle modify status for this shared memory buffer
  void modifySharedMemory(Val* key);

  // Return the status of the shared memory buffer
  // False if TensorView is not shared memory buffer
  bool isModifiedSharedMemory(Val* key) const;

  // Open a new inner most for loop, track which TV it was constructed from
  // according to the computeAt chain.
  void openFor(std::pair<IterDomain*, TensorView*>);

  // Close the inner most for loop
  void popFor();

  // Wrap pushBack in lower_utils if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Initialize a buffer to init_val. If this buffer is in smem or registers,
  // pass in its allocation statement so we can make sure that we insert this
  // initialization after the allocation.
  void initReduction(TensorView* tv, Val* init_val, Expr* alloc_expr = nullptr);

  // Check if expr is a TV op and handle accordingly.
  void handle(Expr*) final;

  // Run the pass and accumulate output in lowered_exprs
  void generate(const std::vector<Expr*>& exprs);

 private:
  // Track number of allocations in each for loop. It is used to insert
  // allocations in the correct order, which is necessary for memory aliasing
  std::unordered_map<kir::ForLoop*, size_t> for_loop_allocations_;

  // Lowered exprs to return
  std::vector<Expr*> lowered_exprs;

  // Fusion pointer for convenience
  Fusion* fusion_;

  // Keep all for loops conveniently to make unrolling easier, basically just a
  // stack of the active for_loops
  std::vector<kir::ForLoop*> for_loops;

  // Track the active computeAt scope, and what view we're "computeAt-ing" into
  std::vector<std::pair<IterDomain*, TensorView*>> compute_at_scope;

  // Predicates from ThreadPredicates that we will extend to reduction buffer
  // initialization
  ThreadPredicateMap& thread_predicates_;

  // Kernel IR builder
  kir::IrBuilder ir_builder_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
