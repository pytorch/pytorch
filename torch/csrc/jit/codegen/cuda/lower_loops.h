
#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Loop nest generator pass will get IR that looks something like:
//! T0[I0o{ceil(I0/4)}, I1o{ceil(I1/128)}, I0iU{4}, I1i{128}] = ...

//  and will generate the loop nest structure for these exprs like:
//!
//! for( i : I0o{ceil(I0/4)} ) {
//!   for( j : I1o{ceil(I1/128)} ) {
//!     for( k : I0i{4} )
//!       for( l : I1i{128} )
//!         T0[I0o{ceil(I0/4)}, I1o{ceil(I1/128)}, I0iU{4}, I1i{128}] = ...
//!
//! It does not generate predicates, but it will generate allocations, and loop
//! nests to initialize reduction buffers.
class TORCH_CUDA_CU_API LoopNestGenerator {
 public:
  static std::vector<Expr*> loweredExprs(const std::vector<Expr*>& exprs);

 private:
  LoopNestGenerator(const std::vector<Expr*>& exprs);

  // Open a new inner most for loop, track which TV it was constructed from
  // according to the computeAt chain.
  void openFor(IterDomain*);

  // Close the inner most for loop
  void closeFor();

  // Appends an expression to the current scope
  void pushFront(Expr* expr);

  void handle(Expr* expr);

  // Run the pass and accumulate output in lowered_exprs_
  void generate(const std::vector<Expr*>& exprs);

 private:
  // Lowered exprs to return
  std::vector<Expr*> lowered_exprs_;

  // Keep all for loops conveniently to make unrolling easier, basically just a
  // stack of the active for_loops
  std::vector<kir::ForLoop*> for_loops_;

  // Loop structure of each expression
  std::unordered_map<TensorView*, std::vector<IterDomain*>> loop_structures_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
