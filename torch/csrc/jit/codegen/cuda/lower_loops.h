
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Loop nest generator pass will get IR that looks something like:
//! T0[I0o{ceil(I0/4)}, I1o{ceil(I1/128)}, I0iU{4}, I1i{128}] = ...* for( i :
//! I0o{ceil(I0/4)} ) { and will generate the loop nest structure for these
//! exprs like:
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
  static std::vector<kir::Expr*> loweredExprs(const std::vector<Expr*>& exprs);

 private:
  LoopNestGenerator(const std::vector<Expr*>& exprs);

  // Open a new inner most for loop, track which TV it was constructed from
  // according to the computeAt chain.
  void openFor(IterDomain*);

  // Close the inner most for loop
  void closeFor();

  // Appends an expression to the current scope
  void pushFront(kir::Expr* expr);

  void handle(Expr* expr);

  // Run the pass and accumulate output in lowered_exprs_
  void generate(const std::vector<Expr*>& exprs);

 private:
  // Lowered exprs to return
  std::vector<kir::Expr*> lowered_exprs_;

  // Keep all for loops conveniently to make unrolling easier, basically just a
  // stack of the active for_loops
  std::vector<kir::ForLoop*> for_loops_;

  // How many loops can the next iteration close
  std::ptrdiff_t max_close = -1;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
