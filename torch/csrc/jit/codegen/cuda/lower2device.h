#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <map>
#include <ostream>
#include <stack>

namespace torch {
namespace jit {
namespace fuser {

// TODO: Change lowering so it can be called multiple times. It would be good to
// keep user references intact so they can lower it as they describe the kernel.
// Right now we can only lower once.

struct TORCH_CUDA_API GPULower : public OptOutDispatch {
 private:
  bool lowered = false;
  Fusion* fusion_;
  std::vector<Expr*> lowered_exprs;
  Expr* active_scope = nullptr;
  // Track the last computeAt TensorView and axis
  const TensorView* active_view;
  unsigned int active_view_axis;

  // Open a new inner most for loop
  void openFor(IterDomain*);
  // Close the inner most for loop
  void closeScope();
  // Close all for loops
  void resetScope();
  // Clear out the last recorded computeAtView
  void clearActiveView();
  // Set active views from computeAtView
  void setActiveView(const TensorView* const);
  // Grab the index variables of the active loop nest
  std::vector<Val*> getLoopIndices();
  // Grab the iterDomains of the active loops
  std::vector<IterDomain*> getLoopIterDomains();
  // Gets the indexing of a TensorView producer. These are values consumed in a
  // TensorView Expr. We use the consumer (left hand side of the =) to compute
  // the indexing into the consumer.
  TensorIndex* getProducerIndex(TensorView* producer, TensorView* consumer);
  TensorIndex* getGlobalProducerIndex(
      TensorView* producer,
      TensorView* consumer);
  TensorIndex* getLocalProducerIndex(
      TensorView* producer,
      TensorView* consumer);
  TensorIndex* getConsumerIndex(TensorView* consumer);
  TensorIndex* getGlobalConsumerIndex(TensorView* consumer);
  TensorIndex* getLocalConsumerIndex(TensorView* consumer);

  // Track how far our for loop scope is
  unsigned int computeForDepth();
  // Push an expr to the active scope
  void pushBack(Expr* expr);
  // Return the parent of the active scope
  Expr* parentScope();

  // Get Register allocation statement for tensorview
  Allocate* getAlloc(TensorView*);
  // Get a predicate based on a particular tensorview
  IfThenElse* getPredicate(const TensorView* const);

  // Custom dispatch for Expr, want to find out of it's a TV op
  void handle(Expr*) final;

  // Remake operations with TensorIndex
  void handle(UnaryOp*) final;
  void handle(BinaryOp*) final;

  // Ignore split/merge/reorder operations,
  // we don't want to print them.
  void handle(Split*) final {}
  void handle(Merge*) final {}
  void handle(Reorder*) final {}

  // Update for loop structure based on producing provided TensorView
  void updateView(TensorView*);

  // TensorViews are all based on symbolic sizes. When we first initialize them
  // we don't know if they're inputs or outputs which would mean that they have
  // runtime shapes. Intermediate tensors (those not going to global memory) do
  // not have this information. Since we need to have the correct information in
  // the kernel being fetched for shapes, we want to replace input and output
  // tensors to reference the runtime structure containing sizes.
  void replaceSizes();

 public:
  // Init printer on ostream
  GPULower(Fusion* _fusion) : fusion_(_fusion) {}

  // print generated code to ostream
  std::vector<Expr*> getLoweredExprs();
  std::ostream& printKernel(
      std::ostream& _os,
      const std::string& kernel_name = "CUDAGeneratedKernel");
};

} // namespace fuser
} // namespace jit
} // namespace torch
