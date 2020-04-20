#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <map>
#include <ostream>
#include <stack>

namespace torch {
namespace jit {
namespace fuser {

// TODO: Change lowering so it can be called multiple times. It would be good to
// keep user references intact so they can lower it as they describe the kernel.
// Right now we can only lower once.

struct TORCH_CUDA_API GPULower : public OptOutMutator {
 private:
  bool lowered = false;
  Fusion* const fusion_;
  std::vector<Expr*> lowered_exprs;
  Expr* active_scope = nullptr;

  // Track the last computeAt TensorView and axis
  const TensorView* active_view;
  unsigned int active_view_axis;

  // Clear out the last recorded computeAtView
  void clearActiveView();
  // Set active views from computeAtView
  void setActiveView(const TensorView* const);

  // Indexing functions
  // Consumer = Producer
  // i.e. T0 = T1... -> T0 is the consumer, T1 is the producer
  // Producer indexing dispatch
  TensorIndex* getProducerIndex(TensorView* producer, TensorView* consumer);
  // Producer if it's in global memory
  TensorIndex* getGlobalProducerIndex(
      TensorView* producer,
      TensorView* consumer);
  // Producer indexing if it's in registers
  TensorIndex* getLocalProducerIndex(
      TensorView* producer,
      TensorView* consumer);
  // Consumer index dispatch
  TensorIndex* getConsumerIndex(TensorView* consumer);
  // Consumer indexing if it's in global memory
  TensorIndex* getGlobalConsumerIndex(TensorView* consumer);
  // Consumer indexing if it's in local memory
  TensorIndex* getLocalConsumerIndex(TensorView* consumer);

  // Get a predicate based on a particular tensorview
  IfThenElse* getPredicate(const TensorView* const);

  // Wrap pushBack in lower_utils if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Custom dispatch for Expr, want to find out of it's a TV op
  Statement* mutate(Expr*) final;

  // Open the for loop.
  Statement* mutate(ForLoop*) final;

  // Open the for loop.
  Statement* mutate(IfThenElse*) final;

  // Remake operations with TensorIndex
  Statement* mutate(UnaryOp*) final;
  Statement* mutate(BinaryOp*) final;

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
