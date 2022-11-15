#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: Replace with mutator as IndexLowering is replacing expr's with
// versions that are doing indexing
class TORCH_CUDA_CU_API IndexLowering : private OptOutConstDispatch {
 public:
  static std::vector<Expr*> getIndexedExprs(std::vector<Expr*> incoming_exprs) {
    FUSER_PERF_SCOPE("GpuLower::Lower::IndexLowering::getIndexedExprs");
    IndexLowering il;
    il.generate(incoming_exprs);
    return il.lowered_exprs_;
  }

 private:
  IndexLowering() = default;

  void pushBack(Expr*);

  // Return the most recently inserted
  //  expression in the current active
  //  scope or global scope.
  Expr* back() const;

  // Insert an expression before the current top-level expression.
  void insertAtTopLevel(Expr* expr);

  void handle(const FullOp*) final;
  void handle(const ARangeOp*) final;
  void handle(const EyeOp*) final;
  void handle(const ViewAsScalar*) final;
  void handle(const UnaryOp*) final;

  void handle(const BinaryOp*) final;
  void handle(const TernaryOp*) final;
  void handle(const SelectOp*) final;
  void handle(const RNGOp*) final;
  void handle(const ReductionOp*) final;
  void handle(const GroupedReductionOp*) final;
  void handle(const WelfordOp*) final;
  void handle(const GroupedWelfordOp*) final;
  void handle(const LoadStoreOp*) final;
  void handle(const MmaOp*) final;
  void handle(const BroadcastOp*) final;

  void handle(const kir::ForLoop*) final;
  void handle(const kir::IfThenElse*) final;
  void handle(const kir::Allocate*) final;
  void handle(const kir::BlockSync*) final;
  void handle(const kir::GridSync*) final;
  void handle(const kir::CpAsyncWait*) final;
  void handle(const kir::CpAsyncCommit*) final;

  void generate(const std::vector<Expr*>& exprs);

  // lower index for producer. The `override_index` is a mapping `id->index`,
  // where `id` must be an IterDomain in the rFactor domain of the producer.
  // This is can used to manually set the index for the given rFactor ID.
  // Currently, this `override_index` is only used by indexing ops like
  // select/index_select.
  Val* lowerSrcIndex(
      Val* val,
      Val* dst,
      const std::unordered_map<IterDomain*, Val*>& override_index = {}) const;

  Val* lowerDstIndex(Val* dst) const;

  void handleBlockReduction(const ReductionOp* rop, Val* out, Val* in);
  void handleGridReduction(const ReductionOp* rop, Val* out, Val* in);

  void handleBlockReduction(
      const GroupedReductionOp* rop,
      const std::vector<Val*>& outputs,
      const std::vector<Val*>& inputs);
  void handleGridReduction(
      const GroupedReductionOp* rop,
      const std::vector<Val*>& outputs,
      const std::vector<Val*>& inputs);

  void handleGridWelford(WelfordOp* new_wop);

  void handleGroupedBlockWelford(
      const GroupedWelfordOp* wop,
      const std::vector<WelfordTriplet>& output_vals,
      const std::vector<WelfordTriplet>& input_vals,
      const std::vector<WelfordTriplet>& init_vals);
  void handleGroupedGridWelford(
      const GroupedWelfordOp* wop,
      const std::vector<WelfordTriplet>& output_vals,
      const std::vector<WelfordTriplet>& input_vals,
      const std::vector<WelfordTriplet>& init_vals);

  // Allocate a unique buffer for grid reductions and broadcast. A
  // buffer is uniquely allocated for each output tensor of an
  // expression.
  kir::Allocate* allocateUniqueBuffer(
      Val* buffer_size,
      DataType dtype,
      bool zero_init,
      TensorView* out_tv,
      std::unordered_map<TensorView*, kir::Allocate*>& alloc_map);

  std::vector<kir::Allocate*> allocateWelfordWorkBuffer(
      const std::vector<WelfordTriplet>& triplets,
      WelfordTriplet::ValName name,
      Val* buffer_size);

  // Allocate a fused reduction object uniquely for a given
  // TensorView. Parameter expr is the expression corresponding to the
  // fused reduction.
  void allocateUniqueFusedReduction(Expr* expr, TensorView* out_tv);

 private:
  std::vector<Expr*> lowered_exprs_;

  // This is a slight work around as scope has a couple definitions, we have the
  // Scope that's in ForLoop/IfThenElse which is really just a wrapper around
  // std::vector<Expr*> and then we have the actual ForLoop/IfThenElse. We want
  // to be able to carry both around because when we push back to a scope it
  // could be either the body or else body of the IfThenElse. However, we want
  // to understand the nesting of IfThenElse/ForLoop nodes.
  kir::Scope* active_scope_ = nullptr;

  // Track for loops to send to indexing. Similar to what's done in
  // kir::IrVisitor
  std::vector<kir::ForLoop*> for_loops_;

  // Maps to keep track of allocated buffers and objects that must be
  // allocated only once
  std::unordered_map<TensorView*, kir::Allocate*> sync_buffer_map_;
  std::unordered_map<TensorView*, kir::Allocate*> work_buffer_map_;
  std::unordered_map<TensorView*, kir::AllocateFusedReduction*>
      fused_reduction_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
