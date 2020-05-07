#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
namespace torch {
namespace jit {
namespace fuser {

struct UnrollPass : public OptOutDispatch {
 private:
  std::unordered_map<Expr*, Expr*> loop_replacement_map;
  Fusion* fusion_;
  const std::vector<Expr*>& incoming_exprs_;

  // Keep all for loops conveniently to make unrolling easier
  std::vector<ForLoop*> for_loops;

  // keep track if we're within an unrolled loop
  bool within_unroll = false;

  // Track the last computeAt TensorView and axis
  const TensorView* active_view;
  unsigned int active_view_axis;

  // Custom dispatch for Expr, want to find out of it's a TV op
  void handle(Expr*) final;

  // Open the for loop.
  void handle(ForLoop*) final;

  UnrollPass(Fusion* _fusion, const std::vector<Expr*>& _incoming_exprs)
      : fusion_(_fusion), incoming_exprs_(_incoming_exprs) {}

  void computeMap();

 public:
  static std::vector<Expr*> runPass(
      Fusion* fusion,
      const std::vector<Expr*>& exprs);
};

struct TORCH_CUDA_API LoopNestGenerator : public OptOutDispatch {
 private:
  std::vector<Expr*> lowered_exprs;
  Fusion* fusion_;

  // Track the last computeAt TensorView and axis
  const TensorView* active_view;
  unsigned int active_view_axis;

  // Keep all for loops conveniently to make unrolling easier
  std::vector<ForLoop*> for_loops;

  // Get Register allocation statement for tensorview
  void pushAlloc(TensorView*);

  // Clear out the last recorded computeAtView
  void clearActiveView();
  // Set active views from computeAtView
  void setActiveView(const TensorView* const);

  // Open a new inner most for loop
  void openFor(IterDomain*);

  // Wrap pushBack in lower_utils if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Update for loop structure based on this TensorView
  void updateLoopNest(TensorView*);

  // Check if a TV op, generate for loop nest around it
  void handle(Expr*) final;

  // Generate the loop nest structure and place it in lowered_exprs
  void generate();

  LoopNestGenerator(Fusion* _fusion) : fusion_(_fusion) {}

 public:
  static std::vector<Expr*> getLoopNest(Fusion* fusion) {
    FusionGuard fg(fusion);
    LoopNestGenerator lng(fusion);
    lng.generate();
    return lng.lowered_exprs;
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch