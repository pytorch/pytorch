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

  // Keep all for loops conveniently to make unrolling easier
  std::vector<ForLoop*> for_loops;
  // computeAT scope is determined by the iterat domain, and the tensor view it
  // belongs to (the final TensorView when following the computeAt path)
  std::vector<std::pair<IterDomain*, TensorView*>> compute_at_scope;

  // Get Register allocation statement for tensorview
  void pushAlloc(TensorView*);

  // Open a new inner most for loop
  void openFor(std::pair<IterDomain*, TensorView*>);
  void popFor();

  // Wrap pushBack in lower_utils if active_scope is null we want it to go
  // straight to lower_exprs
  void pushBack(Expr*);

  // Update for loop structure based on this TensorView
  void updateLoopNest(TensorView*);

  // Update for loop structure based on this TensorView
  void initReduction(TensorView* tv, Val* init_val);

  // Check if a TV op, generate for loop nest around it
  void handle(Expr*) final;

  // Generate the loop nest structure and place it in lowered_exprs
  void generate(const std::vector<Expr*>& exprs);

  LoopNestGenerator(Fusion* _fusion) : fusion_(_fusion) {}

 public:
  static std::vector<Expr*> getLoopNest(
      Fusion* fusion,
      std::vector<Expr*> exprs) {
    FusionGuard fg(fusion);
    LoopNestGenerator lng(fusion);
    lng.generate(exprs);
    return lng.lowered_exprs;
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch