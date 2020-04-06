#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {

// HELPER NAMESPACE
namespace {

bool isTV(const Val* const val) {
  return val->getValType().value() == ValType::TensorView;
}

// Check if we're a TensorView op that we can generate code for.
bool isTVOp(const Expr* expr) {
  if (expr->nOutputs() == 1 && isTV(expr->output(0)) &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp))
    return true;
  return false;
}

TensorView* asTV(Val* val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return static_cast<TensorView*>(val);
}

const TensorView* asConstTV(const Val* const val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return static_cast<const TensorView*>(val);
}

} // namespace

Allocate* LoopNestGenerator::getAlloc(TensorView* tv) {
  TORCH_INTERNAL_ASSERT(
      !(FusionGuard::getCurFusion()->hasInput(tv) ||
        FusionGuard::getCurFusion()->hasOutput(tv)),
      "Tried to allocate an input or output tensor.");

  std::vector<Val*> alloc_dims;

  for (decltype(tv->nDims()) i = tv->getComputeAtAxis(); i < tv->nDims(); i++) {
    IterDomain* dim = tv->getComputeAtAxis(i);
    if (dim->isThreadDim())
      continue;
    //TORCH_INTERNAL_ASSERT()
    alloc_dims.push_back(dim->extent());
  }

  Val* size;
  if (alloc_dims.size() == 0) {
    size = new Int(1);
  } else {
    size = alloc_dims[0];
    for (decltype(alloc_dims.size()) i{1}; i < alloc_dims.size(); i++) {
      size = mul(size, alloc_dims[i]);
    }
  }
  return new Allocate(tv, size);
}

// Clear out the last recorded computeAtView
void LoopNestGenerator::clearActiveView() {
  active_view_axis = 0;
  active_view = nullptr;
}

// Set active views from computeAtView
void LoopNestGenerator::setActiveView(const TensorView* const tv) {
  active_view_axis = tv->getComputeAtAxis();
  active_view = tv->getComputeAtView();
}

void LoopNestGenerator::openFor(IterDomain* id) {
  Expr* new_scope = scope_utils::openFor(active_scope, id);
  if (active_scope == nullptr) {
    pushBack(new_scope);
  }
  active_scope = new_scope;
}

void LoopNestGenerator::pushBack(Expr* expr) {
  if (active_scope == nullptr)
    lowered_exprs.push_back(expr);
  else
    scope_utils::pushBack(active_scope, expr);
}

/*
 *  This is one of the most complex parts of the code lowering logic. what we
 * need to do is: 1) Reduce loop structure
 *    - Reset all loops if active_view == nullptr (I'm not the last in a series
 * of computeAts)
 *    - Else reduce to active_view_axis if loop_depth > active_view_axis
 *  2) Set active_view(_axis)
 *    - If there is a computeAt set for this TV
 *  3) Open to compute At
 *    - If there is a computeAt set for this TV
 *  4) Allocate the output.
 *  5) If this is a reduction, initialize the output (open for loops to inner
 * most, predicate, initialize, close predicate, close to computeAt) 6) Open to
 * inner most loop 7) Open predicate 8) Run operation 9) Close predicate
 */

// Update fors based on tv.
void LoopNestGenerator::updateLoopNest(TensorView* tv) {
  // 1) Reduce loop structure
  if (active_view != nullptr) {
    // - Else reduce to active_view_axis if loop_depth > active_view_axis
    auto depth = scope_utils::computeForDepth(active_scope);
    for (auto i = depth; i > active_view_axis; i--) {
      active_scope = scope_utils::closeScope(active_scope);
    }
  }

  if (tv->hasComputeAt()) {
    //  2) Set active_view(_axis)
    //    - If there is a computeAt set for this TV
    setActiveView(tv);

    //  3) Open to compute At
    //    - If there is a computeAt set for this TV
    auto depth = scope_utils::computeForDepth(active_scope);

    for (auto i = depth; i < tv->getComputeAtAxis(); i++)
      openFor(tv->getComputeAtAxis(i));
  } else {
    if (active_view != nullptr)
      // If we're the last computeAt of a block, active view should match this
      // tv
      TORCH_INTERNAL_ASSERT(
          tv->sameAs(active_view),
          "Error detected in code lowering. Expected ",
          active_view,
          " but recieved ",
          tv);

    clearActiveView();
  }
  //  4) Allocate the output.
  if (!FusionGuard::getCurFusion()->hasInput(tv) &&
      !FusionGuard::getCurFusion()->hasOutput(tv)) {
    pushBack(getAlloc(tv));
  }
  // TODO:
  //  5) If this is a reduction, initialize the output (open for loops to inner
  //  most, predicate, initialize, close predicate, close to computeAt)

  //  6) Open to inner most loop
  for (decltype(tv->nDims()) i = scope_utils::computeForDepth(active_scope);
       i < tv->nDims();
       i++)
    openFor(tv->getComputeAtAxis(i));
}

// Custom dispatch for Expr, want to find out of it's a TV op
void LoopNestGenerator::handle(Expr* expr) {
  if (!isTVOp(expr))
    return;

  TensorView* out = static_cast<TensorView*>(expr->output(0));
  updateLoopNest(out);

  pushBack(expr);
}

// Generate the loop nest structure and place it in lowered_exprs
void LoopNestGenerator::generate() {
  FusionGuard fg(fusion_);

  // Likely we lowered this fusion, we can simply return the lowered expressions
  // Not the safest approach but good enough for now.
  if (fusion_->lowered && lowered_exprs.size() != 0)
    return;

  TORCH_CHECK(
      !fusion_->lowered,
      "Fusions can only be lowered once as of now. You could reuse the lowering using",
      " std::vector<Expr*> GPULower::getLoweredExprs() the result can be printed as",
      " a kernel with   IRPrinter irp(os); irp.printKernel(lowered_exprs, kernel_name);");

  // Initialize members of the class
  lowered_exprs = std::vector<Expr*>();
  active_view = nullptr;
  active_view_axis = 0;

  std::vector<Expr*> exprs = fusion_->exprs(true);
  for (auto* expr : exprs)
    handle(expr);
}

} // namespace fuser
} // namespace jit
} // namespace torch