#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {

// Clear out the last recorded computeAtView
void GPULower::clearActiveView() {
  active_view_axis = 0;
  active_view = nullptr;
}

// Set active views from computeAtView
void GPULower::setActiveView(const TensorView* const tv) {
  active_view_axis = tv->getComputeAtAxis();
  active_view = tv->getComputeAtView();
}

void GPULower::pushBack(Expr* expr) {
  if (active_scope == nullptr)
    lowered_exprs.push_back(expr);
  else
    scope_utils::pushBack(active_scope, expr);
}

Statement* GPULower::mutate(Expr* expr) {
  Statement* mutated_stmt = OptOutMutator::mutate(expr);
  TORCH_INTERNAL_ASSERT(
      mutated_stmt->isExpr(),
      "Tried to generate a kernel but hit a non expression during lowering: ",
      mutated_stmt);
  return mutated_stmt;
}

Statement* GPULower::mutate(IfThenElse* ite) {
  Expr* prev_scope = active_scope;
  active_scope = ite;
  std::vector<Expr*> mutated_exprs;
  bool is_mutated = false;
  for (auto expr : ite->body().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    Expr* mutated_expr = ir_utils::asExpr(mutated_stmt);
    mutated_exprs.push_back(mutated_expr);
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  std::vector<Expr*> mutated_else_exprs;
  for (auto expr : ite->elseBody().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    Expr* mutated_expr = ir_utils::asExpr(mutated_stmt);
    mutated_else_exprs.push_back(mutated_expr);
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  if (is_mutated) {
    ite->body().clear();
    for (auto expr : mutated_exprs)
      ite->body().push_back(expr);
    ite->elseBody().clear();
    for (auto expr : mutated_else_exprs)
      ite->elseBody().push_back(expr);
  }

  active_scope = prev_scope;

  if (is_mutated) {
    auto new_ite = new IfThenElse(
        ite->cond(), mutated_exprs, mutated_else_exprs, ite->parentScope());
    return new_ite;
  }

  return ite;
}

Statement* GPULower::mutate(ForLoop* fl) {
  Expr* prev_scope = active_scope;
  active_scope = fl;
  std::vector<Expr*> mutated_exprs;
  bool is_mutated = false;
  for (auto expr : fl->body().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    Expr* mutated_expr = ir_utils::asExpr(mutated_stmt);
    mutated_exprs.push_back(mutated_expr);
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  active_scope = prev_scope;

  if (is_mutated) {
    auto newFL = new ForLoop(
        fl->index(), fl->iter_domain(), mutated_exprs, fl->parentScope());
    return newFL;
  }

  return fl;
}

Statement* GPULower::mutate(UnaryOp* uop) {
  if (!ir_utils::isTVOp(uop))
    return OptOutMutator::mutate(uop);

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(uop->out()), scope_utils::getLoops(active_scope));
  Val* in = uop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(uop->out()),
        scope_utils::getLoops(active_scope));
  Expr* new_op = new UnaryOp(uop->getUnaryOpType(), out, in);

  return new_op;
}

Statement* GPULower::mutate(BinaryOp* bop) {
  if (!ir_utils::isTVOp(bop))
    return OptOutMutator::mutate(bop);

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(bop->out()), scope_utils::getLoops(active_scope));

  Val* lhs = bop->lhs();
  Val* rhs = bop->rhs();

  if (ir_utils::isTV(lhs))
    lhs = Index::getProducerIndex(
        ir_utils::asTV(lhs),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope));

  if (ir_utils::isTV(rhs))
    rhs = Index::getProducerIndex(
        ir_utils::asTV(rhs),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope));

  Expr* new_op = new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs);

  return new_op;
}

Statement* GPULower::mutate(TernaryOp* top) {
  if (!ir_utils::isTVOp(top))
    return OptOutMutator::mutate(top);

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(top->out()), scope_utils::getLoops(active_scope));
  Val* in1 = top->in1();
  Val* in2 = top->in2();
  Val* in3 = top->in3();

  if (ir_utils::isTV(in1))
    in1 = Index::getProducerIndex(
        ir_utils::asTV(in1),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope));

  if (ir_utils::isTV(in2))
    in2 = Index::getProducerIndex(
        ir_utils::asTV(in2),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope));

  if (ir_utils::isTV(in3))
    in3 = Index::getProducerIndex(
        ir_utils::asTV(in3),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope));

  Expr* new_op = new TernaryOp(top->getTernaryOpType(), out, in1, in2, in3);

  return new_op;
}

Statement* GPULower::mutate(ReductionOp* rop) {
  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(rop),
      "Cannot have a reduction operation on something other than a tensor view.");
  auto loops = scope_utils::getLoops(active_scope);
  TORCH_INTERNAL_ASSERT(
      std::none_of(
          loops.begin(),
          loops.end(),
          [](ForLoop* fl) {
            return fl->iter_domain()->isBlockDim() &&
                fl->iter_domain()->isReduction();
          }),
      "Reduction on block axes not yet supported.");

  bool is_thread_reduce =
      std::any_of(loops.begin(), loops.end(), [](ForLoop* fl) {
        return fl->iter_domain()->isThreadDim() &&
            fl->iter_domain()->isReduction();
      });

  TensorIndex* out = Index::getConsumerIndex(ir_utils::asTV(rop->out()), loops);

  Val* in = rop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(rop->out()),
        scope_utils::getLoops(active_scope));

  if (is_thread_reduce)
    return new ReductionOp(rop->getReductionOpType(), rop->init(), out, in);

  Expr* new_op = new BinaryOp(rop->getReductionOpType(), out, out, in);

  return new_op;
}

// TensorViews are all based on symbolic sizes. When we first initialize them we
// don't know if they're inputs or outputs which would mean that they have
// runtime shapes. Intermediate tensors (those not going to global memory) do
// not have this information. Since we need to have the correct information in
// the kernel being fetched for shapes, we want to replace input and output
// tensors to reference the runtime structure containing sizes.
void GPULower::replaceSizes() {
  Fusion* fusion = FusionGuard::getCurFusion();
  // Sizes of inputs/outputs -> T.size[...]
  std::unordered_map<Val*, Val*> size_map;

  // Grab inputs and outputs
  std::vector<TensorView*> orig_inp_out;
  std::vector<TensorView*> all_tvs;

  for (auto* val : fusion->inputs())
    if (ir_utils::isTV(val))
      orig_inp_out.push_back(ir_utils::asTV(val));

  for (auto* val : fusion->outputs())
    if (ir_utils::isTV(val))
      orig_inp_out.push_back(ir_utils::asTV(val));

  for (auto* val : fusion->deterministic_vals()) {
    if (ir_utils::isTV(val)) {
      all_tvs.push_back(ir_utils::asTV(val));
    }
  }

  // Run through inputs and outputs first. Since we're replacing full
  // tensorviews their names are going to change. We need  the new referenc
  // name for the inputs/outputs. This way we won't reference the wrong tensor
  // view. For example T0 may be translated to T9. We don't want our new
  // variable to be T0->size[...] we need it to be T9->size[...]
  //
  // This could be done in a better way but changing split/merge/reorder to be a
  // TensorDomain focused operation, then we could simply do this process on
  // domains, instead of tensorviews. This would have the benefit that the
  // TensorView wouldn't change, so users pointers will remain valid. The other
  // option which seems less elegant but would also work is build up the domain
  // on the new tensor, and then simply replace it into the original one.
  for (TensorView* tv : orig_inp_out) {
    // Replace the domain with one based on Ti.size[j]
    std::vector<IterDomain*> new_domain_iters;
    TensorDomain* root_td = tv->getRootDomain();

    for (decltype(root_td->nDims()) i{0}; i < root_td->nDims(); i++) {
      // Output sizes could have reduction axes, which isn't what gets output.
      if (root_td->axis(i)->isReduction())
        continue;

      Val* orig_size = root_td->axis(i)->extent();

      std::stringstream ss;
      ss << "T" << tv->name() << ".size[" << i << "]";
      Val* new_size =
          new NamedScalar(ss.str(), orig_size->getDataType().value());
      if (!orig_size->sameAs(new_size) ||
          size_map.find(orig_size) == size_map.end())
        size_map[orig_size] = new_size;
    }
  }

  // If we already lowered all inputs/outputs we can just return.
  if (size_map.size() == 0)
    return;

  // Set domains to be based on symbolic sizes (i.e. Ti.size[...])
  for (TensorView* tv : all_tvs) {
    std::vector<IterDomain*> new_domain_iters;
    TensorDomain* root_td = tv->getRootDomain();

    for (decltype(root_td->nDims()) i{0}; i < root_td->nDims(); i++) {
      Val* new_size = root_td->axis(i)->extent();
      if (size_map.find(new_size) != size_map.end())
        new_size = size_map[new_size];

      new_domain_iters.push_back(new IterDomain(
          root_td->axis(i)->start(),
          new_size,
          root_td->axis(i)->parallel_method(),
          root_td->axis(i)->isReduction(),
          root_td->axis(i)->isRFactorProduct()));
    }

    TensorDomain* old_domain = tv->domain();
    TensorDomain* new_domain = new TensorDomain(new_domain_iters);

    // We should just be able to replace sizes in place, but mutator is setup to
    // do that as it set up to replace vals in Exprs, but
    // IterDomain/TensorDomain are vals.
    std::vector<int> axis_map(new_domain->nDims());
    std::iota(axis_map.begin(), axis_map.end(), 0);
    new_domain = TransformIter::replaySelf(
        new_domain, TransformIter::getHistory(old_domain), axis_map);

    TORCH_INTERNAL_ASSERT(
        old_domain->nDims() == new_domain->nDims(),
        "Tried to set symbolic sizes through the kernel, but hit a snag, Replayed domain should be the same size as the target domain, but got ",
        new_domain->nDims(),
        " and ",
        old_domain->nDims());
    // Parallelize all iter domains
    for (decltype(new_domain->nDims()) i{0}; i < new_domain->nDims(); i++)
      new_domain->axis(i)->parallelize(old_domain->axis(i)->parallel_method());

    tv->setDomain(new_domain);
  }

  // Adjust memory types to make sure they are valid
  for (TensorView* tv : all_tvs) {
    if (fusion->hasInput(tv) || fusion->hasOutput(tv)) {
      tv->setMemoryType(MemoryType::Global);
    } else {
      if (tv->getMemoryType() == MemoryType::Global)
        tv->setMemoryType(MemoryType::Local);
    }
  }
}

namespace {

// Some pre-compilation checks
void validate(Fusion* fusion) {
  FusionGuard fg(fusion);
  fusion->validateInputs();
  for (Val* val : fusion->vals()) {
    if (ir_utils::isTV(val)) {
      TensorView* tv = ir_utils::asTV(val);
      for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
        IterDomain* id = tv->getComputeAtAxis(i).first;

        if (id->isBlockDim())
          TORCH_CHECK(
              !id->isReduction(),
              "Parallelization across blocks on reduction axes not support at the moment but found on, ",
              tv,
              ".");
      }
    } // if ir_utils::isTV
  } // for(Val* val : fusion->vals())
} // validate

} // namespace

// Remove circular computeAt references
void GPULower::fixComputeAt(Fusion* fusion) {
  FusionGuard fg(fusion);

  std::vector<Expr*> exprs = fusion->exprs(true);
  std::set<TensorView*> visited;
  for (auto it = exprs.rbegin(); it != exprs.rend(); it++) {
    Expr* expr = *it;
    if (!ir_utils::isTVOp(expr))
      continue;

    TensorView* tv = ir_utils::asTV(expr->output(0));
    TensorView* ctv = tv->getComputeAtView();

    if (ctv != nullptr && visited.find(ctv) == visited.end()) {
      ctv->setComputeAt(tv, (int)tv->getComputeAtAxis());
      tv->clearComputeAt();
    }
    visited.emplace(tv);
  }
}

// Traverse through the fusion and print CUDA code associated with it
std::vector<Expr*> GPULower::getLoweredExprs() {
  FusionGuard fg(fusion_);

  // Compute at can have some circular references. Before we can call any tv
  // with tv->getComputeAtAxis(i) we need to break those circular dependencies.
  fixComputeAt(fusion_);

  // Initialize members of the class
  active_view = nullptr;
  active_view_axis = 0;

  validate(fusion_);
  replaceSizes();
  auto loop_nests =
      LoopNestGenerator::getLoopNest(fusion_, fusion_->exprs(true));

  auto unrolled_loops = UnrollPass::runPass(fusion_, loop_nests);

  // Run through loop nests and further lower the expressions
  for (auto* expr : unrolled_loops) {
    Statement* mutated_stmt = mutate(expr);
    TORCH_INTERNAL_ASSERT(
        mutated_stmt->isExpr(),
        "Tried to generate a kernel but hit a non expression during lowering: ",
        mutated_stmt);
    lowered_exprs.push_back(static_cast<Expr*>(mutated_stmt));
  }

  return lowered_exprs;
}

std::ostream& GPULower::printKernel(
    std::ostream& os,
    const std::string& kernel_name) {
  FusionGuard fg(fusion_);
  getLoweredExprs();

  IRPrinter irp(os);
  irp.printKernel(lowered_exprs, kernel_name);
  return os;
}

} // namespace fuser
} // namespace jit
} // namespace torch
