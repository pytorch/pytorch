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

TensorIndex* GPULower::getGlobalProducerIndex(
    TensorView* producer,
    TensorView* consumer) {
  // Get new reference so replay inline doesn't change the original.
  TensorView* cloned_tv = producer->clone();
  // This replay will ignore reduction dimensions on the producer
  TransformReplay::fullReplay(consumer, cloned_tv);
  TORCH_INTERNAL_ASSERT(
      scope_utils::getLoopIndices(active_scope).size() == cloned_tv->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  const std::vector<Val*> computed_inds = IndexCompute::computeIndices(
      cloned_tv, scope_utils::getLoopIndices(active_scope));

  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == producer->getRootDomain()->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  std::vector<Val*> strided_inds;
  for (decltype(computed_inds.size()) i{0}; i < computed_inds.size(); i++) {
    std::stringstream ss;
    ss << "T" << producer->name() << ".stride[" << i << "]";
    strided_inds.push_back(
        mul(computed_inds[i], new NamedScalar(ss.str(), DataType::Int)));
  }

  // Probably shouldn't ever hit this
  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new TensorIndex(producer, strided_inds);
}

TensorIndex* GPULower::getLocalProducerIndex(
    TensorView* producer,
    TensorView* consumer) {
  TORCH_INTERNAL_ASSERT(
      scope_utils::computeForDepth(active_scope) == producer->nDims(),
      "Expected a tensor with ",
      scope_utils::computeForDepth(active_scope),
      " dimensions but got one with ",
      producer->nDims());

  std::vector<Val*> loopInds = scope_utils::getLoopIndices(active_scope);
  std::vector<IterDomain*> ranges =
      scope_utils::getLoopIterDomains(active_scope);
  std::vector<Val*> computed_inds;
  std::vector<IterDomain*> used_ranges;
  bool unrolled = false;
  for (decltype(loopInds.size()) i{0}; i < loopInds.size(); i++) {
    if (ranges[i]->parallel_method() == ParallelType::Unroll)
      unrolled = true;
    if (!unrolled && producer->hasComputeAt() &&
        i < producer->getComputeAtAxis())
      continue;
    if (ranges[i]->isThread())
      continue;
    computed_inds.push_back(loopInds[i]);
    used_ranges.push_back(ranges[i]);
  }

  for (decltype(computed_inds.size()) i{0}; i < computed_inds.size(); i++) {
    Val* ind = computed_inds[i];
    for (decltype(used_ranges.size()) j{i + 1}; j < used_ranges.size(); j++)
      ind = mul(ind, used_ranges[i]->extent());
    computed_inds[i] = ind;
  }
  if (computed_inds.size() == 0)
    computed_inds.push_back(new Int(0));

  return new TensorIndex(producer, computed_inds);
}

// Producer is the inputs of an expression
TensorIndex* GPULower::getProducerIndex(
    TensorView* producer,
    TensorView* consumer) {
  if (fusion_->hasInput(producer) || fusion_->hasOutput(producer))
    return getGlobalProducerIndex(producer, consumer);
  return getLocalProducerIndex(producer, consumer);
}

TensorIndex* GPULower::getGlobalConsumerIndex(TensorView* consumer) {
  TORCH_INTERNAL_ASSERT(
      scope_utils::getLoopIndices(active_scope).size() == consumer->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  const std::vector<Val*> computed_inds = IndexCompute::computeIndices(
      consumer, scope_utils::getLoopIndices(active_scope));

  TORCH_INTERNAL_ASSERT(
      computed_inds.size() == consumer->getRootDomain()->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  std::vector<Val*> strided_inds;
  for (decltype(computed_inds.size()) i{0}; i < computed_inds.size(); i++) {
    std::stringstream ss;
    ss << "T" << consumer->name() << ".stride[" << i << "]";
    strided_inds.push_back(
        mul(computed_inds[i], new NamedScalar(ss.str(), DataType::Int)));
  }

  // Probably shouldn't ever hit this
  if (strided_inds.size() == 0)
    strided_inds.push_back(new Int(0));

  return new TensorIndex(consumer, strided_inds);
}

TensorIndex* GPULower::getLocalConsumerIndex(TensorView* consumer) {
  TORCH_INTERNAL_ASSERT(
      scope_utils::computeForDepth(active_scope) == consumer->nDims(),
      "Expected a tensor with ",
      scope_utils::computeForDepth(active_scope),
      " dimensions but got one with ",
      consumer->nDims());

  std::vector<Val*> loopInds = scope_utils::getLoopIndices(active_scope);
  std::vector<IterDomain*> ranges =
      scope_utils::getLoopIterDomains(active_scope);
  std::vector<Val*> computed_inds;
  std::vector<IterDomain*> used_ranges;
  bool unrolled = false;
  for (decltype(loopInds.size()) i{0}; i < loopInds.size(); i++) {
    if (ranges[i]->parallel_method() == ParallelType::Unroll)
      unrolled = true;
    if (!unrolled && consumer->hasComputeAt() &&
        i < consumer->getComputeAtAxis())
      continue;
    if (ranges[i]->isThread())
      continue;
    computed_inds.push_back(loopInds[i]);
    used_ranges.push_back(ranges[i]);
  }

  for (decltype(computed_inds.size()) i{0}; i < computed_inds.size(); i++) {
    Val* ind = computed_inds[i];
    for (decltype(used_ranges.size()) j{i + 1}; j < used_ranges.size(); j++)
      ind = mul(ind, used_ranges[i]->extent());
    computed_inds[i] = ind;
  }

  if (computed_inds.size() == 0)
    computed_inds.push_back(new Int(0));

  return new TensorIndex(consumer, computed_inds);
}

// Consumer is the output of an expression
TensorIndex* GPULower::getConsumerIndex(TensorView* consumer) {
  // GLOBAL MEMORY HANDLING
  if (FusionGuard::getCurFusion()->hasInput(consumer) ||
      FusionGuard::getCurFusion()->hasOutput(consumer))
    return getGlobalConsumerIndex(consumer);
  return getLocalConsumerIndex(consumer);
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

  TensorIndex* out = getConsumerIndex(ir_utils::asTV(uop->out()));
  Val* in = uop->in();
  if (ir_utils::isTV(in))
    in = getProducerIndex(ir_utils::asTV(in), ir_utils::asTV(uop->out()));
  Expr* new_op = new UnaryOp(uop->getUnaryOpType(), out, in);

  return new_op;
}

Statement* GPULower::mutate(BinaryOp* bop) {
  if (!ir_utils::isTVOp(bop))
    return OptOutMutator::mutate(bop);

  TensorIndex* out = getConsumerIndex(ir_utils::asTV(bop->out()));
  Val* lhs = bop->lhs();
  Val* rhs = bop->rhs();

  if (ir_utils::isTV(lhs))
    lhs = getProducerIndex(ir_utils::asTV(lhs), ir_utils::asTV(bop->out()));

  if (ir_utils::isTV(rhs))
    rhs = getProducerIndex(ir_utils::asTV(rhs), ir_utils::asTV(bop->out()));

  Expr* new_op = new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs);

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
          root_td->axis(i)->isReduction()));
    }

    TensorDomain* old_domain = tv->domain();
    TensorDomain* new_domain = TransformReplay::fullReplay(
        old_domain, new TensorDomain(new_domain_iters));

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
}

namespace {

// Some pre-compilation checks
void validate(Fusion* fusion) {
  for (Val* val : fusion->vals()) {
    if (ir_utils::isTV(val)) {
      TensorView* tv = ir_utils::asTV(val);
      for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
        IterDomain* id = tv->getComputeAtAxis(i);

        if (id->isThread())
          TORCH_CHECK(
              !id->isReduction(),
              "Parallelization on reduction axes not support at the moment found on, ",
              tv,
              ".");
      }
    } // if ir_utils::isTV
  } // for(Val* val : fusion->vals())

} // validate
} // namespace

// Traverse through the fusion and print CUDA code associated with it
std::vector<Expr*> GPULower::getLoweredExprs() {
  FusionGuard fg(fusion_);

  validate(fusion_);

  // Initialize members of the class
  active_view = nullptr;
  active_view_axis = 0;

  replaceSizes();

  auto loop_nests = LoopNestGenerator::getLoopNest(fusion_);
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
