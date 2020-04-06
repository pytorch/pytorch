#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {

// START HELPER FUNCTIONS
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
// END HELPER FUNCTIONS

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
      computed_inds.size() == producer->getRootDomain()->size(),
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
  for (decltype(loopInds.size()) i{0}; i < loopInds.size(); i++) {
    if (producer->hasComputeAt() && i < producer->getComputeAtAxis())
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
      computed_inds.size() == consumer->getRootDomain()->size(),
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

  for (decltype(loopInds.size()) i{0}; i < loopInds.size(); i++) {
    if (i < consumer->getComputeAtAxis())
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

IfThenElse* GPULower::getPredicate(const TensorView* const pred_tv) {
  TensorIndex* ti = new TensorIndex(
      pred_tv,
      IndexCompute::computeIndices(
          pred_tv, scope_utils::getLoopIndices(active_scope)));

  std::vector<Int*> all_preds = PredicateCompute::computePredicates(ti);

  std::vector<Int*> preds;

  Int* one = new Int(1);

  for (Int* pred : all_preds)
    if (!pred->sameAs(one))
      preds.push_back(pred);

  if (preds.size() == 0) {
    return new IfThenElse(one, {}, {}, active_scope);
  }

  Int* cond = preds[0];

  for (decltype(preds.size()) i{1}; i < preds.size(); i++)
    cond = static_cast<Int*>(andOp(cond, preds[i]));

  return new IfThenElse(cond, {}, {}, active_scope);
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

Statement* GPULower::mutate(ForLoop* fl) {
  Expr* prev_scope = active_scope;
  active_scope = fl;
  std::vector<Expr*> mutated_exprs;
  bool is_mutated = false;
  for (auto expr : fl->body().exprs()) {
    Statement* mutated_stmt = mutate(expr);

    TORCH_INTERNAL_ASSERT(
        mutated_stmt->isExpr(),
        "Tried to generate a kernel but hit a non expression during lowering: ",
        mutated_stmt);

    mutated_exprs.push_back(static_cast<Expr*>(mutated_stmt));
    if (!(mutated_exprs.back()->sameAs(expr)))
      is_mutated = true;
  }

  if (is_mutated) {
    scope_utils::clearScope(active_scope);
    for (auto expr : mutated_exprs)
      pushBack(expr);
  }

  active_scope = prev_scope;

  if (is_mutated)
    return new ForLoop(
        fl->index(), fl->iter_domain(), mutated_exprs, fl->parentScope());

  return fl;
}

Statement* GPULower::mutate(UnaryOp* uop) {
  if (!isTVOp(uop))
    return OptOutMutator::mutate(uop);

  IfThenElse* pred = getPredicate(asTV(uop->out()));
  bool predicated = !pred->cond()->sameAs(new Int(1));
  if (predicated) {
    pushBack(pred);
    active_scope = pred;
  }

  TensorIndex* out = getConsumerIndex(asTV(uop->out()));
  Val* in = uop->in();
  if (isTV(in))
    in = getProducerIndex(asTV(in), asTV(uop->out()));
  Expr* new_op = new UnaryOp(uop->getUnaryOpType(), out, in);

  if (predicated) {
    active_scope = scope_utils::getParent(active_scope);
    pushBack(new_op);
    return pred;
  }

  return new_op;
}

Statement* GPULower::mutate(BinaryOp* bop) {
  if (!isTVOp(bop))
    return OptOutMutator::mutate(bop);

  IfThenElse* pred = getPredicate(asTV(bop->out()));
  bool predicated = !pred->cond()->sameAs(new Int(1));
  if (predicated) {
    pushBack(pred);
    active_scope = pred;
  }

  TensorIndex* out = getConsumerIndex(asTV(bop->out()));
  Val* lhs = bop->lhs();
  Val* rhs = bop->rhs();

  if (isTV(lhs))
    lhs = getProducerIndex(asTV(lhs), asTV(bop->out()));

  if (isTV(rhs))
    rhs = getProducerIndex(asTV(rhs), asTV(bop->out()));

  Expr* new_op = new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs);

  if (predicated) {
    pushBack(new_op);
    active_scope = scope_utils::getParent(active_scope);
    return pred;
  }

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
  // Replacement of full tensor views
  std::unordered_map<Val*, Val*> tv_map;

  // Grab inputs and outputs
  std::vector<TensorView*> orig_inp_out;
  std::vector<TensorView*> orig_intermediates;

  for (auto* val : fusion->deterministic_vals()) {
    if (isTV(val)) {
      if (fusion->hasInput(val) || fusion->hasOutput(val)) {
        orig_inp_out.push_back(asTV(val));
      } else {
        orig_intermediates.push_back(asTV(val));
      }
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
    TensorView* new_tv =
        new TensorView(tv->domain(), tv->getDataType().value());

    // We can place the new_tv in the map right away.
    tv_map[tv] = new_tv;

    // Replace the domain with one based on Ti.size[j]
    std::vector<IterDomain*> new_domain;
    TensorDomain* root_td = tv->getRootDomain();
    for (decltype(root_td->size()) i{0}; i < root_td->size(); i++) {
      Val* orig_size = root_td->axis(i)->extent();
      std::stringstream ss;
      ss << "T" << new_tv->name() << ".size[" << i << "]";
      Val* new_size =
          new NamedScalar(ss.str(), orig_size->getDataType().value());
      size_map[orig_size] = new_size;

      new_domain.push_back(new IterDomain(
          root_td->axis(i)->start(),
          new_size,
          root_td->axis(i)->parallel_method(),
          root_td->axis(i)->isReduction()));
    }
    new_tv->setDomain(new TensorDomain(new_domain));
  }

  for (TensorView* tv : orig_intermediates) {
    TensorView* new_tv =
        new TensorView(tv->domain(), tv->getDataType().value());
    tv_map[tv] = new_tv;

    std::vector<IterDomain*> new_domain;
    TensorDomain* root_td = tv->getRootDomain();

    for (decltype(root_td->size()) i{0}; i < root_td->size(); i++) {
      Val* new_size = root_td->axis(i)->extent();
      if (size_map.find(new_size) != size_map.end())
        new_size = size_map[new_size];
      new_domain.push_back(new IterDomain(
          root_td->axis(i)->start(),
          new_size,
          root_td->axis(i)->parallel_method(),
          root_td->axis(i)->isReduction()));
    }
    new_tv->setDomain(new TensorDomain(new_domain));
  }

  // Now that we have the base tensor views. Lets fix its members.
  for (auto entry : tv_map) {
    TensorView* orig_tv = asTV(entry.first);
    TensorView* new_tv = asTV(entry.second);

    // Domain in the new TV is the root domain, replay it like the original
    // domain.
    TransformReplay::fullReplay(orig_tv, new_tv);

    // Parallelize all iter domains
    for (decltype(new_tv->domain()->size()) i{0}; i < new_tv->domain()->size();
         i++)
      new_tv->axis(i)->parallelize(orig_tv->axis(i)->parallel_method());

    // Set compute at view and axis
    TensorView* computeAtTV = orig_tv->getComputeAtView();
    if (computeAtTV != nullptr) {
      TORCH_INTERNAL_ASSERT(
          tv_map.find(computeAtTV) != tv_map.end(),
          "Expected to find a translation for ",
          computeAtTV,
          " but one wasn't found.");
      new_tv->setComputeAt(
          asTV(tv_map[computeAtTV]), (int)(orig_tv->getComputeAtAxis()));
    }
  }

  ReplaceAll::instancesOf(tv_map);
}

namespace {

// Some pre-compilation checks
void validate(Fusion* fusion) {
  for (Val* val : fusion->vals()) {
    if (isTV(val)) {
      TensorView* tv = asTV(val);
      for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
        IterDomain* id = tv->getComputeAtAxis(i);

        if (id->isThread())
          TORCH_CHECK(
              !id->isReduction(),
              "Parallelization on reduction axes not support at the moment found on, ",
              tv,
              ".");

        if (tv->hasComputeAt())
          if (i < tv->getComputeAtAxis())
            TORCH_CHECK(
                id->parallel_method() != ParallelType::Unroll,
                "Unroll dimension cannot be outside computeAt, found on: ",
                tv,
                " compute at ",
                tv->getComputeAtView(),
                " axis = ",
                tv->getComputeAtAxis(),
                ".");
      }
    } // if isTV
  } // for(Val* val : fusion->vals())

} // validate
} // namespace

// Traverse through the fusion and print CUDA code associated with it
std::vector<Expr*> GPULower::getLoweredExprs() {
  FusionGuard fg(fusion_);

  // Likely we lowered this fusion, we can simply return the lowered expressions
  // Not the safest approach but good enough for now.
  if (fusion_->lowered && lowered_exprs.size() != 0)
    return lowered_exprs;

  TORCH_CHECK(
      !fusion_->lowered,
      "Fusions can only be lowered once as of now. You could reuse the lowering using",
      " std::vector<Expr*> GPULower::getLoweredExprs() the result can be printed as",
      " a kernel with   IRPrinter irp(os); irp.printKernel(lowered_exprs, kernel_name);");

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

  fusion_->lowered = true;
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