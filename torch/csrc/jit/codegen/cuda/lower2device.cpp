#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>
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

struct parentScope_ : private OptInDispatch {
 private:
  Expr* parent_ = nullptr;

  void handle(ForLoop* fl) final {
    parent_ = fl->parentScope();
  }

  void handle(IfThenElse* ite) final {
    parent_ = ite->parentScope();
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static Expr* parent(Expr* scope) {
    parentScope_ sp;
    sp.handle(scope);
    return sp.parent_;
  }
};

struct forLoopCount : private OptInDispatch {
 private:
  unsigned int count_ = 0;

  void handle(ForLoop* fl) final {
    count_++;
  }

  void handle(IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static unsigned int count(Expr* scope) {
    forLoopCount flc;
    Expr* it = scope;
    while (it != nullptr) {
      flc.handle(it);
      it = parentScope_::parent(it);
    }
    return flc.count_;
  }
};

struct scopePushBack : private OptInDispatch {
 private:
  Expr* _expr = nullptr;
  void handle(ForLoop* fl) final {
    fl->body().push_back(_expr);
  }

  void handle(IfThenElse* ite) final {
    ite->body().push_back(_expr);
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static void pushBack(Expr* scope, Expr* expr) {
    scopePushBack pb;
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    pb._expr = expr;
    pb.handle(scope);
  }
};

struct forLoopIndices : private OptInDispatch {
 private:
  std::vector<Val*> inds_;
  void handle(ForLoop* fl) final {
    inds_.insert(inds_.begin(), fl->index());
  }

  void handle(IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static std::vector<Val*> get(Expr* scope) {
    forLoopIndices fli;
    Expr* it = scope;
    while (it != nullptr) {
      fli.handle(it);
      it = parentScope_::parent(it);
    }
    return fli.inds_;
  }
};

struct forLoopIDs : private OptInDispatch {
 private:
  std::vector<IterDomain*> IDs_;
  void handle(ForLoop* fl) final {
    IDs_.insert(IDs_.begin(), fl->range());
  }

  void handle(IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static std::vector<IterDomain*> get(Expr* scope) {
    forLoopIDs fli;
    Expr* it = scope;
    while (it != nullptr) {
      fli.handle(it);
      it = parentScope_::parent(it);
    }
    return fli.IDs_;
  }
};

} // namespace
// END HELPER FUNCTIONS

// Open a new inner most for loop
void GPULower::openFor(IterDomain* id) {
  ForLoop* new_scope = nullptr;
  if (id->isThread()) {
    new_scope = new ForLoop(
        new NamedScalar(stringify(id->parallel_method()), DataType::Int),
        id,
        {},
        active_scope);
  } else {
    new_scope = new ForLoop(new Int(), id, {}, active_scope);
  }
  pushBack(new_scope);
  active_scope = new_scope;
}

// Close the inner most scope
void GPULower::closeScope() {
  TORCH_INTERNAL_ASSERT(
      active_scope != nullptr,
      "Tried to close the active scope, but there isn't one set.");
  Expr* parent = parentScope_::parent(active_scope);
  active_scope = parent;
}

// Close all scopes
void GPULower::resetScope() {
  active_scope = nullptr;
}

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

std::vector<Val*> GPULower::getLoopIndices() {
  return forLoopIndices::get(active_scope);
}

std::vector<IterDomain*> GPULower::getLoopIterDomains() {
  return forLoopIDs::get(active_scope);
}

TensorIndex* GPULower::getGlobalProducerIndex(
    TensorView* producer,
    TensorView* consumer) {
  // Get new reference so replay inline doesn't change the original.
  TensorView* cloned_tv = producer->clone();
  // This replay will ignore reduction dimensions on the producer
  TransformReplay::fullReplay(consumer, cloned_tv);
  TORCH_INTERNAL_ASSERT(
      getLoopIndices().size() == cloned_tv->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  const std::vector<Val*> computed_inds =
      IndexCompute::computeIndices(cloned_tv, getLoopIndices());

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
      computeForDepth() == producer->nDims(),
      "Expected a tensor with ",
      computeForDepth(),
      " dimensions but got one with ",
      producer->nDims());

  std::vector<Val*> loopInds = getLoopIndices();
  std::vector<IterDomain*> ranges = getLoopIterDomains();
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
      ind = mul(ind, used_ranges[i]->size());
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
      getLoopIndices().size() == consumer->nDims(),
      "Dimensionality error in code generator while computing indexing.");

  const std::vector<Val*> computed_inds =
      IndexCompute::computeIndices(consumer, getLoopIndices());

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
      computeForDepth() == consumer->nDims(),
      "Expected a tensor with ",
      computeForDepth(),
      " dimensions but got one with ",
      consumer->nDims());

  std::vector<Val*> loopInds = getLoopIndices();
  std::vector<IterDomain*> ranges = getLoopIterDomains();
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
      ind = mul(ind, used_ranges[i]->size());
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

// Track how far our for loop scope is
unsigned int GPULower::computeForDepth() {
  return forLoopCount::count(active_scope);
}

// Push an expr to the active scope
void GPULower::pushBack(Expr* expr) {
  if (active_scope == nullptr) {
    lowered_exprs.push_back(expr);
    return;
  }
  scopePushBack::pushBack(active_scope, expr);
}

// Return the parent of the active scope
Expr* GPULower::parentScope() {
  if (active_scope == nullptr)
    return nullptr;
  return parentScope_::parent(active_scope);
}

Allocate* GPULower::getAlloc(TensorView* tv) {
  TORCH_INTERNAL_ASSERT(
      !(FusionGuard::getCurFusion()->hasInput(tv) ||
        FusionGuard::getCurFusion()->hasOutput(tv)),
      "Tried to allocate an input or output tensor.");

  std::vector<Val*> alloc_dims;

  for (decltype(tv->nDims()) i = tv->getComputeAtAxis(); i < tv->nDims(); i++) {
    IterDomain* dim = tv->getComputeAtAxis(i);
    if (dim->isThreadDim())
      continue;
    alloc_dims.push_back(dim->size());
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

IfThenElse* GPULower::getPredicate(const TensorView* const pred_tv) {
  TensorIndex* ti = new TensorIndex(
      pred_tv, IndexCompute::computeIndices(pred_tv, getLoopIndices()));

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

// Custom dispatch for Expr, want to find out of it's a TV op
void GPULower::handle(Expr* expr) {
  if (!isTVOp(expr))
    return;

  TensorView* out = static_cast<TensorView*>(expr->output(0));

  updateView(out);

  // 8) Run operation
  OptOutDispatch::handle(expr);

  // 9) Close predicate
  if (active_scope != nullptr &&
      active_scope->getExprType() == ExprType::IfThenElse)
    closeScope();
}

void GPULower::handle(UnaryOp* uop) {
  TORCH_INTERNAL_ASSERT(
      isTV(uop->out()),
      "Expected a tensor view but got ",
      uop->out()->getValType().value());
  TensorIndex* out = getConsumerIndex(asTV(uop->out()));
  Val* in = uop->in();
  if (isTV(in))
    in = getProducerIndex(asTV(in), asTV(uop->out()));
  pushBack(new UnaryOp(uop->getUnaryOpType(), out, in));
}

void GPULower::handle(BinaryOp* bop) {
  TORCH_INTERNAL_ASSERT(
      isTV(bop->out()),
      "Expected a tensor view but got ",
      bop->out()->getValType().value());
  TensorIndex* out = getConsumerIndex(asTV(bop->out()));
  Val* lhs = bop->lhs();
  Val* rhs = bop->rhs();

  if (isTV(lhs))
    lhs = getProducerIndex(asTV(lhs), asTV(bop->out()));

  if (isTV(rhs))
    rhs = getProducerIndex(asTV(rhs), asTV(bop->out()));

  pushBack(new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs));
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
void GPULower::updateView(TensorView* tv) {
  // 1) Reduce loop structure
  if (active_view == nullptr) {
    // - Reset all loops if active_view == nullptr (I'm not the last in a series
    // of computeAts)
    resetScope();
  } else {
    // - Else reduce to active_view_axis if loop_depth > active_view_axis
    auto depth = computeForDepth();
    for (auto i = depth; i > active_view_axis; i--) {
      closeScope();
    }
  }
  if (tv->hasComputeAt()) {
    //  2) Set active_view(_axis)
    //    - If there is a computeAt set for this TV
    setActiveView(tv);

    //  3) Open to compute At
    //    - If there is a computeAt set for this TV
    auto depth = computeForDepth();
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
  for (decltype(tv->nDims()) i = computeForDepth(); i < tv->nDims(); i++)
    openFor(tv->getComputeAtAxis(i));

  // 7) Open predicate
  IfThenElse* pred = getPredicate(tv);
  if (!pred->cond()->sameAs(new Int(1))) {
    pushBack(pred);
    active_scope = pred;
  }
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
      Val* orig_size = root_td->axis(i)->size();
      std::stringstream ss;
      ss << "T" << new_tv->name() << ".size[" << i << "]";
      Val* new_size =
          new NamedScalar(ss.str(), orig_size->getDataType().value());
      size_map[orig_size] = new_size;

      new_domain.push_back(new IterDomain(
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
      Val* new_size = root_td->axis(i)->size();
      if (size_map.find(new_size) != size_map.end())
        new_size = size_map[new_size];
      new_domain.push_back(new IterDomain(
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

// Traverse through the fusion and print CUDA code associated with it
std::vector<Expr*> GPULower::getLoweredExprs() {
  FusionGuard fg(fusion_);

  TORCH_CHECK(
      !fusion_->lowered,
      "Fusions can only be lowered once as of now. You could reuse the lowering using",
      " std::vector<Expr*> GPULower::getLoweredExprs() the result can be printed as",
      " a kernel with   IRPrinter irp(os); irp.printKernel(lowered_exprs, kernel_name);");

  // Initialize members of the class
  lowered_exprs = std::vector<Expr*>();
  active_view = nullptr;
  active_view_axis = 0;

  replaceSizes();

  // Run through and lower the expressions
  std::vector<Expr*> exprs = fusion_->exprs(true);
  for (auto* expr : exprs)
    handle(expr);

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