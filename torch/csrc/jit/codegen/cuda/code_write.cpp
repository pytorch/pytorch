#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iriostream.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <torch/csrc/jit/codegen/cuda/code_write.h>

namespace torch {
namespace jit {
namespace fuser {

// Flattens the indices from the for loops into a single
// vector. This useful for poassing to IndexCompute.
std::vector<Int*> CodeWrite::getLoopIndices() {
  std::vector<Int*> inds;
  for (auto loop : fors)
    inds.push_back(loop->index());
  return inds;
}

// Final indexing into a TensorView. This is computes the mapping
// from logical indexing into an N-D transformed TensorView back
// into it's original M-D form.
void CodeWrite::printIndexInto(
    std::vector<Int*> indices,
    const TensorView* const tv) {
  bool inpOrOut = FusionGuard::getCurFusion()->hasInput(tv) ||
      FusionGuard::getCurFusion()->hasOutput(tv);

  os << "[";
  // Linearize the indexing for inputs/outputs
  if (inpOrOut) {
    bool first_index = true;
    for (decltype(indices.size()) i{0}; i < indices.size(); i++) {
      if(!first_index)
        os << " + ";
      first_index = false;
      print_inline(indices[i]);
      os << " * T" << tv->name() << ".stride[" << i << "]";
    }
  } else {
    // If not an input or output, our dimensions are what ever the loop
    // structure is from getComputeAtAxis() out, minus thread/block bindings.
    // Only handle registers for now!

    // We may not print anything actually, we don't want to print * or +
    // assuming we've printed something
    bool first_index = true;

    for (decltype(fors.size()) i{tv->getComputeAtAxis()}; i < fors.size();
         i++) {
      if (fors[i]->range()->isThread())
        continue;

      if (!first_index)
        os << " + ";

      first_index = false;

      // Index
      print_inline(fors[i]->index());

      for (decltype(fors.size()) j{i + 1}; j < fors.size(); j++) {
        if (fors[j]->range()->isThread())
          continue;
        os << " * ";
        // Strides
        print_inline(fors[j]->range()->size());
      }
    }

    // If nothing was printed, throw out a 0. Ideally we could try reducing this
    // to a single register, but likely don't need to for now.
    if (first_index)
      os << 0;
  }
  os << "]";
}

// Prints TensorView producers. These are values consumed in
// a TensorView Expr. We use the consumer (left hand side of
// the =) to compute the indexing into the consumer.
void CodeWrite::handle(const TensorView* const tv) {
  TensorDomain* td = tv->domain();

  const TensorView* tv2 = tv;

  if (producer && consumer != nullptr) {
    // get new reference so replay inline doesn't change the original.
    TensorView* tv_ = tv->clone();
    tv_->resetView();
    TransformReplay::fullReplay(consumer, tv_);
    tv2 = tv_;
  } else if (producer) {
    TORCH_INTERNAL_ASSERT(
        false, "Could not find consumer for this producer in CodeWrite.");
  }
  os << "T" << tv->name();

  std::vector<Int*> indices =
      IndexCompute::computeIndices(tv2, getLoopIndices());

  printIndexInto(indices, tv);
}

// If the val provided is in overrides, prints the string
// provided in overrides instead of the val name. This is done for
// things like "threadIDx.x", or "T0.size[0]"
void CodeWrite::handle(const Val* const val) {
  if (*(val->getValType()) == ValType::TensorView)
    handle(static_cast<const TensorView* const>(val));
  else if (overrides.find(val) != overrides.end())
    os << overrides[val];
  else
    IRPrinter::handle(val);
}

// Uses the provided TensorView to compute a predicate and print it.
bool CodeWrite::print_predicate(const TensorView* const pred_tv) {
  std::vector<Int*> indices =
      IndexCompute::computeIndices(pred_tv, getLoopIndices());

  std::vector<Int*> preds =
      PredicateCompute::computePredicates(pred_tv, indices);

  if (preds.size() == 0)
    return false;

  bool first_pred = true;
  os << "if( ";
  for (decltype(preds.size()) i{0}; i < preds.size(); i++) {
    if (preds[i]->sameAs(new Int(1.0)))
      continue;
    if (!first_pred)
      os << " && ";

    print_inline(preds[i]);

    first_pred = false;
  }
  os << " ) {\n";
  ++indent_size;
  indent();
  return true;
}

// Prints the consumer of a TensorView Expr. This will update the base view,
// print a predicate, then print the TensorView.
bool CodeWrite::printConsumer(TensorView* tv) {
  updateView(tv);
  indent();

  // Print predicates, first need to find predicate.
  bool predicated = print_predicate(tv);

  handle(tv);
  os << "\n";
  indent();
  os << "  = ";

  consumer = tv;
  producer = true;

  return predicated;
}

// The UnaryOps captured here will have a TensorView as an output
void CodeWrite::handle(const UnaryOp* const uop) {
  if (!isTVOp(uop)) {
    if (print_inline_)
      IRPrinter::handle(uop);
    return;
  }

  bool predicated = printConsumer(static_cast<TensorView*>(uop->out()));

  if (auto inline_uop = inline_op_str(uop->type())) {
    os << inline_uop.value();
    handle(uop->in());
  } else {
    os << uop->type() << "(";
    handle(uop->in());
    os << ")";
  }

  consumer = nullptr;
  producer = false;

  os << ";\n";

  if (predicated) {
    --indent_size;
    indent();
    os << "}\n";
  }
}

// The BinaryOps captured here will have a TensorView as an output
void CodeWrite::handle(const BinaryOp* const bop) {
  if (!isTVOp(bop)) {
    if (print_inline_)
      IRPrinter::handle(bop);
    return;
  }

  bool predicated = printConsumer(static_cast<TensorView*>(bop->out()));

  if (auto inline_bop = inline_op_str(bop->type())) {
    handle(bop->lhs());
    os << "\n";
    indent();
    os << "  ";
    os << inline_bop.value() << " ";
    handle(bop->rhs());
  } else {
    os << bop->type() << "(";
    handle(bop->lhs());
    os << "\n";
    indent();
    os << ", ";
    handle(bop->rhs());
    os << ")";
  }

  consumer = nullptr;
  producer = false;

  os << ";\n";

  if (predicated) {
    --indent_size;
    indent();
    os << "}\n";
  }
}

void CodeWrite::indent() {
  for (int i = 0; i < indent_size; i++)
    os << "  ";
}

// Pop the inner most for loop
void CodeWrite::closeFor() {
  IterDomain* id = fors.back()->range();
  Val* iterator = fors.back()->index();
  fors.pop_back();
  // Clear overrides associated with this for loop
  if (id->parallel_method() != ParallelType::Serial) {
    auto it = overrides_find(iterator);
    if (it != overrides.end())
      overrides.erase(it);
    return;
  }

  indent_size--;
  indent();
  os << "}" << std::endl;
}

void CodeWrite::bind(IterDomain* id, Val* iterator) {
  if (id->isThread()) {
    std::stringstream ss;
    ss << id->parallel_method();
    overrides_emplace(iterator, ss.str());
  }

  if (id->parallel_method() == ParallelType::Vectorize ||
      id->parallel_method() == ParallelType::Unroll)
    TORCH_CHECK(
        false,
        "Unroll and Vectorize are not yet implemented for code generation.");
}

// Push Back a new for loop scope based on the IterDomain
void CodeWrite::openFor(IterDomain* id) {
  fors.push_back(new ForLoop(new Int(), id, {}));

  if (id->parallel_method() != ParallelType::Serial) {
    bind(id, fors.back()->index());
    return;
  }

  indent();
  indent_size++;

  os << "for( size_t ";
  handle(fors.back()->index());
  os << " = " << new Int(0) << "; ";
  handle(fors.back()->index());
  os << " < ";
  print_inline(id->size());
  os << "; ++";
  handle(fors.back()->index());
  os << " ) {" << std::endl;
}

// Clear out the last active computeAt view
void CodeWrite::clearActiveView() {
  active_view_axis = 0;
  active_view = nullptr;
}

// Pop back all for loops.
void CodeWrite::resetFors() {
  while (!fors.empty())
    closeFor();

  reset_fors = false;
  clearActiveView();
}

// Print register allocation of a TensorView
void CodeWrite::printAlloc(TensorView* tv) {
  if (FusionGuard::getCurFusion()->hasInput(tv) ||
      FusionGuard::getCurFusion()->hasOutput(tv))
    return;

  Int* size = new Int(1);
  for (auto i = tv->getComputeAtAxis(); i < tv->nDims(); i++) {
    size = static_cast<Int*>(mul(size, tv->axis(i)->size()));
  }

  indent();
  os << tv->getDataType().value() << " T" << tv->name() << "[";
  print_inline(size);
  os << "];" << std::endl;
}

// Update fors based on tv.
void CodeWrite::updateView(TensorView* tv) {
  // If previous tv flaged that fors need to be reset, clear them all
  if (reset_fors)
    resetFors();

  // Hit the final statment in a string of compute_at's. Need to close the for
  // loops down to the previous compute_at axis, then need to put my remaining
  // for loops on there. Also need to set reset_fors flag.

  // tv is not part of a computeAt structure, or it's the final tv in a
  // computeAt structure.
  if (!tv->hasComputeAt()) {
    // If we're the last computeAt of a block of computeAt TVs.
    if (active_view != nullptr && tv->sameAs(active_view)) {
      int depth = fors.size();
      // reduce down to previous active view_axis
      for (int i = active_view_axis; i < depth; i++)
        closeFor();
      // Remove the active view
      clearActiveView();
    } else {
      // I'm not the final computeAt of a block, I'm independent.
      // Reset the loop structure
      resetFors();
    }
    // All active fors have been closed
    printAlloc(tv);

    for (int i = fors.size(); i < tv->nDims(); i++)
      openFor(tv->getComputeAtAxis(i));
    reset_fors = true;

  } else {
    active_view_axis = tv->getComputeAtAxis();
    active_view = tv->getComputeAtView();

    int depth = fors.size();
    for (int i = active_view_axis; i < depth; i++)
      closeFor();
    // Allocate tv if not input/output
    printAlloc(tv);
    for (int i = fors.size(); i < tv->nDims(); i++)
      openFor(tv->getComputeAtAxis(i));
  }
}

// Check if we're a TensorView op that we can generate code for.
bool CodeWrite::isTVOp(const Expr* expr) {
  if (expr->nOutputs() == 1 &&
      expr->output(0)->getValType().value() == ValType::TensorView)
    if (expr->getExprType().value() == ExprType::BinaryOp ||
        expr->getExprType().value() == ExprType::UnaryOp)
      return true;
  return false;
}

// Setup the map from values to strings.
void CodeWrite::setupOverrides() {
  // Grab all the values used in the fusion (based on traversing
  // backwards from outputs)
  std::vector<Val*> used_vals = FindUsedVals::find();
  // If the value is a TensorView, we're going to grab it's root domain
  // and map the size used for the root domain to T.size[...]
  for (Val* val : used_vals) {
    if (val->getValType().value() == ValType::TensorView) {
      if(!(
        FusionGuard::getCurFusion()->hasInput(val)
      ||FusionGuard::getCurFusion()->hasOutput(val)
      )) continue;
      TensorView* tv = static_cast<TensorView*>(val);
      TensorDomain* td = tv->domain();
      if (tv->tensor() == nullptr)
        continue;

      TensorDomain* root = TransformIter::getRoot(tv->domain());
      for (decltype(root->size()) i{0}; i < root->size(); i++) {
        if (overrides_find(root->axis(i)->size()) == overrides.end()) {
          std::stringstream ss;
          ss << "T" << tv->name() << ".size[" << i << "]";
          overrides_emplace(root->axis(i)->size(), ss.str());
        }
      }
    }
  }
}

// Print the header for the kernel, the inputs/outputs
// TODO: Push this out to another class so we don't need dispatch implemented here
void CodeWrite::header() {
  os << "__global__ void " << kernel_name_ << "(";

  std::deque<Val*> vals;
  Fusion* fusion = FusionGuard::getCurFusion();
  for (decltype(fusion->nInputs()) i{0}; i < fusion->nInputs(); i++)
    vals.push_back(fusion->input(i));
  for (decltype(fusion->nOutputs()) i{0}; i < fusion->nOutputs(); i++)
    vals.push_back(fusion->output(i));

  for (Val* val : vals) {
    switch (val->getValType().value()) {
      case (ValType::TensorView):
        switch (val->getDataType().value()) {
          case (DataType::Float):
            os << "Tensor<float> T";
            break;
          case (DataType::Int):
            os << "Tensor<int> T";
            break;
          default:
            TORCH_CHECK(
                false,
                "CodeWrite::header() found an input to the fusion of unexpected val type.");
        }
        break;
      case (ValType::Scalar):
        switch (val->getDataType().value()) {
          case (DataType::Float):
            os << "float f";
            break;
          case (DataType::Int):
            os << "int i";
            break;
          default:
            TORCH_CHECK(
                false,
                "CodeWrite::header() found an input to the fusion of unexpected val type.");
        }
        break;
      default:
        TORCH_CHECK(
            false,
            "CodeWrite::header() found an input to the fusion of unexpected data type.");
    }

    os << val->name();
    if (val != vals.back())
      os << ", ";
  }

  os << "){\n";
  indent_size++;
}

// Traverse through the fusion and print CUDA code associated with it
void CodeWrite::traverse(Fusion* fusion) {
  FusionGuard fg(fusion);
  // reset state.
  producer = false;
  consumer = nullptr;

  fors = std::vector<const ForLoop*>();
  indent_size = 0;
  active_view = nullptr;
  active_view_axis = 0;
  reset_fors = false;

  std::set<IterDomain*> bound_iters;
  std::map<const Val* const, std::string> overrides;

  setupOverrides();
  // IterVisitor::traverse(fusion, from_outputs_only, breadth_first, val_types);
  std::vector<Expr*> exprs = FusionGuard::getCurFusion()->exprs(true);

  header();
  for (auto* expr : exprs)
    IRPrinter::handle(expr);
  resetFors();
  os << "}\n";
  indent_size--;
}

} // namespace fuser
} // namespace jit
} // namespace torch
