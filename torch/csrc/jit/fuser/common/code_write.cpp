#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/transform_replay.h>
#include <torch/csrc/jit/fuser/common/type.h>

#include <torch/csrc/jit/fuser/common/code_write.h>

namespace torch {
namespace jit {
namespace fuser {

std::vector<Int*> CodeWrite::getLoopIndices() {
  std::vector<Int*> inds;
  for (auto pair : fors)
    inds.push_back(pair.first);
  return inds;
}

void CodeWrite::printIndexInto(std::vector<Int*> indices, const TensorView* const tv){
   
  TensorDomain* root = TransformReplay::getRoot(tv->domain());

  bool inpOrOut
     = FusionGuard::getCurFusion()->isInput(tv)
    || FusionGuard::getCurFusion()->isOutput(tv);

  os << "[";
  for (decltype(indices.size()) i{0}; i < indices.size(); i++) {
    print_inline(indices[i]);
    os << " * ";
    if(inpOrOut)
      os << "TV" << tv->name() << "->stride(" << i << ")";
    else
      print_inline(root->axis(i)->size());

    if (i != (indices.size() - 1) )
      os << " + ";
  }
  os << "]";
}

void CodeWrite::print(const TensorView* const tv) {
  TensorDomain* td = tv->domain();

  const TensorView* tv2 = tv;

  if (producer && consumer != nullptr) {
    // get new reference so replay inline doesn't change the original.
    TensorView* tv_ = tv->clone();
    tv_->resetView();
    TransformReplay::fullReplay(consumer, tv_);
    tv2 = tv_;
  } else if (producer) {
    throw std::runtime_error(
        "Could not find consumer for producer in CodeWrite.");
  }
  os << "TV" << tv2->name();

  std::vector<Int*> indices =
    IndexCompute::computeIndices(tv2, getLoopIndices());

  printIndexInto(indices, tv);
}

void CodeWrite::print(const Val* const val) {
  if (*(val->getValType()) == ValType::TensorView)
    print(static_cast<const TensorView* const>(val));
  else if (overrides.find(val) != overrides.end())
    os << overrides[val];
  else
    Printer::print(val);
}

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
    if (preds[i]->same_as(new Int(1.0)))
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

bool CodeWrite::print_lhs(TensorView* tv) {
  updateView(tv);
  indent();

  // Print predicates, first need to find predicate.
  bool predicated = print_predicate(tv);

  print(tv);
  os << " = ";

  consumer = tv;
  producer = true;

  return predicated;
}

// Already filtered so output is a TensorView
void CodeWrite::print(const UnaryOp* const uop) {
  if (!isTVOp(uop)) {
    if (print_inline_)
      Printer::print(uop);
    return;
  }

  bool predicated = print_lhs(static_cast<TensorView*>(uop->out()));

  if (auto inline_uop = inline_op_str(uop->type())) {
    os << inline_uop.value();
    print(uop->in());
  } else {
    os << uop->type() << "(";
    print(uop->in());
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

void CodeWrite::print(const BinaryOp* const bop) {
  if (!isTVOp(bop)) {
    if (print_inline_)
      Printer::print(bop);
    return;
  }

  bool predicated = print_lhs(static_cast<TensorView*>(bop->out()));

  if (auto inline_bop = inline_op_str(bop->type())) {
    print(bop->lhs());
    os << " " << inline_bop.value() << " ";
    print(bop->rhs());
  } else {
    os << bop->type() << "(";
    print(bop->lhs());
    os << ", ";
    print(bop->rhs());
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

void CodeWrite::closeFor() {
  IterDomain* id = fors.back().second;
  Val* iterator = fors.back().first;
  fors.pop_back();
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
  switch (id->parallel_method()) {
    case (ParallelType::BIDz):
      overrides_emplace(iterator, "blockIdx.z");
      bound_iters.emplace(id);
      break;
    case (ParallelType::BIDy):
      overrides_emplace(iterator, "blockIdx.y");
      bound_iters.emplace(id);
      break;
    case (ParallelType::BIDx):
      overrides_emplace(iterator, "blockIdx.x");
      bound_iters.emplace(id);
      break;
    case (ParallelType::TIDz):
      overrides_emplace(iterator, "threadIdx.z");
      bound_iters.emplace(id);
      break;
    case (ParallelType::TIDy):
      overrides_emplace(iterator, "threadIdx.y");
      bound_iters.emplace(id);
      break;
    case (ParallelType::TIDx):
      overrides_emplace(iterator, "threadIdx.x");
      bound_iters.emplace(id);
      break;
    case (ParallelType::Vectorize):
    case (ParallelType::Unroll):
      throw std::runtime_error(
          "Unroll and Vectorize are not yet implemented for code generation.");
    case (ParallelType::Serial):
      break;
  }
}

void CodeWrite::openFor(IterDomain* id) {
  fors.push_back({new Int(), id});

  if (id->parallel_method() != ParallelType::Serial) {
    bind(id, fors.back().first);
    return;
  }

  indent();
  indent_size++;

  os << "for( size_t ";
  print(fors.back().first);
  os << " = 0; ";
  print(fors.back().first);
  os << " < ";
  print_inline(id->size());
  os << "; ++";
  print(fors.back().first);
  os << " ) {" << std::endl;
}

void CodeWrite::clearActiveView() {
  active_view_axis = 0;
  active_view = nullptr;
}

void CodeWrite::resetFors() {
  while (!fors.empty())
    closeFor();

  reset_fors = false;
  clearActiveView();
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
    if (active_view != nullptr && tv->same_as(active_view)) {
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
    for (int i = fors.size(); i < tv->domain()->size(); i++)
      openFor(tv->getAxis(i));
    reset_fors = true;
  } else {
    active_view_axis = tv->getComputeAtAxis();
    active_view = tv->getComputeAtView();

    int depth = fors.size();
    for (int i = active_view_axis; i < depth; i++)
      closeFor();
    for (int i = fors.size(); i < tv->domain()->size(); i++)
      openFor(tv->getAxis(i));
  }
}

bool CodeWrite::isTVOp(const Expr* expr) {
  if (expr->nOutputs() == 1 &&
      expr->output(0)->getValType().value() == ValType::TensorView)
    if (expr->getExprType().value() == ExprType::BinaryOp ||
        expr->getExprType().value() == ExprType::UnaryOp)
      return true;
  return false;
}

void CodeWrite::setupOverrides() {
  std::set<Val*> used_vals = FindUsedVals::find();
  for (Val* val : used_vals) {
    if (val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);
      TensorDomain* td = tv->domain();
      if (tv->tensor() == nullptr)
        continue;

      TensorDomain* root = TransformIter::getRoot(tv->domain());
      for (decltype(root->size()) i{0}; i < root->size(); i++) {
        if (overrides_find(root->axis(i)->size()) == overrides.end()) {
          std::stringstream ss;
          ss << "TV" << tv->name() << "->size(" << i << ")";
          overrides_emplace(root->axis(i)->size(), ss.str());
        }
      }
    }
  }
}

void CodeWrite::header() {
  os << "__global__ void kernel(";

  std::deque<Val*> vals;
  Fusion* fusion = FusionGuard::getCurFusion();
  for (decltype(fusion->nInputs()) i{0}; i < fusion->nInputs(); i++)
    vals.push_back(fusion->input(i));
  for (decltype(fusion->nOutputs()) i{0}; i < fusion->nOutputs(); i++)
    vals.push_back(fusion->output(i));

  for (Val* val : vals) {
    switch (val->getValType().value()) {
      case (ValType::TensorView):
        os << "TensorView* TV";
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
            throw std::runtime_error(
                "Could not figure out data type for CodeWrite::header().");
        }
      default:
        throw std::runtime_error(
            "Could not figure out value type for CodeWrite::header().");
    }

    os << val->name();
    if (val != vals.back())
      os << ", ";
  }

  os << "){\n";
  indent_size++;
}

void CodeWrite::traverse(
    Fusion* fusion,
    bool from_outputs_only,
    bool breadth_first,
    std::unordered_set<ValType> val_types) {
  FusionGuard fg(fusion);
  // reset state.
  producer = false;
  consumer = nullptr;

  fors = std::vector<std::pair<Int*, IterDomain*>>();
  indent_size = 0;
  active_view = nullptr;
  active_view_axis = 0;
  reset_fors = false;

  std::set<IterDomain*> bound_iters;
  std::map<const Val* const, std::string> overrides;

  setupOverrides();
  // IterVisitor::traverse(fusion, from_outputs_only, breadth_first, val_types);
  std::vector<Expr*> exprs = FusionGuard::getCurFusion()->exprs();

  header();
  for (auto* expr : exprs)
    Printer::print(expr);
  resetFors();
  os << "}\n";
  indent_size--;
}

} // namespace fuser
} // namespace jit
} // namespace torch