#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void OptOutMutator::mutate(Statement* s) {
  Statement::mutatorDispatch(this, s);
}

void OptOutMutator::mutate(Expr* e) {
  Expr::mutatorDispatch(this, e);
}

void OptOutMutator::mutate(Val* v) {
  Val::mutatorDispatch(this, v);
}

void OptOutMutator::registerMutation(Val* val, Val* mutation) {
  bool val_is_ns = val->vtype() == ValType::NamedScalar;
  bool mutation_is_ns = mutation->vtype() == ValType::NamedScalar;
  bool val_is_scalar = val->vtype() == ValType::Scalar;
  bool mutation_is_scalar = mutation->vtype() == ValType::Scalar;
  TORCH_INTERNAL_ASSERT(
      mutation->dtype() == val->dtype() &&
          (mutation->vtype() == val->vtype() ||
           ((val_is_ns && mutation_is_scalar) ||
            (mutation_is_ns && val_is_scalar))),
      "Mutations are not allowed to change types, tried to go from: (",
      val->vtype(),
      ", ",
      val->dtype(),
      ") to: (",
      mutation->vtype(),
      ", ",
      mutation->dtype(),
      ")");
  mutations[val] = mutation;
}

void OptOutMutator::mutate(Bool* b) {}

void OptOutMutator::mutate(Double* d) {}

void OptOutMutator::mutate(Int* i) {}

void OptOutMutator::mutate(ComplexDouble* c) {}

void OptOutMutator::mutate(NamedScalar* ns) {}

void OptOutMutator::mutate(IterDomain* id) {
  Val* start = maybeMutated(id->start());
  Val* extent = maybeMutated(id->extent());
  Val* expanded_extent = nullptr;
  if (id->hasExpandedExtent()) {
    expanded_extent = maybeMutated(id->expandedExtent());
  }
  Val* stop_offset = maybeMutated(id->stopOffset());
  if (start->sameAs(id->start()) && extent->sameAs(id->extent()) &&
      (!id->hasExpandedExtent() ||
       expanded_extent->sameAs(id->expandedExtent())) &&
      stop_offset->sameAs(id->stopOffset())) {
    return;
  }
  registerMutation(
      id,
      IterDomainBuilder(id)
          .start(start)
          .extent(extent)
          .stop_offset(stop_offset)
          .expanded_extent(expanded_extent)
          .build());
}

void OptOutMutator::mutate(TensorDomain* td) {
  bool mutated = false;

  auto updateIdVec = [&](const std::vector<IterDomain*>& ids) {
    std::vector<IterDomain*> updated_ids;
    for (auto id : ids) {
      auto updated_id = maybeMutated(id)->as<IterDomain>();
      updated_ids.push_back(updated_id);
      if (!updated_id->sameAs(id)) {
        mutated = true;
      }
    }
    return updated_ids;
  };

  std::vector<IterDomain*> root_dom = updateIdVec(td->getRootDomain());
  std::vector<IterDomain*> rfactor_dom = td->hasRFactor()
      ? updateIdVec(td->getMaybeRFactorDomain())
      : std::vector<IterDomain*>();
  std::vector<IterDomain*> domain = updateIdVec(td->domain());

  if (!mutated) {
    return;
  }

  Val* mutated_val = IrBuilder::create<TensorDomain>(
      td->container(), root_dom, rfactor_dom, domain, td->contiguity());
  registerMutation(td, mutated_val);
}

void OptOutMutator::mutate(TensorView* tv) {
  TensorDomain* td = maybeMutated(tv->domain())->as<TensorDomain>();
  if (!tv->domain()->sameAs(td)) {
    tv->setDomain(td);
  }
  // Don't register tv mutations as we just want to update the TD
}

void OptOutMutator::mutate(kir::Predicate*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}

void OptOutMutator::mutate(kir::TensorIndex*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}

void OptOutMutator::mutate(FullOp* fop) {
  Val* out = maybeMutated(fop->output(0));
  Val* fill_value = maybeMutated(fop->getFillValue());

  if (out->sameAs(fop->output(0))) {
    return;
  }
  auto container = fop->container();
  container->removeExpr(fop);
  IrBuilder::create<FullOp>(container, out, fill_value, fop->dtype());
}

void OptOutMutator::mutate(SelectOp* sop) {
  Val* out = maybeMutated(sop->output(0));
  Val* in = maybeMutated(sop->input(0));
  Val* index = maybeMutated(sop->input(1));
  IterDomain* select_axis =
      maybeMutated(sop->getSelectAxis())->as<IterDomain>();

  if (out->sameAs(sop->output(0)) && in->sameAs(sop->output(0)) &&
      index->sameAs(sop->output(1)) &&
      select_axis->sameAs(sop->getSelectAxis())) {
    return;
  }
  auto container = sop->container();
  container->removeExpr(sop);
  IrBuilder::create<SelectOp>(container, out, in, select_axis, index);
}

void OptOutMutator::mutate(ARangeOp* aop) {
  Val* out = maybeMutated(aop->output(0));

  if (out->sameAs(aop->output(0))) {
    return;
  }
  auto container = aop->container();
  container->removeExpr(aop);
  IrBuilder::create<ARangeOp>(
      container,
      out,
      aop->start(),
      aop->end(),
      aop->step(),
      aop->dtype(),
      aop->getLinearLogicalIndex());
}

void OptOutMutator::mutate(EyeOp* eop) {
  Val* out = maybeMutated(eop->output(0));

  if (out->sameAs(eop->output(0))) {
    return;
  }
  auto container = eop->container();
  container->removeExpr(eop);
  IrBuilder::create<EyeOp>(
      container, out, eop->dtype(), eop->getIndex1(), eop->getIndex2());
}

void OptOutMutator::mutate(UnaryOp* uop) {
  Val* out = maybeMutated(uop->out());
  Val* in = maybeMutated(uop->in());

  if (out->sameAs(uop->out()) && in->sameAs(uop->in())) {
    return;
  }
  auto container = uop->container();
  auto uop_type = uop->getUnaryOpType();
  container->removeExpr(uop);
  IrBuilder::create<UnaryOp>(container, uop_type, out, in);
}

void OptOutMutator::mutate(BinaryOp* bop) {
  Val* out = maybeMutated(bop->out());
  Val* lhs = maybeMutated(bop->lhs());
  Val* rhs = maybeMutated(bop->rhs());

  if (out->sameAs(bop->out()) && lhs->sameAs(bop->lhs()) &&
      rhs->sameAs(bop->rhs())) {
    return;
  }

  auto container = bop->container();
  auto bop_type = bop->getBinaryOpType();
  container->removeExpr(bop);
  IrBuilder::create<BinaryOp>(container, bop_type, out, lhs, rhs);
}

void OptOutMutator::mutate(TernaryOp* top) {
  Val* out = maybeMutated(top->out());
  Val* in1 = maybeMutated(top->in1());
  Val* in2 = maybeMutated(top->in2());
  Val* in3 = maybeMutated(top->in3());

  if (out->sameAs(top->out()) && in1->sameAs(top->in1()) &&
      in2->sameAs(top->in2()) && in3->sameAs(top->in3())) {
    return;
  }

  auto container = top->container();
  auto top_type = top->getTernaryOpType();
  container->removeExpr(top);
  IrBuilder::create<TernaryOp>(container, top_type, out, in1, in2, in3);
}

void OptOutMutator::mutate(RNGOp* rop) {
  Val* out = maybeMutated(rop->output(0));
  Val* philox_idx = maybeMutated(rop->getPhiloxIndex());

  auto& parameters = rop->getParameters();
  std::vector<Val*> mutated_parameters;
  bool all_mutated_same = true;
  for (auto v : parameters) {
    mutated_parameters.emplace_back(maybeMutated(v));
    all_mutated_same = all_mutated_same && mutated_parameters.back()->sameAs(v);
  }

  if (out->sameAs(rop->output(0)) &&
      ((philox_idx == nullptr && rop->getPhiloxIndex() == nullptr) ||
       philox_idx->sameAs(rop->getPhiloxIndex())) &&
      all_mutated_same) {
    return;
  }

  auto container = rop->container();
  auto rop_type = rop->getRNGOpType();
  container->removeExpr(rop);
  IrBuilder::create<RNGOp>(
      container,
      rop_type,
      out,
      rop->dtype(),
      mutated_parameters,
      rop->getRNGOffset(),
      philox_idx);
}

void OptOutMutator::mutate(ReductionOp* rop) {
  Val* out = maybeMutated(rop->out());
  Val* in = maybeMutated(rop->in());
  Val* init = rop->init();
  if (out->sameAs(rop->out()) && in->sameAs(rop->in()) &&
      init->sameAs(rop->init())) {
    return;
  }

  auto container = rop->container();
  auto rop_type = rop->getReductionOpType();
  container->removeExpr(rop);
  IrBuilder::create<ReductionOp>(
      container, rop_type, init, out, in, rop->isAllreduce());
}

void OptOutMutator::mutate(GroupedReductionOp* rop) {
  bool is_same = true;

  std::vector<Val*> outputs;
  for (auto out : rop->outputs()) {
    auto maybe_mutated = maybeMutated(out);
    is_same = is_same && maybe_mutated->sameAs(out);
    outputs.push_back(maybe_mutated);
  }

  std::vector<Val*> inputs;
  for (auto in : rop->inputs()) {
    auto maybe_mutated = maybeMutated(in);
    is_same = is_same && maybe_mutated->sameAs(in);
    inputs.push_back(maybe_mutated);
  }

  std::vector<Val*> init_vals;
  for (auto init : rop->initVals()) {
    auto maybe_mutated = maybeMutated(init);
    is_same = is_same && maybe_mutated->sameAs(init);
    init_vals.push_back(maybe_mutated);
  }

  if (is_same) {
    return;
  }

  auto container = rop->container();
  const auto& rop_types = rop->getReductionOpTypes();
  container->removeExpr(rop);
  IrBuilder::create<GroupedReductionOp>(
      container, rop_types, init_vals, outputs, inputs, rop->isAllreduce());
}

namespace {
inline bool compareOptional(Val* a, Val* b) {
  if (!a || !b) {
    return (!a && !b);
  }
  return a->sameAs(b);
}

} // namespace

void OptOutMutator::mutate(WelfordOp* wop) {
  Val* out_avg = maybeMutated(wop->outAvg());
  Val* out_var = maybeMutated(wop->outVar());
  Val* out_N = maybeMutated(wop->outN());

  Val* in_avg = maybeMutated(wop->inAvg());
  Val* in_var = wop->inVar() ? maybeMutated(wop->inVar()) : nullptr;
  Val* in_N = maybeMutated(wop->inN());

  Val* init_avg = wop->initAvg() ? maybeMutated(wop->initAvg()) : nullptr;
  Val* init_var = wop->initVar() ? maybeMutated(wop->initVar()) : nullptr;
  Val* init_N = maybeMutated(wop->initN());

  const bool out_compare = out_avg->sameAs(wop->outAvg()) &&
      out_var->sameAs(wop->outVar()) && out_N->sameAs(wop->outN());
  const bool in_compare = in_avg->sameAs(wop->inAvg()) &&
      compareOptional(in_var, wop->inVar()) && in_N->sameAs(wop->inN());
  const bool init_compare = compareOptional(init_avg, wop->initAvg()) &&
      compareOptional(init_var, wop->initVar()) && init_N->sameAs(wop->initN());

  if (out_compare && init_compare && in_compare) {
    return;
  }

  auto container = wop->container();
  container->removeExpr(wop);
  IrBuilder::create<WelfordOp>(
      container,
      out_avg,
      out_var,
      out_N,
      in_avg,
      in_var,
      in_N,
      init_avg,
      init_var,
      init_N,
      wop->isAllreduce());
}

void OptOutMutator::mutate(GroupedWelfordOp* wop) {
  bool is_same = true;

  std::vector<WelfordTriplet> output_vals;
  for (const auto& out : wop->outputVals()) {
    auto maybe_mutated =
        out.transform([&](Val* val) { return maybeMutated(val); });
    is_same = is_same && maybe_mutated.sameAs(out);
    output_vals.push_back(maybe_mutated);
  }

  std::vector<WelfordTriplet> input_vals;
  for (const auto& inp : wop->inputVals()) {
    auto maybe_mutated =
        inp.transform([&](Val* val) { return maybeMutated(val); });
    is_same = is_same && maybe_mutated.sameAs(inp);
    input_vals.push_back(maybe_mutated);
  }

  std::vector<WelfordTriplet> init_vals;
  for (const auto& init : wop->initVals()) {
    auto maybe_mutated =
        init.transform([&](Val* val) { return maybeMutated(val); });
    is_same = is_same && maybe_mutated.sameAs(init);
    init_vals.push_back(maybe_mutated);
  }

  if (is_same) {
    return;
  }

  auto container = wop->container();
  container->removeExpr(wop);
  IrBuilder::create<GroupedWelfordOp>(
      container, output_vals, input_vals, init_vals, wop->isAllreduce());
}

void OptOutMutator::mutate(MmaOp* mma) {
  Val* out = maybeMutated(mma->out());
  Val* in_a = maybeMutated(mma->inA());
  Val* in_b = maybeMutated(mma->inB());
  Val* init = mma->init();

  if (out->sameAs(mma->out()) && in_a->sameAs(mma->inA()) &&
      in_b->sameAs(mma->inB())) {
    return;
  }

  auto container = mma->container();
  auto options = mma->options();
  container->removeExpr(mma);
  C10_UNUSED auto new_mma =
      IrBuilder::create<MmaOp>(container, out, in_a, in_b, init, options);
}

void OptOutMutator::mutate(LoadStoreOp* ldst) {
  Val* out = maybeMutated(ldst->out());
  Val* in = maybeMutated(ldst->in());
  auto op_type = ldst->opType();

  if (out->sameAs(ldst->out()) && in->sameAs(ldst->in())) {
    return;
  }

  auto container = ldst->container();
  container->removeExpr(ldst);
  IrBuilder::create<LoadStoreOp>(container, op_type, out, in);
}

void OptOutMutator::mutate(BroadcastOp* bop) {
  Val* out = maybeMutated(bop->out());
  Val* in = maybeMutated(bop->in());

  if (out->sameAs(bop->out()) && in->sameAs(bop->in())) {
    return;
  }

  auto container = bop->container();
  auto flags = bop->getBroadcastDimFlags();
  container->removeExpr(bop);
  IrBuilder::create<BroadcastOp>(container, out, in, flags);
}

void OptOutMutator::mutate(SqueezeOp* sop) {
  Val* out = maybeMutated(sop->out());
  Val* in = maybeMutated(sop->in());

  if (out->sameAs(sop->out()) && in->sameAs(sop->in())) {
    return;
  }

  auto container = sop->container();
  auto flags = sop->getSqueezeDimFlags();
  container->removeExpr(sop);
  IrBuilder::create<SqueezeOp>(container, out, in, flags);
}

void OptOutMutator::mutate(TransposeOp* top) {
  TensorView* out = maybeMutated(top->out())->as<TensorView>();
  TensorView* in = maybeMutated(top->in())->as<TensorView>();

  if (out->sameAs(top->out()) && in->sameAs(top->in())) {
    return;
  }

  auto container = top->container();
  auto new2old = top->new2old();
  container->removeExpr(top);
  IrBuilder::create<TransposeOp>(container, out, in, new2old);
}

void OptOutMutator::mutate(ExpandOp* eop) {
  bool is_same = true;

  TensorView* out = maybeMutated(eop->out())->as<TensorView>();
  is_same = is_same && out->sameAs(eop->out());
  TensorView* in = maybeMutated(eop->in())->as<TensorView>();
  is_same = is_same && in->sameAs(eop->in());

  std::vector<Val*> expanded_extents;
  expanded_extents.reserve(eop->expanded_extents().size());
  for (auto expanded_extent : eop->expanded_extents()) {
    expanded_extents.push_back(maybeMutated(expanded_extent));
    if (!expanded_extents.back()->sameAs(expanded_extent)) {
      is_same = false;
    }
  }

  if (is_same) {
    return;
  }

  auto container = eop->container();
  container->removeExpr(eop);
  IrBuilder::create<ExpandOp>(container, out, in, expanded_extents);
}

void OptOutMutator::mutate(ShiftOp* sop) {
  Val* out = maybeMutated(sop->out())->asVal();
  Val* in = maybeMutated(sop->in())->asVal();

  if (out->sameAs(sop->out()) && in->sameAs(sop->in())) {
    return;
  }

  auto offsets = sop->offsets();
  auto pad_width = sop->padWidth();
  auto container = sop->container();
  container->removeExpr(sop);
  IrBuilder::create<ShiftOp>(container, out, in, offsets, pad_width);
}

void OptOutMutator::mutate(GatherOp* op) {
  Val* out = maybeMutated(op->out())->asVal();
  Val* in = maybeMutated(op->in())->asVal();

  if (out->sameAs(op->out()) && in->sameAs(op->in())) {
    return;
  }

  auto window_shape = op->windowShape();
  auto pad_width = op->padWidth();
  auto container = op->container();
  container->removeExpr(op);
  IrBuilder::create<GatherOp>(container, out, in, window_shape, pad_width);
}

void OptOutMutator::mutate(ViewAsScalar* vop) {
  TensorView* out = maybeMutated(vop->out())->as<TensorView>();
  TensorView* in = maybeMutated(vop->in())->as<TensorView>();
  IterDomain* vid = maybeMutated(vop->vector_id())->as<IterDomain>();
  Val* idx = maybeMutated(vop->index());

  if (out->sameAs(vop->out()) && in->sameAs(vop->in()) &&
      vid->sameAs(vop->vector_id()) &&
      ((idx == nullptr && vop->index() == nullptr) ||
       idx->sameAs(vop->index()))) {
    return;
  }

  auto container = vop->container();
  container->removeExpr(vop);
  IrBuilder::create<ViewAsScalar>(container, out, in, vid, idx);
}

void OptOutMutator::mutate(ViewOp* vop) {
  TensorView* out = maybeMutated(vop->out())->as<TensorView>();
  TensorView* in = maybeMutated(vop->in())->as<TensorView>();

  if (out->sameAs(vop->out()) && in->sameAs(vop->in())) {
    return;
  }

  auto container = vop->container();
  container->removeExpr(vop);
  IrBuilder::create<ViewOp>(container, out, in);
}

void OptOutMutator::mutate(Split* s) {
  IterDomain* ot = maybeMutated(s->outer())->as<IterDomain>();
  IterDomain* inr = maybeMutated(s->inner())->as<IterDomain>();
  IterDomain* in = maybeMutated(s->in())->as<IterDomain>();
  Val* fact = maybeMutated(s->factor())->as<Val>();
  Val* start_offset = maybeMutated(s->startOffset());
  Val* stop_offset = maybeMutated(s->stopOffset());

  if (ot->sameAs(s->outer()) && inr->sameAs(s->inner()) &&
      in->sameAs(s->in()) && areEqualScalars(fact, s->factor()) &&
      start_offset->sameAs(s->startOffset()) &&
      stop_offset->sameAs(s->stopOffset())) {
    return;
  }

  auto container = s->container();
  auto inner_split = s->innerSplit();
  container->removeExpr(s);
  C10_UNUSED auto new_node = IrBuilder::create<Split>(
      container, ot, inr, in, fact, inner_split, start_offset, stop_offset);
}

void OptOutMutator::mutate(Merge* m) {
  IterDomain* ot = maybeMutated(m->out())->as<IterDomain>();
  IterDomain* otr = maybeMutated(m->outer())->as<IterDomain>();
  IterDomain* in = maybeMutated(m->inner())->as<IterDomain>();

  if (ot->sameAs(m->out()) && otr->sameAs(m->outer()) &&
      in->sameAs(m->inner())) {
    return;
  }

  auto container = m->container();
  container->removeExpr(m);
  C10_UNUSED auto new_node = IrBuilder::create<Merge>(container, ot, otr, in);
}

void OptOutMutator::mutate(Swizzle2D* m) {
  IterDomain* outx = maybeMutated(m->outX())->as<IterDomain>();
  IterDomain* outy = maybeMutated(m->outY())->as<IterDomain>();

  IterDomain* inx = maybeMutated(m->inX())->as<IterDomain>();
  IterDomain* iny = maybeMutated(m->inY())->as<IterDomain>();

  auto swizzle_type = m->swizzleType();

  if (outx->sameAs(m->outX()) && outy->sameAs(m->outY()) &&
      inx->sameAs(m->inX()) && iny->sameAs(m->inY())) {
    return;
  }
  auto container = m->container();
  container->removeExpr(m);
  FusionGuard::getCurFusion()->removeExpr(m);
  C10_UNUSED auto new_node = IrBuilder::create<Swizzle2D>(
      container, outx, outy, inx, iny, swizzle_type);
}

void OptOutMutator::mutate(kir::Allocate*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::BlockSync*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::GridSync*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::CpAsyncWait*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::CpAsyncCommit*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::InitMagicZero*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::UpdateMagicZero*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::ForLoop*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::IfThenElse*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::GridReduction*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::GroupedGridReduction*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::GridBroadcast*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::GridWelford*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::GroupedGridWelford*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}
void OptOutMutator::mutate(kir::AllocateFusedReduction*) {
  TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
}

void OptOutMutator::removeExpr(IrContainer* container, Expr* expr) {
  container->removeExpr(expr);
}
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
