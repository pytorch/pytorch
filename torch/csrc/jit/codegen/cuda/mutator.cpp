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
  mutations_[val] = mutation;
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

void OptOutMutator::mutate(Expr* op) {
  std::vector<Val*> mutated_inputs;
  mutated_inputs.reserve(op->inputs().size());
  for (auto input : op->inputs()) {
    mutated_inputs.emplace_back(maybeMutated(input));
  }

  std::vector<Val*> mutated_outputs;
  mutated_outputs.reserve(op->outputs().size());
  for (auto output : op->outputs()) {
    mutated_outputs.emplace_back(maybeMutated(output));
  }

  std::vector<Statement*> mutated_attrs;
  mutated_attrs.reserve(op->attributes().size());
  for (auto attr : op->attributes()) {
    if (auto attr_val = dynamic_cast<Val*>(attr)) {
      mutated_attrs.emplace_back(maybeMutated(attr_val));
    } else {
      mutated_attrs.emplace_back(attr);
    }
  }

  bool all_same = true;
  for (auto i : c10::irange(op->outputs().size())) {
    if (!all_same) {
      break;
    }
    all_same = all_same && mutated_outputs[i] == op->output(i);
  }
  for (auto i : c10::irange(op->inputs().size())) {
    if (!all_same) {
      break;
    }
    all_same = all_same && mutated_inputs[i] == op->input(i);
  }
  for (auto i : c10::irange(op->attributes().size())) {
    if (!all_same) {
      break;
    }
    bool same =
        ((mutated_attrs[i] == nullptr) && (op->attribute(i) == nullptr)) ||
        mutated_attrs[i] == op->attribute(i);
    all_same = all_same && same;
  }

  if (all_same) {
    return;
  }

  auto container = op->container();
  auto newObjectFunc = op->newObjectFunc();
  removeExpr(container, op);
  auto new_expr =
      newObjectFunc(container, mutated_inputs, mutated_outputs, mutated_attrs);
  registerNewExpr(new_expr);
}

void OptOutMutator::removeExpr(IrContainer* container, Expr* expr) const {
  container->removeExpr(expr);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
