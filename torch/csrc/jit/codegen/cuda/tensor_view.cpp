#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {
DataType aten_opt_type_map(const c10::optional<at::ScalarType>& scalar_type) {
  return scalar_type.has_value() ? aten_to_data_type(scalar_type.value())
                                 : DataType::Null;
}

c10::optional<TensorContiguity> infer_contiguity_from_tensor_type(
    const std::shared_ptr<c10::TensorType>& tensor_type) {
  if (!tensor_type->isComplete()) {
    return c10::nullopt;
  } else {
    return TensorContiguity(
        *(tensor_type->sizes().concrete_sizes()),
        *(tensor_type->strides().concrete_sizes()));
  }
}

} // namespace

TensorView* split_(TensorView* tv, int axis, int factor) {
  TensorDomain* td = tv->domain();

  if (axis < 0)
    axis += td->size();

  assert(axis >= 0 && axis < td->size());

  IterDomain* id = td->axis(axis);

  if (id->parallel_method() != ParallelType::Serial)
    TORCH_CHECK(
        false,
        "Splitting an axis of non-Serial iteration is not supported at this time."
        " Parallelization strategy must be set after calling split.");

  if (tv->getComputeAtView() != nullptr)
    if (axis < tv->getComputeAtAxis())
      TORCH_CHECK(false, "Cannot split axis within the compute at range.");

  std::vector<IterDomain*> new_domain;

  Int* fact = new Int(factor);
  Int* one = new Int(1);

  for (decltype(td->size()) i = 0; i < td->size(); i++) {
    if (i != axis)
      new_domain.push_back(td->axis(i));
    else {
      // outer loop size
      Val* vo = ceilDiv(id->size(), fact);
      Int* so = static_cast<Int*>(vo);

      // outer loop IterDomain
      IterDomain* ido =
          new IterDomain(so, id->parallel_method(), id->isReduction());
      new_domain.push_back(ido);

      // inner loop IterDomain
      IterDomain* idi =
          new IterDomain(fact, id->parallel_method(), id->isReduction());
      new_domain.push_back(idi);
    }
  }
  TensorDomain* split_td = new TensorDomain(new_domain);
  Split* split_node = new Split(split_td, td, axis, fact); // For record keeping
  tv->setDomain(split_td);
  return tv;
}

TensorView* merge_(TensorView* tv, int axis) {
  TensorDomain* td = tv->domain();

  if (axis < 0)
    axis += td->size();

  assert(axis >= 0 && axis + 1 < td->size());

  if (tv->getComputeAtView() != nullptr)
    if (axis < tv->getComputeAtAxis())
      TORCH_CHECK(false, "Cannot split axis within compute at range.");

  IterDomain* first = td->axis(axis);
  IterDomain* second = td->axis(axis + 1);

  assert(first->isReduction() == second->isReduction());
  assert(first->parallel_method() == second->parallel_method());

  Val* merged_id_size = mul(first->size(), second->size());
  IterDomain* merged_id = new IterDomain(
      static_cast<Int*>(merged_id_size),
      first->parallel_method(),
      first->isReduction());

  std::vector<IterDomain*> new_domain;
  for (decltype(td->size()) i = 0; i < td->size(); i++) {
    if (i < axis || i > axis + 1)
      new_domain.push_back(td->axis(i));
    else if (i == axis) {
      new_domain.push_back(merged_id);
    }
  }
  TensorDomain* merged_td = new TensorDomain(new_domain);
  Merge* merge_node = new Merge(merged_td, td, axis); // For record keeping
  tv->setDomain(merged_td);
  return tv;
}

/*
 * Takes axis2pos map, axis2pos[old_pos] = new_pos, to modify the ordering of
 * the iter axes.
 */
TensorView* reorder_(
    TensorView* tv,
    const std::unordered_map<int, int>& axis2pos) {
  TensorDomain* td = tv->domain();
  auto ndims = td->size();
  // Map to save from previous order, to new order.
  std::vector<int> pos2axis(ndims, -1);

  // Go through each old and new position, make sure they're within 0-ndims
  for (std::pair<int, int> elem : axis2pos) {
    int old_pos = elem.first;
    int new_pos = elem.second;

    if (old_pos < 0)
      old_pos += ndims;
    if (new_pos < 0)
      new_pos += ndims;

    assert(old_pos >= 0 && old_pos < ndims && new_pos >= 0 && new_pos < ndims);

    if (pos2axis[new_pos] != -1)
      TORCH_CHECK(false, "Reorder found duplicate destination positions.");

    pos2axis[new_pos] = old_pos;
  }

  std::set<int> old_positions(pos2axis.begin(), pos2axis.end());
  old_positions.erase(-1);

  if (old_positions.size() != axis2pos.size())
    TORCH_INTERNAL_ASSERT(
        false, "Reorder found duplicate destination positions.");

  std::set<int> all_positions;
  for (decltype(ndims) i{0}; i < ndims; i++)
    all_positions.insert(i);

  // Check what positions haven't been specified.
  std::set<int> positions_left;
  std::set_difference(
      all_positions.begin(),
      all_positions.end(),
      old_positions.begin(),
      old_positions.end(),
      std::inserter(positions_left, positions_left.end()));

  // Fill in positions that weren't specified, in relative order,
  // in empty spots in the set of new positions.
  // pos2axis[new_position] = old_position
  auto it = positions_left.begin(); // old positions left
  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    if (pos2axis[i] == -1)
      pos2axis[i] = *it++;
  }

  // pos2axis is now filled
  if (tv->getComputeAtView() != nullptr) {
    for (int i = 0; i < tv->getComputeAtAxis(); i++) {
      if (pos2axis[i] != i)
        TORCH_CHECK(false, "Cannot reorder axis within compute at range.");
    }
  }

  std::vector<IterDomain*> reordered_domain;
  for (int entry : pos2axis)
    reordered_domain.push_back(td->axis(entry));

  TensorDomain* reordered_td = new TensorDomain(reordered_domain);
  Reorder* merge_node = new Reorder(reordered_td, td, pos2axis);
  tv->setDomain(reordered_td);
  return tv;
}

TensorView::TensorView(TensorDomain* _domain, DataType dtype)
    : Val(ValType::TensorView, dtype), domain_(_domain) {}

TensorView::TensorView(const std::shared_ptr<c10::TensorType>& tensor_type)
    : Val(ValType::TensorView, aten_opt_type_map(tensor_type->scalarType())) {
  std::vector<IterDomain*> sizes;
  TORCH_CHECK(
      tensor_type->dim().has_value(), "Requires static rank for Tensor");
  for (int i = 0; i < tensor_type->dim().value(); i++) {
    sizes.push_back(new IterDomain(new Int()));
  }
  domain_ = new TensorDomain(sizes);
}

TensorView* TensorView::clone() const {
  TensorView* new_view = new TensorView(domain_, getDataType().value());
  new_view->compute_at_view_ = compute_at_view_;
  new_view->compute_at_axis_ = compute_at_axis_;
  return new_view;
}

TensorView* TensorView::newForOutput(DataType dtype) const {
  std::vector<IterDomain*> domain_copy;
  for (decltype(this->nDims()) i = 0; i < this->nDims(); i++) {
    // If reduction axis, don't copy it over. Reduction axes are owned by
    // consumers and we're copying over a producer.
    if (this->axis(i)->isReduction())
      continue;
    domain_copy.push_back(new IterDomain(this->axis(i)->size()));
  }
  TensorDomain* td = new TensorDomain(domain_copy);
  return new TensorView(td, dtype);
};

TensorDomain* TensorView::getRootDomain() const {
  return TransformIter::getRoot(this->domain());
};

void TensorView::resetView() {
  setDomain(getRootDomain());
  compute_at_view_ = nullptr;
  compute_at_axis_ = 0;
}

std::vector<IterDomain*>::size_type TensorView::nDims() const {
  return domain()->size();
}

IterDomain* TensorView::axis(int pos) const {
  if (pos < 0)
    pos += domain()->size();
  TORCH_CHECK(
      pos >= 0 && pos < domain()->size(),
      "Tried to access position ",
      pos,
      " in domain: ",
      domain());
  return domain()->axis(pos);
}

void TensorView::copyDomain(const TensorDomain* td) {
  std::vector<IterDomain*> idv;
  for (decltype(td->size()) i = 0; i < td->size(); i++)
    idv.push_back(td->axis(i));
  setDomain(new TensorDomain(idv));
}

TensorView* TensorView::computeAt(TensorView* consumer, int axis) {
  /*
   * Recursive compute_at:
   * Recurse backward from consumer, to this, make sure there's a dependency
   * chain there. Call ComputeAt for all tensors between this and consumer.
   *
   * Compute at modifies this, not consumer.
   */
  TORCH_CHECK(
      !this->sameAs(consumer), "Cannot call this->computeAt(this, ...)");
  if (axis < 0)
    // Compute at is funny where size is the maximum acceptable value instead of
    // size-1
    axis += int(consumer->nDims()) + 1;

  TORCH_CHECK(
      axis >= 0 && axis < consumer->nDims() + 1,
      "Compute at called on an axis outside valid range.");

  std::stack<Val*> dep_chain =
      DependencyCheck::getDependencyChain(this, consumer);
  // forward apply to uses of this.
  // Recursively apply replay.
  TensorView* running_consumer = consumer;
  // dep_chain = deps <- consumer (this chain doesn't contain this)
  // We want to apply:
  //  dep[n-1].compute_at(consumer)
  //  ...
  //  this.compute_at(dep[0])
  while (!dep_chain.empty()) {
    Val* val = dep_chain.top();
    dep_chain.pop();
    TORCH_INTERNAL_ASSERT(
        val->getValType() == ValType::TensorView,
        "When following the transform dependency chain, an invalid value was found.");
    TensorView* tv = static_cast<TensorView*>(val);
    if (tv->sameAs(consumer))
      continue;
    tv->computeAt(running_consumer, axis);
    // TransformReplay::replay(running_consumer, tv, axis);
    running_consumer = tv; // replay is in-place
  }
  // At this point running_consumer is the direct consumer of this

  // If another view consumes this, we may be computing this at a position that
  // doesn't match that consumer. Likely producing too little for that next
  // consumer
  for (Expr* other_use : FusionGuard::getCurFusion()->uses(this)) {
    for (Val* maybe_other_consumer : other_use->outputs()) {
      if (*(maybe_other_consumer->getValType()) != ValType::TensorView)
        continue;

      TensorView* other_consumer =
          static_cast<TensorView*>(maybe_other_consumer);
      if (running_consumer->sameAs(other_consumer))
        continue;

      if (DependencyCheck::isDependencyOf(running_consumer, other_consumer)) {
        // There seem to be two choices here, either running_consumer or
        // consumer I believe they end up being equivelent, but uncertain they
        // actually are.
        running_consumer->computeAt(other_consumer, axis);
      } else {
        other_consumer->computeAt(running_consumer, axis);
      }
    }
  }

  if (FusionGuard::getCurFusion()->origin(this) == nullptr)
    return this;

  // Dep chain doesn't contain this, try to run on replay, as it may be merging
  // two independent loop nests of the same sizes.

  // Reset view otherwise will conflict with replay.

  this->compute_at_view_ = nullptr;
  this->compute_at_axis_ = -1;
  TransformReplay::replay(running_consumer, this, axis);
  this->compute_at_view_ = running_consumer;
  this->compute_at_axis_ = (unsigned int)axis;
  return this;
}

} // namespace fuser
} // namespace jit
} // namespace torch
