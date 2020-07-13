#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {
DataType aten_opt_type_map(const c10::optional<at::ScalarType>& scalar_type) {
  return scalar_type.has_value() ? aten_to_data_type(scalar_type.value())
                                 : DataType::Null;
}
} // namespace

TensorView::TensorView(TensorDomain* _domain, DataType dtype)
    : Val(ValType::TensorView, dtype), domain_(_domain) {}

TensorView::TensorView(const std::shared_ptr<c10::TensorType>& tensor_type)
    : Val(ValType::TensorView,
          aten_opt_type_map(tensor_type->scalarType()),
          false) {
  std::vector<IterDomain*> sizes;
  TORCH_CHECK(
      tensor_type->dim().has_value(), "Requires static rank for Tensor");
  for (decltype(tensor_type->dim().value()) i = 0;
       i < tensor_type->dim().value();
       i++) {
    sizes.push_back(new IterDomain(new Int(0), new Int()));
  }
  domain_ = new TensorDomain(sizes);

  this->name_ = fusion_->registerVal(this);
}

bool TensorView::hasReduction() const {
  return domain()->hasReduction();
}

bool TensorView::hasBroadcast() const {
  return domain()->hasBroadcast();
}

const std::vector<IterDomain*>& TensorView::getRootDomain() const {
  return domain()->rootDomain();
};

std::vector<IterDomain*>::size_type TensorView::nDims() const {
  return domain()->nDims();
}

IterDomain* TensorView::axis(int pos) const {
  if (pos < 0)
    pos += domain()->nDims();
  TORCH_CHECK(
      pos >= 0 && (unsigned int)pos < domain()->nDims(),
      "Tried to access position ",
      pos,
      " in domain: ",
      domain());
  return domain()->axis(pos);
}

TensorView* TensorView::unsafeClone() const {
  TensorView* new_view = new TensorView(domain_, getDataType().value());
  new_view->compute_at_view_ = compute_at_view_;
  new_view->relative_compute_at_axis_ = relative_compute_at_axis_;
  new_view->this_compute_at_axis_ = this_compute_at_axis_;
  new_view->setMemoryType(memory_type_);
  new_view->name_ = name();

  return new_view;
}

void TensorView::setComputeAt(TensorView* computeAtView, int axis) {
  compute_at_view_ = computeAtView;
  relative_compute_at_axis_ = axis;
  setThisComputeAtAxis();

  TORCH_INTERNAL_ASSERT(
      getThisComputeAtAxis() >= 0 &&
          (unsigned int)getThisComputeAtAxis() <= nDims(),
      "Invalid computeAt on ",
      this,
      " tried to set to local axis ",
      getThisComputeAtAxis());

  TORCH_INTERNAL_ASSERT(
      std::none_of(
          domain()->domain().begin(),
          domain()->domain().begin() + getThisComputeAtAxis(),
          [](IterDomain* id) { return id->isReduction(); }),
      "Invalid computeAt, reduction domain inside computeAt axis.");
}

void TensorView::copyDomain(const TensorDomain* td) {
  std::vector<IterDomain*> idv;
  for (decltype(td->nDims()) i = 0; i < td->nDims(); i++)
    idv.push_back(td->axis(i));
  setDomain(new TensorDomain(idv));
}

// Where in compute_at_view does this->axis(pos) match up?
int TensorView::getComputeAtRelPos(int pos) {
  if (!hasComputeAt())
    return pos;

  if (!compute_at_view_->hasBroadcast())
    return pos;

  size_t pos_cav = 0, pos_this = 0;
  while ((int)pos_this < pos) {
    TORCH_INTERNAL_ASSERT(
        pos_cav < nDims(), "Error computing relative position in computeAt.");
    if (compute_at_view_->axis(pos_cav)->isBroadcast() &&
        !(axis(pos_this)->isBroadcast())) {
      pos_cav++;
    } else {
      pos_cav++;
      pos_this++;
    }
  }

  return pos_cav;
}

void TensorView::setThisComputeAtAxis() {
  if (compute_at_view_ == nullptr) {
    relative_compute_at_axis_ = 0;
    this_compute_at_axis_ = 0;
    return;
  }

  // this[is{i1}, is{i2},] -> compute at compute_at_view[bS{i0}, iS{i1}, iS{i2}]
  // axis = 2 this compute at axis = 1

  // pos in compute at view
  size_t pos_cav = 0, pos_this = 0;
  while (pos_cav < relative_compute_at_axis_ && pos_this < nDims()) {
    if (compute_at_view_->axis(pos_cav)->isBroadcast() &&
        !(axis(pos_this)->isBroadcast())) {
      pos_cav++;
    } else {
      pos_cav++;
      pos_this++;
    }
  }

  TORCH_INTERNAL_ASSERT(
      pos_cav == relative_compute_at_axis_ ||
          (pos_cav < compute_at_view_->nDims() &&
           compute_at_view_->axis(pos_cav)->isBroadcast()),
      "Error seting up relative position between this and what we view into.");

  this_compute_at_axis_ = pos_this;
}

// Actually applies transformation
void TensorView::computeAt_impl(
    TensorView* consumer,
    int consumer_compute_at_axis) {
  // Reset view otherwise will conflict with replay.
  clearComputeAt();
  // replay this as consumer / producer as consumer
  TransformReplay::replayPasC(this, consumer, consumer_compute_at_axis);
  setComputeAt(consumer, consumer_compute_at_axis);
}

// Actually applies transformation
void TensorView::forwardComputeAt_impl(
    TensorView* producer,
    int producer_compute_at_axis) {
  // Reset view otherwise will conflict with replay.
  producer->clearComputeAt();
  TransformReplay::replayCasP(this, producer, producer_compute_at_axis);
  producer->setComputeAt(this, producer_compute_at_axis);
}

namespace {
// Wrapper around set_intersection
template <typename T>
std::set<T> set_intersection(const std::set<T>& set1, const std::set<T>& set2) {
  std::set<T> intersection;
  std::set_intersection(
      set1.begin(),
      set1.end(),
      set2.begin(),
      set2.end(),
      std::inserter(intersection, intersection.begin()));
  return intersection;
}

// convert an iterable of Val* to be an iterable of TensorView*
template <typename T1, typename T2>
T1 tv_iterable(const T2& val_iterable) {
  T1 tv_iterable = T1();
  std::transform(
      val_iterable.begin(),
      val_iterable.end(),
      std::back_inserter(tv_iterable),
      [](Val* v) {
        TORCH_INTERNAL_ASSERT(
            v->getValType().value() == ValType::TensorView,
            "When following the computeAt dependency chain, a non TensorView value was found.");
        return static_cast<TensorView*>(v);
      });
  return tv_iterable;
}
} // namespace

TensorView* TensorView::computeAt(TensorView* consumer, int axis) {
  TORCH_CHECK(
      this->fusion() == consumer->fusion(),
      this,
      " and ",
      consumer,
      " are not in the same fusion.");

  FusionGuard fg(this->fusion());

  TORCH_CHECK(
      !this->sameAs(consumer), "Cannot call this->computeAt(this, ...)");

  if (axis < 0)
    // Compute at is a bit strange where size is the maximum acceptable value
    // instead of size-1
    axis += int(consumer->nDims()) + 1;

  TORCH_CHECK(
      axis >= 0 && (unsigned int)axis < consumer->nDims() + 1,
      "Compute at called on an axis outside valid range.");

  // If not direct relationship follow dependency chain from consumer to
  // producer.
  auto dep_chains = DependencyCheck::getAllDependencyChains(this, consumer);

  std::deque<Val*> dep_chain;
  if (!dep_chains.empty())
    dep_chain = dep_chains.front();

  // Make sure there is a dependency chain, if not it's an invalid computeAt.
  // We could do indirect computeAts, but it's not supported at this time.
  TORCH_CHECK(
      !dep_chain.empty(),
      "Compute At expects ",
      this,
      " is a dependency of ",
      consumer,
      ", however it is not.");

  // Validate dependency chain returned as expected
  TORCH_INTERNAL_ASSERT(
      dep_chain.back() == consumer && dep_chain[0] == this,
      "Error computing dependency chain.");

  // Start the replay going from consumer, through the dependency chain to
  // producer. After this section, producer should look like consumer, and there
  // should be a computeAt chain going from producer to consumer. Proper
  // computeAts are setup, though they will be over-written in a later stage.
  while (dep_chain.size() > 1) {
    Val* consumer_val = dep_chain.back();
    dep_chain.pop_back();
    Val* producer_val = dep_chain.back();

    TORCH_INTERNAL_ASSERT(
        consumer_val->getValType().value() == ValType::TensorView &&
            producer_val->getValType().value() == ValType::TensorView,
        "When following the computeAt dependency chain, a non TensorView value was found.");

    TensorView* running_consumer = static_cast<TensorView*>(consumer_val);
    TensorView* running_producer = static_cast<TensorView*>(producer_val);
    // Axis is relative to consumer, however as we propagate computeAt, it may
    // move. This is why we have TensorView->getThisComputeAtAxis() which
    // returns where in a TensorView does the computeAt (relative to consumer)
    // line up. Mismatch is due to broadcast.
    int compute_at_axis = axis;
    if (running_consumer != consumer)
      compute_at_axis = (int)running_consumer->getThisComputeAtAxis();
    running_producer->computeAt_impl(running_consumer, compute_at_axis);
  }

  /*
   * Compute At has now worked from consumer to producer, transforming producer
   * to match computeAt selected in consumer We now need to work from producer
   * up to its consumers (including indirect consumption) so their use also
   * matches. If we can find a TV that contains all uses of producer (common
   * consumer), we can terminate this propagation there. If not, we need to
   * propagate all the way to outputs.
   */

  // Start looking for a common consumer of producer

  // Grab all uses of producer in fusion
  auto val_all_consumer_chains =
      DependencyCheck::getAllDependencyChainsTo(this);

  // Convert dep chains to tensor view chains
  std::deque<std::deque<TensorView*>> all_consumer_chains;
  for (const auto& val_dep_chain : val_all_consumer_chains)
    all_consumer_chains.push_back(
        tv_iterable<std::deque<TensorView*>>(val_dep_chain));

  // Set arith to find a common consumer, start with first use chain of producer
  std::set<TensorView*> common_consumers(
      all_consumer_chains.front().begin(), all_consumer_chains.front().end());

  // Run through all use chains of producer, and intersect them
  for (auto dep_chain : all_consumer_chains)
    common_consumers = set_intersection(
        common_consumers,
        std::set<TensorView*>(dep_chain.begin(), dep_chain.end()));

  // Remove all TVs between producer and consumer as we don't want a common
  // consumer placed logically before consumer provided in computeAt
  for (const auto& dep_chain : dep_chains) {
    auto tv_chain = tv_iterable<std::deque<TensorView*>>(dep_chain);
    for (auto tv : tv_chain) {
      if (tv != consumer)
        common_consumers.erase(tv);
    }
  }

  // If there is a common consumer, grab the first one (topologically)
  TensorView* common_consumer = nullptr;
  if (!common_consumers.empty()) {
    for (TensorView* tv : all_consumer_chains.front())
      if (common_consumers.find(tv) != common_consumers.end()) {
        common_consumer = tv;
        break;
      }
  }

  // Forward propagate the transformationthrough all use chains until
  // common_consumer if there is one otherwise until we hit all output TVs
  std::set<TensorView*> output_set;
  // computeAt axis in outputs don't necessarily match up, make sure to keep the
  // relative computeAt position in each output
  std::vector<std::pair<TensorView*, int>> ordered_outputs;
  for (auto dep_chain : all_consumer_chains) {
    // All dep chains start with this.
    TORCH_INTERNAL_ASSERT(
        dep_chain.front() == this,
        "Invalid dependency chain found during computeAt, ",
        dep_chain.front(),
        " should be ",
        this);
    TORCH_INTERNAL_ASSERT(
        this->hasComputeAt(),
        "Error detected during computeAt, ",
        this,
        ", should have a computeAt set at this point even though we will over-write it.");
    int running_producer_compute_at = (int)this->getThisComputeAtAxis();
    while (dep_chain.size() > 1) {
      TensorView* running_producer = dep_chain.front();
      dep_chain.pop_front();
      TensorView* running_consumer = dep_chain.front();

      if (running_producer == common_consumer)
        break;
      // Axis is relative to consumer, and may not necessarily apply to all
      // intermediate steps. Fortunately producer is guarenteed to have a valid
      // computeAt set, so we can use the compute at axis relative to producer.
      running_consumer->forwardComputeAt_impl(
          running_producer, running_producer_compute_at);
      running_producer_compute_at =
          (int)running_producer->getThisComputeAtAxis();
      int consumer_compute_at =
          (int)running_producer->getRelativeComputeAtAxis();

      if (dep_chain.size() == 1) { // last one
        if (output_set.find(running_consumer) == output_set.end()) {
          output_set.emplace(running_consumer);
          ordered_outputs.emplace_back(std::pair<TensorView*, int>(
              running_consumer, consumer_compute_at));
        }
      }
    }
  }

  if (!ordered_outputs.empty())
    for (auto it = ordered_outputs.begin(); it + 1 != ordered_outputs.end();
         it++)
      (*it).first->computeAt_impl(
          (*(it + 1)).first,
          (*(it + 1)).second); // use recorded position, not axis.

  return this;
}

TensorView* TensorView::split(int axis, unsigned int factor) {
  if (axis < 0)
    axis += domain()->nDims();

  if (getComputeAtView() != nullptr)
    if (axis < (int)getThisComputeAtAxis())
      TORCH_CHECK(
          false,
          "Cannot split axis within compute at range. Axis = ",
          axis,
          " thisComputeAtAxis = ",
          getThisComputeAtAxis());

  domain()->split(axis, factor);
  return this;
}

// Merge "axis" and "axis+1" into 1 dimension
TensorView* TensorView::merge(int axis_o, int axis_i) {
  if (axis_o < 0)
    axis_o += domain()->nDims();

  if (axis_i < 0)
    axis_i += domain()->nDims();

  if (getComputeAtView() != nullptr)
    if (axis_o + 1 < (int)getThisComputeAtAxis() ||
        axis_i + 1 < (int)getThisComputeAtAxis())
      TORCH_CHECK(
          false,
          "Cannot merge axis within compute at range. Either axis ",
          axis_o,
          " or ",
          axis_i,
          " are within thisComputeAtAxis = ",
          getThisComputeAtAxis());

  domain()->merge(axis_o, axis_i);
  return this;
}

TensorView* TensorView::reorder(const std::unordered_map<int, int>& old2new_) {
  domain()->reorder(old2new_);
  return this;
}

/*
 * Take reduction axes out of this domain, and create a new domain. New domain
 * will be used to create this domain. For example: TV1[I0, I1] = TV0[I0, R0,
 * R1, I1] TV0->rfactor({1}) TV0 is transformed to -> TV0[I0, R1, I1] The
 * TensorView returned is: TV2[I0, R0, I3, I1] The reduction will now beset
 * as: TV1[I0, R1, I1] = TV2[I0, R0, I3, I1] TV0[I0, I1] = TV1[I0, R1, I1]
 */

TensorView* TensorView::rFactor(const std::vector<int>& axes) {
  FusionGuard fg(this->fusion());
  Expr* origin_expr = this->fusion()->origin(this);
  TORCH_CHECK(
      origin_expr != nullptr &&
          origin_expr->getExprType() == ExprType::ReductionOp,
      "Error rfactoring ",
      this,
      " its origin is either a nullptr or not a reduction.");
  TORCH_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  ReductionOp* this_origin = static_cast<ReductionOp*>(origin_expr);

  // Split tensor view into 2 parts
  auto domain_pair = domain()->rFactor(axes);

  // Producer in the pair
  auto producer_domain = domain_pair.first;
  // Consumer in the pair
  auto consumer_domain = domain_pair.second;

  // This domain will be the consumer, so create the producer
  TensorView* producer =
      new TensorView(producer_domain, this->getDataType().value());

  // Set domain of consumer
  this->setDomain(consumer_domain);
  TensorView* consumer = this;

  // Setup dependency chain, inserting producer before this op.
  // Expr* producer_origin =
  new ReductionOp(
      this_origin->getReductionOpType(),
      this_origin->init(),
      producer,
      this_origin->in());

  // Expr* consumer_origin =
  new ReductionOp(
      this_origin->getReductionOpType(),
      this_origin->init(),
      consumer,
      producer);

  return producer;
}

} // namespace fuser
} // namespace jit
} // namespace torch
