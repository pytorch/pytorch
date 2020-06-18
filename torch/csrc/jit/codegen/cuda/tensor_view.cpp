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

bool TensorView::hasBlockReduction() const {
  return domain()->hasBlockReduction();
}

bool TensorView::hasGridReduction() const {
  return domain()->hasGridReduction();
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
        pos_cav < compute_at_view_->nDims(),
        "Error computing relative position in computeAt.");
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
  // Reset view otherwise will conflict with replay. Don't think this is true
  // anymore.
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

// Takes this tensor and tries to set up the schedule so that it's computed
// relative to consumer within axis. Simple example is if both this and consumer
// are 2D and this->computeAt(consumer, 1) is called, then only 1D of this will
// ever be materialized at the same time.
//
// Roughly what this function will do is:
//
// (1) Find if there's a consumer which contains all uses of this (aka
// producer). This "common_consumer" must be at consumer or after in the graph.
//
// (2) We will forward the computeAt from consumer to common_consumer.
//
// (3) We will find all paths from common_consumer back to this (aka producer).
// We will propagate the computeAt down this dependency chain to this (aka
// producer).
//
// (4) If there is not a common_consumer we will still have propagated down from
// consumer through all paths to this (aka producer). We will then make sure
// there isn't any broadcast axes within the computeAt axis (if so for now it's
// an error). We then propagate the compute at from this (aka producer) to all
// terminating Vals (registered outputs or leaves).
//
// (5) Set computeAt relative to outputs/leaves found in (4)
TensorView* TensorView::computeAt(TensorView* consumer, int axis) {
  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      this->fusion() == consumer->fusion(),
      this,
      " and ",
      consumer,
      " are not in the same fusion.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(this->fusion());

  // Make sure this and consumer are not the same tensor, that's illegal
  TORCH_CHECK(
      !this->sameAs(consumer), "Cannot call this->computeAt(this, ...)");

  // We support negative axes, so increment it by consumer->nDims() + 1 and make
  // sure the result is within consumer->nDims() + 1. being at consumer->nDims()
  // means this will be computed inline with consumer, hence the +1.
  if (axis < 0)
    axis += int(consumer->nDims()) + 1;
  TORCH_CHECK(
      axis >= 0 && (unsigned int)axis < consumer->nDims() + 1,
      "Compute at called on an axis outside valid range.");

  // Start (1): Look through all the use chains of producer. Check if there's a
  // single consumer for all chains at or after the consumer specified in the
  // computeAt call.

  // Grab all paths from this to  of producer in fusion.
  auto val_all_dep_chains = DependencyCheck::getAllDependencyChainsTo(this);

  // Right now we only support compute at if at some point in the graph consumer
  // is dependent on this.
  TORCH_CHECK(
      !val_all_dep_chains.empty(),
      "Compute At expects ",
      this,
      " is a dependency of ",
      consumer,
      ", however it is not.");

  // Convert dep chains to tensor view chains.
  std::deque<std::deque<TensorView*>> tv_all_dep_chains;
  for (const auto& val_dep_chain : val_all_dep_chains)
    tv_all_dep_chains.push_back(
        tv_iterable<std::deque<TensorView*>>(val_dep_chain));

  // Convert the first chain to a set.
  std::set<TensorView*> common_consumers(
      tv_all_dep_chains.front().begin(), tv_all_dep_chains.front().end());

  // Run through all use chains of producer, and intersect them to find common
  // TVs
  for (auto dep_chain : tv_all_dep_chains)
    common_consumers = set_intersection(
        common_consumers,
        std::set<TensorView*>(dep_chain.begin(), dep_chain.end()));

  // Remove all TVs from producer to consumer as common consumer must be at or
  // after consumer
  for (const auto& dep_chain :
       DependencyCheck::getAllDependencyChains(this, consumer)) {
    auto tv_chain = tv_iterable<std::deque<TensorView*>>(dep_chain);
    for (auto tv : tv_chain) {
      if (tv != consumer)
        common_consumers.erase(tv);
    }
  }

  // If there is a common consumer, grab the first one at or after consumer
  TensorView* common_consumer = nullptr;
  if (!common_consumers.empty()) {
    for (TensorView* tv : tv_all_dep_chains.front())
      if (common_consumers.find(tv) != common_consumers.end()) {
        common_consumer = tv;
        break;
      }
    TORCH_INTERNAL_ASSERT(
        common_consumer != nullptr,
        "Hit a logical inconsistency in the computeAt pass.");
  }

  // Record what axis we computeAt into, this is a map from "consumer" to
  // computeAt point in "consumer", consumer in this context changes as we
  // traverse the graph
  std::unordered_map<TensorView*, int> compute_at_axis_lookup;
  compute_at_axis_lookup[consumer] = axis;

  // Track if we set computeAt on a TV
  std::unordered_set<TensorView*> compute_at_set;
  // Track if a TV was transformed by this pass
  std::unordered_set<TensorView*> transformed;

  // Start (2): Prop forward the computeAt from consumer to common_consumer if
  // it exists
  if (common_consumer != nullptr && common_consumer != consumer) {
    int running_producer_compute_at = axis;
    std::deque<Val*> dep_chain =
        DependencyCheck::getSingleDependencyChain(consumer, common_consumer);
    TORCH_INTERNAL_ASSERT(
        !dep_chain.empty(), "Computed an invalid common_consumer.");
    std::deque<TensorView*> tv_dep_chain =
        tv_iterable<std::deque<TensorView*>>(dep_chain);

    TensorView* running_consumer = tv_dep_chain.front();
    tv_dep_chain.pop_front();

    TensorView* running_producer = nullptr;

    while (!tv_dep_chain.empty()) {
      running_producer = running_consumer;
      running_consumer = tv_dep_chain.front();
      tv_dep_chain.pop_front();

      if (transformed.find(running_consumer) != transformed.end())
        continue;

      running_consumer->forwardComputeAt_impl(
          running_producer, running_producer_compute_at);

      // Update compute_at_set, transformed, and compute_at_axis_lookup
      compute_at_set.emplace(running_producer);
      transformed.emplace(running_consumer);

      auto it = compute_at_axis_lookup.find(running_consumer);
      if (it != compute_at_axis_lookup.end()) {
        TORCH_INTERNAL_ASSERT(
            it->second == running_producer->getRelativeComputeAtAxis(),
            "Hit a logical inconsistency in the computeAt pass.");
      } else {
        compute_at_axis_lookup[running_consumer] =
            running_producer->getRelativeComputeAtAxis();
      }
    }
  }

  // Start (3): Propagate back from common_consumer if it exists, or consumer
  // through all paths to producer
  TensorView* running_consumer =
      common_consumer == nullptr ? consumer : common_consumer;

  // Grab all chains from common_consumer to this
  const auto val_all_consumer_chains =
      DependencyCheck::getAllDependencyChains(this, running_consumer);

  // Convert dep chains to tensor view chains
  std::deque<std::deque<TensorView*>> tv_all_consumer_chains;
  for (const auto& val_dep_chain : val_all_consumer_chains)
    tv_all_consumer_chains.push_back(
        tv_iterable<std::deque<TensorView*>>(val_dep_chain));

  for (auto tv_chain : tv_all_consumer_chains) {
    TensorView* running_producer = tv_chain.back();
    tv_chain.pop_back();

    while (!tv_chain.empty()) {
      running_consumer = running_producer;
      running_producer = tv_chain.back();
      tv_chain.pop_back();

      auto it = compute_at_axis_lookup.find(running_consumer);
      TORCH_INTERNAL_ASSERT(
          it != compute_at_axis_lookup.end(),
          "Should have already visisted a consumer, but encountered one that wasn't.");

      if (transformed.find(running_producer) != transformed.end()) {
        if (compute_at_set.find(running_producer) == compute_at_set.end())
          running_producer->setComputeAt(running_consumer, it->second);
        continue;
      }

      running_producer->computeAt_impl(running_consumer, it->second);

      // Update both compute_at_ed and compute_at_axis_lookup
      compute_at_set.emplace(running_producer);
      transformed.emplace(running_producer);

      it = compute_at_axis_lookup.find(running_producer);
      if (it != compute_at_axis_lookup.end()) {
        TORCH_INTERNAL_ASSERT(
            it->second == running_producer->getThisComputeAtAxis(),
            "Hit a logical inconsistency in the computeAt pass.");
      } else {
        compute_at_axis_lookup[running_producer] =
            running_producer->getThisComputeAtAxis();
      }
    } // while (!tv_chain.empty())
  } // for (auto tv_chain : tv_all_consumer_chains)

  TORCH_INTERNAL_ASSERT(
      this->hasComputeAt(),
      "Hit a logical inconsistency in the computeAt pass.");
  bool has_bcast = this->getThisComputeAtAxis() != axis;
  TORCH_INTERNAL_ASSERT(
      !has_bcast || common_consumer != nullptr,
      "A broadcast dim was detected at somepoint to be within computeAt.",
      " However, there isn't a TV that contains all uses of this.",
      " This is not supported at this time.");

  // Forward propagate the transformations through all use chains if we don't
  // have a common_consumer
  if (common_consumer != nullptr)
    return this;

  // Start (4): Propagate computeAt from producer through all uses to
  // terminating values (registered outputs or leaf vals) Hold on to terminating
  // outputs. Keep the order they were added so we can sort them later.
  std::unordered_map<TensorView*, size_t> output_set;
  size_t output_count = 0;

  // computeAt axis in outputs don't necessarily match up, make sure to keep the
  // relative computeAt position in each output
  std::vector<std::pair<TensorView*, int>> ordered_outputs;
  for (auto dep_chain : tv_all_dep_chains) {
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

      if (transformed.find(running_consumer) == transformed.end()) {
        TransformReplay::replayCasP(
            running_consumer, running_producer, running_producer_compute_at);
      }
      if (compute_at_set.find(running_producer) == compute_at_set.end()) {
        running_producer->setComputeAt(
            running_consumer, running_producer_compute_at);
      }

      transformed.emplace(running_consumer);
      compute_at_set.emplace(running_producer);

      running_producer_compute_at =
          (int)running_producer->getThisComputeAtAxis();
      int consumer_compute_at =
          (int)running_producer->getRelativeComputeAtAxis();

      if (dep_chain.size() == 1) { // last one
        if (output_set.find(running_consumer) == output_set.end()) {
          output_set.emplace(std::make_pair(running_consumer, output_count++));
          ordered_outputs.emplace_back(std::pair<TensorView*, int>(
              running_consumer, consumer_compute_at));
        }
      }
    }
  }

  if (!ordered_outputs.empty())
    for (auto it = ordered_outputs.begin(); it + 1 != ordered_outputs.end();
         it++)
      it->first->setComputeAt((it + 1)->first, (it + 1)->second);

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
