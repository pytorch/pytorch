#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>

// Cleanup
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
DataType aten_opt_type_map(const c10::optional<at::ScalarType>& scalar_type) {
  return scalar_type.has_value() ? aten_to_data_type(scalar_type.value())
                                 : DataType::Null;
}
} // namespace

TensorView::TensorView(TensorDomain* domain, DataType dtype, MemoryType mtype)
    : Val(ValType::TensorView, dtype), domain_(domain), memory_type_(mtype) {
  // Don't do this after transforms
  if (domain_->domain() == domain_->getRootDomain()) {
    // Mark the size-1 axes as broadcast to support implicit broadcast semantic
    for (auto* id : domain_->domain()) {
      if (!id->isBroadcast() && !id->isReduction() &&
          id->rawExtent()->isOneInt()) {
        id->convertToBroadcast();
      }
    }
  }
}

TensorView::TensorView(const std::shared_ptr<c10::TensorType>& tensor_type)
    : Val(ValType::TensorView,
          aten_opt_type_map(tensor_type->scalarType()),
          false) {
  std::vector<IterDomain*> sizes;

  TORCH_CHECK(
      tensor_type->dim().has_value(), "Requires static rank for Tensor");

  for (const auto i : c10::irange(tensor_type->dim().value())) {
    if (tensor_type->sizes()[i].has_value() &&
        tensor_type->sizes()[i].value() == 1) {
      // If size is known to be 1, assuem it needs to be broadcasted.
      sizes.push_back(new IterDomain(
          new Int(0),
          new Int(1),
          ParallelType::Serial,
          IterType::BroadcastWithStride));
    } else {
      sizes.push_back(new IterDomain(new Int(0), new Int()));
    }
  }

  // default to non_contiguous;
  std::vector<bool> contig_info(tensor_type->dim().value(), false);

  // we iterate through stride_index_, which goes from fastest changing
  // dimension to slowest, instead of iterating through sizes. This allows
  // easier contiguity check;
  for (const auto i : c10::irange(tensor_type->dim().value())) {
    // if we don't have contiguous dimension at current stride index, don't
    // bother;
    const auto& stride_property_i = tensor_type->stride_properties()[i];
    if (stride_property_i.has_value() &&
        stride_property_i->stride_index_.has_value() &&
        stride_property_i->contiguous_.has_value() &&
        stride_property_i->contiguous_.value() == true) {
      const size_t index = stride_property_i->stride_index_.value();
      if (i == 0) {
        // mark fastest changing dimension collapsible only when it's the last
        // dim;
        contig_info[index] = (index == tensor_type->dim().value() - 1);
      } else {
        // check the neighboring faster dimension;
        if (auto left_index_opt =
                tensor_type->stride_properties()[static_cast<int>(i) - 1]
                    ->stride_index_) {
          // collapse if two axes are neighboring in both sizes & stride_index;
          contig_info[index] = (left_index_opt.value() == (index + 1));
        }
      }
    }
  }

  domain_ = new TensorDomain(sizes, contig_info);
  name_ = fusion_->registerVal(this);
}

TensorView::TensorView(const TensorView* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      domain_(ir_cloner->clone(src->domain_)),
      compute_at_pos_(src->compute_at_pos_),
      max_producer_pos_(src->max_producer_pos_),
      memory_type_(src->memory_type_),
      swizzle_type_(src->swizzle_type_) {
  for (const auto id : src->axesToSwizzle()) {
    axes_to_swizzle_.push_back(ir_cloner->clone(id));
  }
}

bool TensorView::hasAnyReduction() const {
  return domain()->noReductions().size() != domain()->domain().size();
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

bool TensorView::hasBlockBroadcast() const {
  return domain()->hasBlockBroadcast();
}

bool TensorView::hasBroadcast() const {
  return domain()->hasBroadcast();
}

bool TensorView::hasRFactor() const {
  return domain()->hasRFactor();
}

c10::optional<unsigned int> TensorView::getReductionAxis() const {
  return domain()->getReductionAxis();
}

const std::vector<IterDomain*>& TensorView::getRootDomain() const {
  return domain()->getRootDomain();
};

const std::vector<IterDomain*>& TensorView::getRFactorDomain() const {
  return domain()->getRFactorDomain();
};

const std::vector<IterDomain*>& TensorView::getMaybeRFactorDomain() const {
  return domain()->getMaybeRFactorDomain();
};

std::vector<IterDomain*>::size_type TensorView::nDims() const {
  return domain()->nDims();
}

IterDomain* TensorView::axis(int pos) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to access an axis in a 0-dim TensorView");
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

void TensorView::setComputeAt(unsigned int pos) {
  if (pos <= compute_at_pos_) {
    return;
  }

  TORCH_INTERNAL_ASSERT(
      (unsigned)pos <= nDims(),
      "Invalid this computeAt position for T",
      name(),
      ": ",
      pos);

  compute_at_pos_ = pos;
}

void TensorView::setMaxProducer(unsigned int pos) {
  if (pos <= max_producer_pos_) {
    return;
  }

  TORCH_INTERNAL_ASSERT(
      (unsigned)pos <= nDims(),
      "Invalid max producer position for T",
      name(),
      ": ",
      pos);

  max_producer_pos_ = pos;
}

TensorView* TensorView::computeAt(
    TensorView* consumer,
    int position,
    ComputeAtMode mode) {
  // Make sure this and consumer are not the same tensor, that's illegal
  TORCH_CHECK(!sameAs(consumer), "Cannot call this->computeAt(this, ...)");

  // We support negative axes, so increment it by consumer->nDims() + 1 and make
  // sure the result is within consumer->nDims() + 1. being at consumer->nDims()
  // means producer will be computed inline with consumer, hence the +1.
  if (position < 0)
    position += int(consumer->nDims()) + 1;

  TORCH_CHECK(
      (position >= 0 && (unsigned int)position < consumer->nDims() + 1) ||
          mode == ComputeAtMode::BestEffort,
      "Compute at called on an position outside valid range.");

  if (mode == ComputeAtMode::BestEffort) {
    position = std::max(-1, position);
    position = std::min((int)consumer->nDims(), position);
  }

  ComputeAt::runAt(this, consumer, (unsigned int)position, mode);

  return this;
}

TensorView* TensorView::computeWith(
    TensorView* consumer,
    int position,
    ComputeAtMode mode) {
  // Make sure this and consumer are not the same tensor, that's illegal
  TORCH_CHECK(!sameAs(consumer), "Cannot call this->computeAt(this, ...)");

  // We support negative axes, so increment it by this->nDims() + 1 and make
  // sure the result is within this->nDims() + 1. being at this->nDims()
  // means producer will be computed inline with this, hence the +1.
  if (position < 0)
    position += int(this->nDims()) + 1;
  TORCH_CHECK(
      position >= 0 && (unsigned int)position < this->nDims() + 1,
      "Compute at called on an position outside valid range.");

  ComputeAt::runWith(this, consumer, (unsigned int)position, mode);

  return this;
}

TensorView* TensorView::split(int axis_, Val* factor, bool inner_split) {
  // Only check things associated with axis, factor will be validated in
  // IterDomain
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do split on a 0-dim TensorView");

  if (axis_ < 0)
    axis_ += domain()->nDims();

  TORCH_INTERNAL_ASSERT(
      axis_ >= 0,
      "Split axis is less than 0 even after adjusting for nDims: ",
      axis_);

  TORCH_CHECK(
      axis_ >= (int)getComputeAtPosition(),
      "Cannot split axis within compute at position. Axis = ",
      axis_,
      " computeAtPosition = ",
      getComputeAtPosition());

  TORCH_CHECK(
      axis_ >= (int)getMaxProducerPosition(),
      "Cannot split axis within max producer position. Axis = ",
      axis_,
      " maxProducerPosition = ",
      getMaxProducerPosition());

  TORCH_CHECK(
      axis(axis_)->getParallelType() == ParallelType::Serial,
      "Splitting an axis of non-Serial parallel type is not supported at this time."
      " Parallelization strategy must be set after calling split.");

  domain()->split(axis_, factor, inner_split);
  return this;
}

TensorView* TensorView::split(int axis, unsigned int factor, bool inner_split) {
  split(axis, new Int(factor), inner_split);
  return this;
}

// Merge "axis" and "axis+1" into 1 dimension
TensorView* TensorView::merge(int axis_o, int axis_i) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do merge on a 0-dim TensorView");

  if (axis_o < 0)
    axis_o += domain()->nDims();

  if (axis_i < 0)
    axis_i += domain()->nDims();

  TORCH_CHECK(
      axis_o >= (int)getComputeAtPosition() &&
          axis_i >= (int)getComputeAtPosition(),
      false,
      "Cannot merge axes within compute at position. Either axis ",
      axis_o,
      " or ",
      axis_i,
      " are within computeAtPosition = ",
      getComputeAtPosition());

  TORCH_CHECK(
      axis_o >= (int)getMaxProducerPosition() &&
          axis_i >= (int)getMaxProducerPosition(),
      "Cannot merge axes within max producer position. Either axis ",
      axis_o,
      " or ",
      axis_i,
      " are within maxProducerPosition = ",
      getMaxProducerPosition());

  TORCH_CHECK(
      axis(axis_o)->getParallelType() == ParallelType::Serial ||
          axis(axis_i)->getParallelType() == ParallelType::Serial,
      "Merging axes of non-Serial parallel type is not supported at this time."
      " Parallelization strategy must be set after calling split.");

  domain()->merge(axis_o, axis_i);
  return this;
}

TensorView* TensorView::reorder(const std::unordered_map<int, int>& old2new_) {
  TORCH_INTERNAL_ASSERT(
      !(nDims() == 0 && old2new_.size() > 0),
      "Tried to reorder a 0-dim TensorView");

  for (auto entry : old2new_) {
    auto old_pos = entry.first < 0 ? entry.first + (int)nDims() : entry.first;
    auto new_pos =
        entry.second < 0 ? entry.second + (int)nDims() : entry.second;
    if (old_pos == new_pos) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        old_pos >= 0,
        "Found \"old\" position that's less than 0 even though already adjusted by nDims: ",
        old_pos);
    TORCH_INTERNAL_ASSERT(
        new_pos >= 0,
        "Found \"new\" position that's less than 0 even though already adjusted by nDims: ",
        new_pos);
    TORCH_CHECK(
        old_pos >= (int)getComputeAtPosition() &&
            new_pos >= (int)getComputeAtPosition(),
        "Cannot reorder axes within compute at position. Either axis ",
        old_pos,
        " or ",
        new_pos,
        " are within computeAtPosition = ",
        getComputeAtPosition());

    TORCH_CHECK(
        old_pos >= (int)getMaxProducerPosition() &&
            new_pos >= (int)getMaxProducerPosition(),
        "Cannot reorder axes within max producer position. Either axis ",
        old_pos,
        " or ",
        new_pos,
        " are within maxProducerPosition = ",
        getMaxProducerPosition());
  }

  domain()->reorder(old2new_);
  return this;
}

TensorView* TensorView::swizzle(
    SwizzleType type,
    const std::vector<int>& axes) {
  swizzle_type_ = type;

  // Clear previously set swizzle axes if any
  if (axes_to_swizzle_.size()) {
    axes_to_swizzle_.clear();
  }

  if (swizzle_type_ == SwizzleType::Transpose) {
    TORCH_CHECK(
        axes.size() == 2,
        "Invalid axis list: ",
        axes,
        ". Number of axes must be two.");
    TORCH_CHECK(
        axes[0] != axes[1],
        "Invalid axis list: ",
        axes,
        ". Two distinctive axes must be given.");
    TORCH_CHECK(
        getMemoryType() == MemoryType::Shared,
        "Transpose swizzle is meant for tensors on shared memory.");
    for (auto pos : axes) {
      if (pos < 0) {
        pos += nDims();
      }
      TORCH_CHECK(pos >= 0 && pos < (int)nDims(), "Invalid axis: ", pos);
      TORCH_CHECK(
          pos >= (int)getComputeAtPosition(),
          "Invalid axis: ",
          pos,
          ". Axis outside computeAt position is not allocated.");
      TORCH_CHECK(
          !axis(pos)->isReduction(),
          "Invalid axis: ",
          pos,
          ". Swizzling a reduction axis is not supported");
      TORCH_CHECK(
          !axis(pos)->isBroadcast(),
          "Invalid axis: ",
          pos,
          ". Swizzling a broadcast axis is not supported");
      axes_to_swizzle_.push_back(axis(pos));
    }
  }

  return this;
}

TensorView* TensorView::rFactor(const std::vector<int>& axes) {
  // TODO: I think we should do this but
  // NVFuserTest.FusionSmemBlockGemmCache_CUDA prevents it from going in at the
  // moment.

  // TORCH_INTERNAL_ASSERT(
  //     !hasComputeAt(), "Cannot rfactor tensors after compute at has been
  //     set.");
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to rFactor a 0-dim TensorView");
  TORCH_INTERNAL_ASSERT(definition()->isA<ReductionOp>());
  FusionGuard fg(fusion());
  TORCH_CHECK(
      definition() != nullptr &&
          definition()->getExprType() == ExprType::ReductionOp,
      "Error rfactoring ",
      this,
      " its definition is either a nullptr or not a reduction.");
  TORCH_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  ReductionOp* this_definition = definition()->as<ReductionOp>();

  // Split tensor view into 2 parts
  auto domain_pair = domain()->rFactor(axes);

  // Producer in the pair
  auto producer_domain = domain_pair.first;
  // Consumer in the pair
  auto consumer_domain = domain_pair.second;

  // This domain will be the consumer, so create the producer
  TensorView* producer = new TensorView(producer_domain, getDataType().value());

  // Set domain of consumer
  setDomain(consumer_domain);
  TensorView* consumer = this;

  // Setup dependency chain, inserting producer before this op.
  // Expr* producer_definition =
  new ReductionOp(
      this_definition->getReductionOpType(),
      this_definition->init(),
      producer,
      this_definition->in());

  // Expr* consumer_definition =
  new ReductionOp(
      this_definition->getReductionOpType(),
      this_definition->init(),
      consumer,
      producer);

  return producer;
}

TensorView* TensorView::welfordRfactorHelper(
    TensorView* tv,
    const std::vector<int>& axes) {
  // Hack:
  // Semantically we should always keep the outputs of welfordOp scheduled
  // the same but the user end cannot guarantee that.
  // In order to guarantee that the rFactor is defined meaningfully the
  // scheduling of the output TV that got the rfactor call is force replayed
  // towards the other two

  if (!sameAs(tv)) {
    auto root = tv->getRootDomain();
    auto this_root = getRootDomain();

    // construct a trivial root domain map
    std::unordered_map<IterDomain*, IterDomain*> id_map;
    for (size_t i = 0; i < root.size(); i++) {
      id_map[this_root[i]] = root[i];
    }

    // replay on the target tv
    ReplayTransformations replay(domain()->domain(), id_map);

    // construct the new tensor domain
    std::vector<IterDomain*> new_id;
    for (auto id : domain()->domain()) {
      TORCH_INTERNAL_ASSERT(
          replay.getReplay().count(id), "Welford Replay Failed");
      new_id.push_back(replay.getReplay().at(id));
    }

    std::vector<bool> new_contig(
        tv->domain()->contiguity().begin(), tv->domain()->contiguity().end());
    // replace tensor domain of target tv
    tv->setDomain(new TensorDomain(tv->getRootDomain(), new_id, new_contig));
  }

  // Split tensor view into 2 parts
  auto domain_pair = tv->domain()->rFactor(axes);
  // Producer in the pair
  auto producer_domain = domain_pair.first;
  // Consumer in the pair
  auto consumer_domain = domain_pair.second;

  // This domain will be the consumer, so create the producer
  TensorView* producer =
      new TensorView(producer_domain, tv->getDataType().value());

  // Set domain of consumer
  tv->setDomain(consumer_domain);

  return producer;
}

WelfordResult TensorView::rFactor(
    const std::vector<int>& axes,
    TensorView* var,
    TensorView* avg,
    TensorView* n) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to rFactor a 0-dim TensorView");
  FusionGuard fg(fusion());
  TORCH_CHECK(
      definition() != nullptr &&
          definition()->getExprType() == ExprType::WelfordOp,
      "Error rfactoring welford ",
      this,
      " its definition is either a nullptr or not a welford.");
  TORCH_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  WelfordOp* wop = definition()->as<WelfordOp>();

  TORCH_INTERNAL_ASSERT(
      avg->sameAs(wop->outAvg()), "Welford rfactor not used correctly");
  TORCH_INTERNAL_ASSERT(
      var->sameAs(wop->outVar()), "Welford rfactor not used correctly");
  TORCH_INTERNAL_ASSERT(
      n->sameAs(wop->outN()), "Welford rfactor not used correctly");

  std::unordered_map<TensorView*, TensorView*> tv2rf{
      {var, nullptr}, {avg, nullptr}, {n, nullptr}};

  // Make sure this gets rfactored last so everybody gets
  //  replayed correctly
  for (auto& it : tv2rf) {
    if (!sameAs(it.first)) {
      it.second = welfordRfactorHelper(it.first, axes);
    }
  }

  for (auto& it : tv2rf) {
    if (sameAs(it.first)) {
      it.second = welfordRfactorHelper(it.first, axes);
    }
  }

  TensorView* producer_var = tv2rf.at(var);
  TensorView* producer_avg = tv2rf.at(avg);
  TensorView* producer_n = tv2rf.at(n);

  // Setup dependency chain, inserting producer before this op.
  // Expr* producer_definition =
  new WelfordOp(
      producer_var,
      producer_avg,
      producer_n, /*out var/avg/count */
      wop->initVar(),
      wop->initAvg(),
      wop->initN(), /*init var/avg/count */
      wop->inVar(),
      wop->inAvg(),
      wop->inN());

  // Expr* consumer_definition =
  new WelfordOp(
      var,
      avg,
      n,
      wop->initVar(),
      wop->initAvg(),
      wop->initN(),
      producer_var,
      producer_avg,
      producer_n);

  return WelfordResult(producer_var, producer_avg, producer_n);
}

std::vector<TensorView*> TensorView::duplicate() {
  FusionGuard fg(fusion());

  TORCH_CHECK(
      !fusion()->hasInput(this) && !fusion()->hasOutput(this),
      "Cannot duplicate input or output tensors");

  auto usages = fusion()->unordered_uses(this);
  TORCH_CHECK(
      usages.size() > 1, "Cannot duplicate TensorView that is only used once");

  // Warning: error may occur if the same TensorView
  // is used multiple times in the same expression
  std::vector<TensorView*> duplicates;
  size_t count = 0;
  for (auto expr : usages) {
    // Skip the first usage to reuse original TensorView
    if (count > 0) {
      auto root_domain = getRootDomain();
      TensorView* producer = new TensorView(
          new TensorDomain(
              root_domain, std::vector<bool>(root_domain.size(), true)),
          getDataType().value());

      producer->setDomain(
          TransformReplay::fullSelfReplay(producer->domain(), this->domain()));

      createExprConsumer(definition(), producer);
      createExprProducer(expr, this, producer);

      // Set ComputeAt position for this duplicate TV
      producer->setComputeAt(getComputeAtPosition());

      duplicates.push_back(producer);
    }
    ++count;
  }
  return duplicates;
}

namespace {

// Note: This may be included as an independent member function
// TensorView if it's determined to be useful more generally.
int getMappedConsumerAxis(
    TensorView* producer_tv,
    unsigned int producer_axis,
    TensorView* consumer_tv) {
  auto c2p_root_map =
      PairwiseRootDomainMap(producer_tv, consumer_tv)
          .mapConsumerToProducer(consumer_tv->domain(), producer_tv->domain());
  auto replay = BestEffortReplay(
                    producer_tv->domain()->domain(),
                    consumer_tv->domain()->domain(),
                    c2p_root_map,
                    true)
                    .getReplay();
  auto producer_id = producer_tv->axis(int(producer_axis));
  IterDomain* consumer_id = nullptr;
  for (const auto& m : replay) {
    if (m.second == producer_id) {
      consumer_id = m.first;
    }
  }
  TORCH_INTERNAL_ASSERT(
      consumer_id != nullptr, "Mapped consumer IterDomain not found");
  auto consumer_axis = std::distance(
      consumer_tv->domain()->domain().begin(),
      std::find(
          consumer_tv->domain()->domain().begin(),
          consumer_tv->domain()->domain().end(),
          consumer_id));
  return consumer_axis;
}

} // namespace

TensorView* TensorView::cache_before() {
  FusionGuard fg(fusion());

  TORCH_CHECK(
      definition() != nullptr && !isFusionInput(),
      "Error adding cache_before ",
      this,
      " its definition is a nullptr and we restrict using cache_before on an input.");

  TORCH_CHECK(
      isFusionOutput() || definition()->getExprType() != ExprType::ReductionOp,
      "Error adding cache_before ",
      this,
      " its definition is a reduction and it is not an output, instead please use cache_after.");

  // Create Producer Domain
  // This domain will be the consumer, so create the producer
  auto root_domain = getRootDomain();
  TensorView* producer = new TensorView(
      new TensorDomain(
          root_domain, std::vector<bool>(root_domain.size(), true)),
      getDataType().value());

  // Set domain of consumer
  TensorView* consumer = this;

  // Avoid replaying cache redundantly. Just for efficiency; not
  // required for correctness.
  bool cache_replayed = false;

  // this TV is an output and its definition is a reduction
  // remove reduction axis from this tv
  bool consumer_replay_needed = false;
  if (definition()->getExprType() == ExprType::ReductionOp) {
    size_t i = 0;
    auto no_reduction_root_domain = TensorDomain::noReductions(getRootDomain());
    std::vector<IterDomain*> new_root_domain(no_reduction_root_domain.size());
    for (const auto& dom : no_reduction_root_domain) {
      new_root_domain[i++] = dom->clone();
    }
    // Transform producer like consumer. Note replayPasC not possible yet as
    // there is no producer-consumer relationship.
    producer->setDomain(TransformReplay::fullSelfReplay(
        producer->domain(), consumer->domain()));
    cache_replayed = true;
    consumer->setDomain(new TensorDomain(
        new_root_domain, std::vector<bool>(new_root_domain.size(), true)));
    // The consumer domain should be transformed like the producer,
    // but replayCasP can't be used yet as there is no
    // producer-consumer relationship established yet. Just track
    // it here and replay later after the expression is set.
    consumer_replay_needed = true;
  }

  // Insert producer - Cache_Before (CB) - before this TV.
  // Before: Prev TV -> [Definition Op] -> This TV
  // After:  Prev TV -> [Definition Op] -> New CB TV -> [Set Op] -> This TV

  // Get inputs for origin expression
  auto expr_inputs = definition()->inputs();
  auto def_expr = definition();
  // Expr* producer_definition =
  createExprConsumer(def_expr, producer);

  // Expr* producer_uses =
  new UnaryOp(UnaryOpType::Set, consumer, producer);

  // definition_ is no longer valid
  // setDefinition(nullptr);

  if (consumer_replay_needed) {
    TransformReplay::replayCasP(consumer, producer, -1);
  }

  // Make the cache tensor computed at the consumer if the
  // consumer is computed at another tensor. The position is
  // the same as this position of the consumer. Note that since
  // the consumer is computed at another tensor at this position,
  // there must not be reduction domains in domains until this
  // position, so the removal of reduction domains should not affect
  // position indices.
  // First, make the cache tensor needs look like the consumer. The
  // minimum number of axes to share is getComputeAtPosition(), but
  // it's safe to fully replay.

  // Before: This TV -> Next TV
  // After:  New TV (CB) -> This TV -> Next TV
  if (hasComputeAt()) {
    if (!cache_replayed) {
      TransformReplay::replayPasC(producer, consumer, -1);
      cache_replayed = true;
    }
    producer->setComputeAt(getComputeAtPosition());
  }

  // If the consumer was the target of computeAt by producer's inputs,
  // change the computeAt target to the cache tensor.

  // Before: Prev TV -> This TV
  // After:  Prev TV -> New TV (CB) -> This TV
  // Iterate over definition expression inputs for cache_before on outputs
  size_t producer_this_pos = producer->getComputeAtPosition();
  for (TensorView* producer_of_producer :
       ir_utils::filterByType<TensorView>(expr_inputs)) {
    if (producer_of_producer->hasComputeAt()) {
      if (!cache_replayed) {
        TransformReplay::replayPasC(producer, consumer, -1);
        cache_replayed = true;
      }
      TORCH_INTERNAL_ASSERT(producer_of_producer->getComputeAtPosition() > 0);
      size_t producer_pos =
          getMappedConsumerAxis(
              producer_of_producer,
              int(producer_of_producer->getComputeAtPosition()) - 1,
              producer) +
          1;
      producer_this_pos = std::max(producer_this_pos, producer_pos);
    }
  }

  // Finally, make the cache tensor computed at the consumer. The
  // position is set at the deepest position among the position where
  // its inputs are computed at. If that position is equal or smaller
  // than the position already set by the case where the consumer has
  // computeAt, nothing needs to be done.
  // Note that this step isn't strictly necessary in terms of the
  // Fusion IR semantics, but it's likely what users would want to do
  // anyway.
  if (producer_this_pos > producer->getComputeAtPosition()) {
    // The relative position at the consumer must not include the
    // reduction domains.
    for (size_t i = 0; i < producer_this_pos; ++i) {
      if (i < producer->getComputeAtPosition()) {
        // No CA axes can be reduction.
        TORCH_INTERNAL_ASSERT(!producer->axis(i)->isReduction());
      } else if (producer->axis(i)->isReduction()) {
        producer_this_pos = i;
        break;
      }
    }
    if (producer_this_pos > producer->getComputeAtPosition()) {
      producer->setComputeAt(producer_this_pos);
    }
  }

  return producer;
}

TensorView* TensorView::cache_fork() {
  FusionGuard fg(fusion());

  // Before: [Expr] -> This TV (Global Output) -> [Usage Expr]
  // After:  [Expr] -> This TV (Local) -> [Usage Expr] > Next TV
  //                            (Fork) -> [Set Expr]   -> New TV (Global Output)

  TORCH_CHECK(
      fusion()->hasOutput(this) && !this->uses().empty(),
      "Error adding cache_fork ",
      this,
      " this TensorView must be an output with subsequent uses");

  // This domain will be the producer, so create the consumer
  auto root_domain = TensorDomain::noReductions(getRootDomain());
  TensorView* new_output = new TensorView(
      new TensorDomain(
          root_domain, std::vector<bool>(root_domain.size(), true)),
      getDataType().value());

  // Create write operation from this TV to new output
  new UnaryOp(UnaryOpType::Set, new_output, this);

  // The new TV becomes an output.
  // New TV has global memory type.
  // This TV has local memory type.
  fusion()->replaceOutput(this, new_output);

  // Transform new output according to this TV
  TransformReplay::replayCasP(new_output, this, -1);

  // Set the computeAt for this forked TensorView. It is a terminating
  // output, so set only this position.
  if (hasComputeAt()) {
    auto this_ca_pos = getComputeAtPosition();
    new_output->setComputeAt(this_ca_pos);
  }
  return new_output;
}

TensorView* TensorView::cache_after() {
  FusionGuard fg(fusion());

  const bool kIsFusionInput = fusion()->hasInput(this);

  // Get all the uses for this Tensorview
  TORCH_CHECK(
      !fusion()->hasOutput(this),
      "Error adding cache_after ",
      this,
      " we restrict using cache_after on an output.");

  // Create Consumer Domain
  // Keep Broadcast Axis (Permanent)
  // Remove Reduction Axis
  size_t i = 0;
  auto no_reduction_root_domain = TensorDomain::noReductions(getRootDomain());
  std::vector<IterDomain*> new_root_domain(no_reduction_root_domain.size());
  for (const auto& dom : no_reduction_root_domain) {
    new_root_domain[i++] = dom->clone();
  }

  // This domain will be the producer, so create the consumer
  TensorView* consumer = new TensorView(
      new TensorDomain(
          new_root_domain, std::vector<bool>(new_root_domain.size(), true)),
      getDataType().value());

  // Set domain of producer - No Change
  TensorView* producer = this;

  // Insert consumer - Cache_After (CA) - after this TV.
  // Before: This TV -> [Use Op] -> Next TV
  // After:  This TV -> [Set Op] -> New CA TV -> [Use Op] -> Next TV

  // Expr* consumer_uses =
  for (auto expr : fusion()->unordered_uses(this)) {
    createExprProducer(expr, this, consumer);
  }

  // Expr* consumer_definition =
  new UnaryOp(UnaryOpType::Set, consumer, producer);

  // Before: This TV -> Next TV
  // After:  This TV -> New TV (After) -> Next TV
  if (hasComputeAt()) {
    TransformReplay::replayCasP(consumer, producer, -1);
    consumer->setComputeAt(getComputeAtPosition());
  } else if (kIsFusionInput) {
    bool cache_replayed = false;
    // Check users of this TV for computeAt for cache_after on inputs
    for (const auto& expr : fusion()->unordered_uses(consumer)) {
      for (TensorView* output :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        if (output->hasComputeAt()) {
          if (!cache_replayed) {
            // Completely transform consumer according to output
            TransformReplay::replayPasC(consumer, output, -1);
            cache_replayed = true;
          }
          auto output_ca_pos = output->getComputeAtPosition();
          auto this_pos =
              TransformReplay::replayPasC(consumer, output, output_ca_pos)
                  .second;
          consumer->setComputeAt(this_pos);
        }
      }
    }
  }

  return consumer;
}

void TensorView::setMemoryType(MemoryType mt) {
  memory_type_ = mt;
  if (fusion()->hasInput(this) || fusion()->hasOutput(this)) {
    TORCH_INTERNAL_ASSERT(
        mt == MemoryType::Global,
        "Tried to set an input or output to the fusion to a non-global memory type.");
  }
}

namespace {

// Create New Expr given consumer - [output of the expression]
struct CreateExprConsumer : public OptInDispatch {
 public:
  static void create(Expr* expr, TensorView* consumer) {
    CreateExprConsumer cec(consumer);
    cec.handle(expr);
  }

 private:
  explicit CreateExprConsumer(TensorView* consumer) : consumer_(consumer) {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  void handle(UnaryOp* unary_expr) final {
    new UnaryOp(unary_expr->getUnaryOpType(), consumer_, unary_expr->in());
  }

  void handle(BinaryOp* binary_expr) final {
    new BinaryOp(
        binary_expr->getBinaryOpType(),
        consumer_,
        binary_expr->lhs(),
        binary_expr->rhs());
  }

  void handle(TernaryOp* ternary_expr) final {
    new TernaryOp(
        ternary_expr->getTernaryOpType(),
        consumer_,
        ternary_expr->in1(),
        ternary_expr->in2(),
        ternary_expr->in3());
  }

  void handle(ReductionOp* reduction_expr) final {
    new ReductionOp(
        reduction_expr->getReductionOpType(),
        reduction_expr->init(),
        consumer_,
        reduction_expr->in());
  }

  void handle(BroadcastOp* broadcast_expr) final {
    new BroadcastOp(
        consumer_,
        broadcast_expr->in(),
        broadcast_expr->getBroadcastDimFlags());
  }

  void handle(TransposeOp* transpose_expr) final {
    new TransposeOp(consumer_, transpose_expr->in(), transpose_expr->new2old());
  }

 private:
  TensorView* consumer_ = nullptr;
};

// Create New Expr given producer - [an input for the expression]
struct CreateExprProducer : public OptInDispatch {
 public:
  static void create(Expr* expr, TensorView* current, TensorView* producer) {
    CreateExprProducer cep(current, producer);
    cep.handle(expr);
  }

 private:
  explicit CreateExprProducer(TensorView* current, TensorView* producer)
      : current_(current), producer_(producer) {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  void handle(UnaryOp* unary_expr) final {
    new UnaryOp(unary_expr->getUnaryOpType(), unary_expr->out(), producer_);
  }

  void handle(BinaryOp* binary_expr) final {
    const bool lhs_match = binary_expr->lhs()->sameAs(current_);
    const bool rhs_match = binary_expr->rhs()->sameAs(current_);

    if (lhs_match && rhs_match) {
      new BinaryOp(
          binary_expr->getBinaryOpType(),
          binary_expr->out(),
          producer_,
          producer_);
    } else if (lhs_match) {
      new BinaryOp(
          binary_expr->getBinaryOpType(),
          binary_expr->out(),
          producer_,
          binary_expr->rhs());
    } else {
      new BinaryOp(
          binary_expr->getBinaryOpType(),
          binary_expr->out(),
          binary_expr->lhs(),
          producer_);
    }
  }

  void handle(TernaryOp* ternary_expr) final {
    const bool in1_match = ternary_expr->in1()->sameAs(current_);
    const bool in2_match = ternary_expr->in2()->sameAs(current_);
    const bool in3_match = ternary_expr->in3()->sameAs(current_);

    if (in1_match && in2_match && in3_match) {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          producer_,
          producer_,
          producer_);
    } else if (in1_match && in2_match) {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          producer_,
          producer_,
          ternary_expr->in3());
    } else if (in2_match && in3_match) {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          ternary_expr->in1(),
          producer_,
          producer_);
    } else if (in1_match) {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          producer_,
          ternary_expr->in2(),
          ternary_expr->in3());
    } else if (in2_match) {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          ternary_expr->in1(),
          producer_,
          ternary_expr->in3());
    } else {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          ternary_expr->in1(),
          ternary_expr->in2(),
          producer_);
    }
  }

  void handle(ReductionOp* reduction_expr) final {
    new ReductionOp(
        reduction_expr->getReductionOpType(),
        reduction_expr->init(),
        reduction_expr->out(),
        producer_);
  }

  void handle(BroadcastOp* broadcast_expr) final {
    new BroadcastOp(
        broadcast_expr->out(),
        producer_,
        broadcast_expr->getBroadcastDimFlags());
  }

  void handle(TransposeOp* transpose_expr) final {
    new TransposeOp(
        transpose_expr->out(), producer_, transpose_expr->new2old());
  }

 private:
  TensorView* current_ = nullptr;
  TensorView* producer_ = nullptr;
};

} // namespace

// In Cache Before, for the definition expr of the original tensor,
// we create a new operation where the original tensor is replaced
// with the new cache tensor. This function creates a new expr
// given the consumer, the output of the expression.
void TensorView::createExprConsumer(Expr* expr, TensorView* consumer) {
  CreateExprConsumer::create(expr, consumer);
}

// In Cache After, for all the uses of the original tensor, we create
// a new operation where the original tensor is replaced with the new
// cache tensor. This function creates a new expr given a producer,
// an input for the expression.
void TensorView::createExprProducer(
    Expr* expr,
    TensorView* current,
    TensorView* producer) {
  CreateExprProducer::create(expr, current, producer);
}

TensorViewBuilder& TensorViewBuilder::ndims(size_t ndims) {
  TORCH_CHECK(shape_.empty() || shape_.size() == ndims);
  TORCH_CHECK(contiguity_.empty() || contiguity_.size() == ndims);
  ndims_ = ndims;
  return *this;
}

TensorViewBuilder& TensorViewBuilder::dtype(DataType dtype) {
  dtype_ = dtype;
  return *this;
}

TensorViewBuilder& TensorViewBuilder::contiguity(std::vector<bool> contiguity) {
  TORCH_CHECK(contiguity_.empty(), "Attempting to reset contiguity");
  if (!contiguity.empty()) {
    TORCH_CHECK(ndims_ == 0 || ndims_ == contiguity.size());
    ndims_ = contiguity.size();
  }
  contiguity_ = std::move(contiguity);
  return *this;
}

TensorViewBuilder& TensorViewBuilder::shape(std::vector<int64_t> shape) {
  TORCH_CHECK(shape_.empty(), "Attempting to reset shape");
  if (!shape.empty()) {
    TORCH_CHECK(ndims_ == 0 || ndims_ == shape.size());
    ndims_ = shape.size();
  }
  shape_ = std::move(shape);
  return *this;
}

TensorView* TensorViewBuilder::build() const {
  // Build the domain
  std::vector<IterDomain*> domain(ndims_, nullptr);
  for (size_t i = 0; i < ndims_; i++) {
    if (shape_.empty() || shape_[i] == -1) {
      domain[i] = new IterDomain(new Int(0), new Int());
    } else {
      TORCH_CHECK(
          shape_[i] > 0,
          "Invalid extent value. ",
          "For a tensor representing a single scalar use ndims = 0 with no sizes set.");
      domain[i] = new IterDomain(new Int(0), new Int(shape_[i]));
    }
  }

  // Create the final TensorView
  return new TensorView(new TensorDomain(domain, contiguity_), dtype_);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
