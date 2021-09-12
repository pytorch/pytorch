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

TensorView::TensorView(TensorDomain* _domain, DataType dtype, MemoryType mtype)
    : Val(ValType::TensorView, dtype), domain_(_domain), memory_type_(mtype) {}

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
      compute_at_view_(ir_cloner->clone(src->compute_at_view_)),
      relative_compute_at_axis_(src->relative_compute_at_axis_),
      this_compute_at_axis_(src->this_compute_at_axis_),
      memory_type_(src->memory_type_) {}

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

TensorView* TensorView::unsafeClone() const {
  TensorView* new_view = new TensorView(domain_, getDataType().value());
  new_view->compute_at_view_ = compute_at_view_;
  new_view->relative_compute_at_axis_ = relative_compute_at_axis_;
  new_view->this_compute_at_axis_ = this_compute_at_axis_;
  new_view->memory_type_ = memory_type_;
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

void TensorView::setComputeAt(
    TensorView* computeAtView,
    int thisPos,
    int relPos) {
  compute_at_view_ = computeAtView;
  relative_compute_at_axis_ = relPos;
  this_compute_at_axis_ = thisPos;
  TORCH_INTERNAL_ASSERT(
      this_compute_at_axis_ <= nDims(), "Manually set an invalid computeAt.");
}

// Where in compute_at_view does this->axis(pos) match up?
// TODO: This doesn't seem like the safest function as a fusion output can ref
// another fusion output,  we may want to check that there is a direct
// consumer/producer relationship between this and compute_at view before using
// this function, and creating another pass to handle relative outputs.
int TensorView::getComputeAtRelPos(int pos) {
  if (!hasComputeAt()) {
    return pos;
  }

  if (!compute_at_view_->hasBroadcast()) {
    return pos;
  }

  size_t pos_cav = 0, pos_this = 0;

  // We could be in an instance where pos == 0, but consumer[0] is bcast and
  // this[0] is not

  while (compute_at_view_->axis(pos_cav)->isBroadcast() &&
         !(axis(pos_this)->isBroadcast())) {
    pos_cav++;
  }

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

TensorView* TensorView::computeAt(TensorView* consumer, int axis) {
  // Make sure this and consumer are not the same tensor, that's illegal
  TORCH_CHECK(!sameAs(consumer), "Cannot call this->computeAt(this, ...)");

  // We support negative axes, so increment it by consumer->nDims() + 1 and make
  // sure the result is within consumer->nDims() + 1. being at consumer->nDims()
  // means producer will be computed inline with consumer, hence the +1.
  if (axis < 0)
    axis += int(consumer->nDims()) + 1;
  TORCH_CHECK(
      axis >= 0 && (unsigned int)axis < consumer->nDims() + 1,
      "Compute at called on an axis outside valid range.");

  ComputeAt::run(this, consumer, (unsigned int)axis);

  return this;
}

TensorView* TensorView::split(int axis, Val* factor) {
  // Only check things associated with axis, factor will be validated in
  // IterDomain
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do split on a 0-dim TensorView");

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

TensorView* TensorView::split(int axis, unsigned int factor) {
  domain()->split(axis, new Int(factor));
  return this;
}

// Merge "axis" and "axis+1" into 1 dimension
TensorView* TensorView::merge(int axis_o, int axis_i) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do merge on a 0-dim TensorView");
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
  TORCH_INTERNAL_ASSERT(
      !(nDims() == 0 && old2new_.size() > 0),
      "Tried to reorder a 0-dim TensorView");
  domain()->reorder(old2new_);
  return this;
}

TensorView* TensorView::rFactor(const std::vector<int>& axes) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to rFactor a 0-dim TensorView");
  FusionGuard fg(fusion());
  Expr* origin_expr = fusion()->origin(this);
  TORCH_CHECK(
      origin_expr != nullptr &&
          origin_expr->getExprType() == ExprType::ReductionOp,
      "Error rfactoring ",
      this,
      " its origin is either a nullptr or not a reduction.");
  TORCH_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  ReductionOp* this_origin = origin_expr->as<ReductionOp>();

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

TensorView* TensorView::cache_before() {
  FusionGuard fg(fusion());

  Expr* origin_expr = fusion()->origin(this);
  TORCH_CHECK(
      origin_expr != nullptr && !fusion()->hasInput(this),
      "Error adding cache_before ",
      this,
      " its origin is a nullptr and we restrict using cache_before on an input.");

  TORCH_CHECK(
      fusion()->hasOutput(this) ||
          origin_expr->getExprType() != ExprType::ReductionOp,
      "Error adding cache_before ",
      this,
      " its origin is a reduction and it is not an output, instead please use cache_after.");

  // Create Producer Domain
  // This domain will be the consumer, so create the producer
  auto root_domain = getRootDomain();
  TensorView* producer = new TensorView(
      new TensorDomain(
          root_domain, std::vector<bool>(root_domain.size(), true)),
      getDataType().value());

  // Set domain of consumer
  TensorView* consumer = this;

  // this TV is an output and its origin is a reduction
  // remove reduction axis from this tv
  if (origin_expr->getExprType() == ExprType::ReductionOp) {
    size_t i = 0;
    auto no_reduction_root_domain = TensorDomain::noReductions(getRootDomain());
    std::vector<IterDomain*> new_root_domain(no_reduction_root_domain.size());
    for (const auto& dom : no_reduction_root_domain) {
      new_root_domain[i++] = dom->clone();
    }
    consumer->setDomain(new TensorDomain(
        new_root_domain, std::vector<bool>(new_root_domain.size(), true)));
  }

  // Insert producer - Cache_Before (CB) - before this TV.
  // Before: Prev TV -> [Origin Op] -> This TV
  // After:  Prev TV -> [Origin Op] -> New CB TV -> [Set Op] -> This TV

  // Get inputs for origin expression
  auto expr_inputs = origin_expr->inputs();

  // Expr* producer_origin =
  createExprConsumer(origin_expr, producer);

  // Expr* producer_uses =
  new UnaryOp(UnaryOpType::Set, consumer, producer);

  // Before: This TV -> Next TV
  // After:  New TV (CB) -> This TV -> Next TV
  if (hasComputeAt()) {
    TransformReplay::replayPasC(producer, consumer, -1);
    auto this_ca_pos = getThisComputeAtAxis();
    producer->computeAt(consumer, this_ca_pos);
  } else {
    // Before: Prev TV -> This TV
    // After:  Prev TV -> New TV (CB) -> This TV
    // Iterate over origin expression inputs for cache_before on outputs
    for (TensorView* origin_input :
         ir_utils::filterByType<TensorView>(expr_inputs)) {
      if (origin_input->hasComputeAt() &&
          origin_input->getComputeAtView() == this) {
        TransformReplay::replayPasC(producer, consumer, -1);

        auto origin_ca_pos = origin_input->getThisComputeAtAxis();
        auto origin_rel_ca_pos = origin_input->getRelativeComputeAtAxis();
        origin_input->computeAt(producer, origin_ca_pos);
        producer->setComputeAt(consumer, origin_rel_ca_pos);
      }
    }
  }

  return producer;
}

TensorView* TensorView::cache_after() {
  FusionGuard fg(fusion());

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

  // Expr* consumer_origin =
  new UnaryOp(UnaryOpType::Set, consumer, producer);

  // Before: This TV -> Next TV
  // After:  This TV -> New TV (After) -> Next TV
  if (hasComputeAt()) {
    TransformReplay::replayCasP(consumer, producer, -1);

    auto rel_ca_pos = getRelativeComputeAtAxis();
    auto this_ca_pos = getThisComputeAtAxis();
    auto this_ca_view = getComputeAtView();

    computeAt(consumer, this_ca_pos);
    consumer->setComputeAt(this_ca_view, rel_ca_pos);
  } else {
    // Check users of this TV for computeAt for cache_after on inputs
    for (const auto& expr : fusion()->unordered_uses(consumer)) {
      for (TensorView* output :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        if (output->hasComputeAt()) {
          TransformReplay::replayPasC(consumer, output, -1);
          auto output_ca_pos = output->getThisComputeAtAxis();
          consumer->setComputeAt(output, output_ca_pos);
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
    new BroadcastOp(consumer_, broadcast_expr->in());
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
    if (binary_expr->lhs()->sameAs(current_)) {
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
    if (ternary_expr->in1()->sameAs(current_)) {
      new TernaryOp(
          ternary_expr->getTernaryOpType(),
          ternary_expr->out(),
          producer_,
          ternary_expr->in2(),
          ternary_expr->in3());
    } else if (ternary_expr->in2()->sameAs(current_)) {
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
    new BroadcastOp(broadcast_expr->out(), producer_);
  }

 private:
  TensorView* current_ = nullptr;
  TensorView* producer_ = nullptr;
};

} // namespace

// In Cache Before, for the origin expr of the original tensor,
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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
