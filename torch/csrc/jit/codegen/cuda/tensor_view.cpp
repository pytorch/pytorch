#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
// #include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>

// Cleanup
// #include <torch/csrc/jit/codegen/cuda/mutator.h>
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

TensorView* TensorView::computeAt(TensorView* consumer, int axis) {
  // Make sure this and consumer are not the same tensor, that's illegal
  TORCH_CHECK(
      !this->sameAs(consumer), "Cannot call this->computeAt(this, ...)");

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

TensorView* TensorView::split(int axis, unsigned int factor) {
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
