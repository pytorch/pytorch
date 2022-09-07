#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_double_buffer.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/mma_utils.h>

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

TensorView::TensorView(
    IrBuilderPasskey passkey,
    TensorDomain* domain,
    DataType dtype,
    MemoryType mtype)
    : Val(passkey, ValType::TensorView, dtype),
      domain_(domain),
      memory_type_(mtype) {}

TensorView::TensorView(
    IrBuilderPasskey passkey,
    const std::shared_ptr<c10::TensorType>& tensor_type)
    : Val(passkey,
          ValType::TensorView,
          aten_opt_type_map(tensor_type->scalarType())) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  std::vector<IterDomain*> sizes;

  TORCH_CHECK(
      tensor_type->dim().has_value(), "Requires static rank for Tensor");

  for (const auto i : c10::irange(tensor_type->dim().value())) {
    if (tensor_type->sizes()[i].has_value() &&
        tensor_type->sizes()[i].value() == 1) {
      // If size is known to be 1, assuem it needs to be broadcasted.
      sizes.push_back(
          IterDomainBuilder(
              passkey.ir_container_->zeroVal(), passkey.ir_container_->oneVal())
              .iter_type(IterType::Broadcast)
              .build());
    } else {
      sizes.push_back(
          IterDomainBuilder(
              passkey.ir_container_->zeroVal(), IrBuilder::create<Int>())
              .build());
    }
  }
  // [ Note -- stride_properties in tensor type ]
  //
  // `stride_properties()` returns a vector<optional<Stride>>, while
  //     Stride {
  //       optional<size_t> stride_index_;
  //       optional<bool> contiguous_;
  //       optional<size_t> stride_;
  //     };
  // To keep things simple, we ignore all the optional wrapper, as in reality,
  // they would always be available unless we start doing multiple profiling
  // runs.
  //
  //   `stride_properties()` returns the vector of Stride, where it is ordered
  //   from the fastest to slowest dimensions. i.e. stride_properties()[i] would
  //   give us the i-th fastest dimension. where:
  //     1. `Stride::stride_index_` gives the index to the dimension;
  //     2. `Stride::contiguous_` indicates whether this dimension is
  //     memory-dense*;
  //     3. `Stride::stride_` is the actual stride for the given dimension.
  // * note that memory-dense means different things depending on the order of
  // the dimension. checkout `TensorType::computeStrideProps` for details

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
        // check the neighboring faster dimension, collapse if it is considered
        // as inner dimension per stride_index
        auto inner_index_opt =
            tensor_type->stride_properties()[static_cast<int>(i) - 1]
                ->stride_index_;
        if (inner_index_opt.has_value() &&
            inner_index_opt.value() == (index + 1)) {
          // collapse if inner dimension has non-broadcasted strides
          auto inner_stride_opt =
              tensor_type->stride_properties()[static_cast<int>(i) - 1]
                  ->stride_;
          contig_info[index] =
              inner_stride_opt.has_value() && inner_stride_opt.value() != 0;
        }
      }
    }
  }

  domain_ = IrBuilder::create<TensorDomain>(sizes, contig_info);
}

TensorView::TensorView(
    IrBuilderPasskey passkey,
    const std::shared_ptr<Value>& jit_value)
    : TensorView(passkey, jit_value->type()->cast<c10::TensorType>()) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
}

void TensorView::convertRfactorToRootDomain() {
  // For a given TensorView, does its domain (root / rfactor) contain any
  // concrete sized extents?
  auto is_concrete_tensor = [](TensorView* tv) {
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (!id->extent()->isConstScalar()) {
        return false;
      }
    }
    return true;
  };

  // Create a new root domain and replacement TensorDomain.
  // Given an rfactor domain, create a new IterDomain.
  // Otherwise, clone the previous IterDomain
  auto createReplacementDomain =
      [this](const std::vector<Val*>& replacement_extents) {
        TORCH_INTERNAL_ASSERT(
            !replacement_extents.empty() &&
            getMaybeRFactorDomain().size() == replacement_extents.size());
        size_t idx = 0;
        std::vector<IterDomain*> new_root_domain(
            getMaybeRFactorDomain().size());
        for (const auto& id : getMaybeRFactorDomain()) {
          if (replacement_extents[idx] != nullptr) {
            new_root_domain[idx] = IterDomainBuilder(id)
                                       .extent(replacement_extents[idx])
                                       .resetSchedulingParams()
                                       .build();
            ++idx;
          } else {
            TORCH_INTERNAL_ASSERT(!id->isRFactorProduct());
            new_root_domain[idx++] = id->cloneWithoutRFactor();
          }
        }

        TORCH_INTERNAL_ASSERT(
            new_root_domain.size() == domain()->contiguity().size());
        setDomain(IrBuilder::create<TensorDomain>(
            container(), new_root_domain, domain()->contiguity()));
      };

  std::vector<Val*> rfactor_extents;
  std::unordered_map<Val*, Val*> replacement_map;
  const auto kThisIsConcreteTensor = is_concrete_tensor(this);
  for (const auto& id : getMaybeRFactorDomain()) {
    if (id->isRFactorProduct()) {
      // Create new symbolic extents for rfactor iterDomains
      auto domain_extent = (!kThisIsConcreteTensor)
          ? IrBuilder::create<Int>(container())
          : id->extent();
      rfactor_extents.push_back(domain_extent);
      replacement_map.emplace(id->extent(), domain_extent);
    } else {
      rfactor_extents.push_back(nullptr);
    }
  }
  createReplacementDomain(rfactor_extents);

  // Propagate new extent throughout fusion using ValReplacementMutator
  ir_utils::replaceValue(fusion(), replacement_map);
}

TensorView::TensorView(const TensorView* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      domain_(ir_cloner->clone(src->domain_)),
      compute_at_pos_(src->compute_at_pos_),
      max_producer_pos_(src->max_producer_pos_),
      memory_type_(src->memory_type_),
      swizzle_type_(src->swizzle_type_),
      is_double_buffered_(src->is_double_buffered_),
      is_circular_buffered_(src->is_circular_buffered_),
      circular_buffer_stage_(src->circular_buffer_stage_),
      cpu_scalar_(src->cpu_scalar_),
      has_swizzle_op_(src->has_swizzle_op_) {
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

// sets cpu_scalar_ value, which is special handling for CPU based zero-dim
// tensors (i.e. CPU Tensors that only have one value). This is only used if
// on an input value, otherwise ignored. This is important as special handling
// because these "scalars" should be type promoted as a tensor, but we want to
// avoid explicit copying of the data, so we want to pass the data value as a
// standard kernel argument value.
void TensorView::setCpuScalar(bool is_cpu_scalar) {
  TORCH_INTERNAL_ASSERT(
      nDims() == 0, "Only 0-dim tensors can be marked as a cpu scalar.");
  cpu_scalar_ = is_cpu_scalar;
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

void TensorView::setComputeAt(unsigned int pos, bool decrease) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  if (pos <= compute_at_pos_ && !decrease) {
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

void TensorView::setMaxProducer(unsigned int pos, bool decrease) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  if (pos <= max_producer_pos_ && !decrease) {
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
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
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
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
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

TensorView* TensorView::split(
    int axis_,
    Val* factor,
    bool inner_split,
    bool trim_out_of_bounds) {
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

  domain()->split(axis_, factor, inner_split, trim_out_of_bounds);
  return this;
}

TensorView* TensorView::split(
    int axis,
    unsigned int factor,
    bool inner_split,
    bool trim_out_of_bounds) {
  split(axis, IrBuilder::create<Int>(factor), inner_split, trim_out_of_bounds);
  return this;
}

// Merge "axis_o" and "axis_i" into 1 dimension
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
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
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
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
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

TensorView* TensorView::swizzle(
    Swizzle2DType swizzle_type,
    int x,
    int y,
    SwizzleMode swizzle_mode) {
  has_swizzle_op_ = true;
  if (x < 0) {
    x += domain()->nDims();
  }
  if (y < 0) {
    y += domain()->nDims();
  }

  TORCH_CHECK(
      x >= (int)getComputeAtPosition(),
      false,
      "Cannot swizzle axes within compute at position. Axis ",
      x,
      " is within computeAtPosition = ",
      getComputeAtPosition());

  TORCH_CHECK(
      y >= (int)getMaxProducerPosition(),
      "Cannot swizzle axes within max producer position. Axis ",
      y,
      " is within maxProducerPosition = ",
      getMaxProducerPosition());

  // Disable unsupported use cases at the current step.
  //  Currently do not support reducing or broadcasting
  //   swizzled dimensions.
  auto all_inputs = InputsOf::outputs(fusion(), {axis(x), axis(y)});
  for (auto id : ir_utils::filterByType<IterDomain>(all_inputs)) {
    TORCH_INTERNAL_ASSERT(
        !id->isBroadcast() && !id->isReduction(),
        "Unsupported use case for swizzle.");
  }

  // Also checking that the scheduler is not trying to
  //  compose swizzles, which is not yet supported either.
  auto all_exprs = DependencyCheck::getAllValsBetween(
      {all_inputs.begin(), all_inputs.end()}, {axis(x), axis(y)});
  for (auto expr : all_exprs) {
    TORCH_INTERNAL_ASSERT(
        !expr->isA<Swizzle2D>(), "Composing swizzles is not yet supported");
  }

  // Check swizzle specific constraints on the input axes:
  if (swizzle_type != Swizzle2DType::ZShape) {
    ExpressionEvaluator const_eval(fusion());

    auto x_id = axis(x);
    auto y_id = axis(y);

    TORCH_INTERNAL_ASSERT(
        x_id->extent()->isConstInt() && y_id->extent()->isConstInt(),
        "Only constant iterdomains supported on given swizzle type");

    int in_x_size = x_id->extent()->evaluateInt();
    int in_y_size = y_id->extent()->evaluateInt();

    // Check size constraints based on swizzle type
    if (swizzle_type == Swizzle2DType::Transpose ||
        swizzle_type == Swizzle2DType::XOR) {
      TORCH_INTERNAL_ASSERT(
          in_x_size == in_y_size, "Swizzle: equal dim iterdomains only");
    }

    if (swizzle_type == Swizzle2DType::Scatter) {
      TORCH_INTERNAL_ASSERT(
          in_y_size == 4, "Swizzle: unsupported id size must be 4 ", in_y_size);
      TORCH_INTERNAL_ASSERT(
          in_x_size == 8 || in_x_size == 16 || in_x_size == 32,
          "Swizzle: unsupported id size must be 8, 16, or 32 ",
          in_x_size);
    }
  }

  domain()->swizzle(swizzle_type, x, y, swizzle_mode);

  return this;
}

TensorView* TensorView::rFactor(const std::vector<int>& axes) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  // TODO: I think we should do this but
  // NVFuserTest.FusionSmemBlockGemmCache_CUDA prevents it from going in at the
  // moment.

  // TORCH_INTERNAL_ASSERT(
  //     !hasComputeAt(), "Cannot rfactor tensors after compute at has been
  //     set.");
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to rFactor a 0-dim TensorView");
  FusionGuard fg(fusion());
  TORCH_CHECK(
      definition() != nullptr &&
          (definition()->getExprType() == ExprType::ReductionOp ||
           definition()->getExprType() == ExprType::MmaOp),
      "Error rfactoring ",
      this,
      " its definition is either a nullptr or not a reduction.");
  TORCH_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  TORCH_CHECK(
      !definition()->isA<GroupedReductionOp>(),
      "For GroupedReducitonOp, use TensorView::rFactor(const std::vector<int>& axes, const std::vector<TensorView*>& tvs)");

  // Split tensor view into 2 parts
  auto domain_pair = domain()->rFactor(axes);

  // Producer in the pair
  auto producer_domain = domain_pair.first;
  // Consumer in the pair
  auto consumer_domain = domain_pair.second;

  // This domain will be the consumer, so create the producer
  TensorView* producer =
      IrBuilder::create<TensorView>(producer_domain, getDataType().value());

  // Set domain of consumer
  setDomain(consumer_domain);
  TensorView* consumer = this;

  if (auto this_reduction = dynamic_cast<ReductionOp*>(definition())) {
    // Setup dependency chain, inserting producer before this op.
    // Expr* producer_definition =
    IrBuilder::create<ReductionOp>(
        this_reduction->getReductionOpType(),
        this_reduction->init(),
        producer,
        this_reduction->in());

    // Expr* consumer_definition =
    IrBuilder::create<ReductionOp>(
        this_reduction->getReductionOpType(),
        this_reduction->init(),
        consumer,
        producer);
  } else if (auto this_mma = dynamic_cast<MmaOp*>(definition())) {
    // Initial reduction that still uses mma to combine
    //  the input.
    IrBuilder::create<MmaOp>(
        producer,
        this_mma->inA(),
        this_mma->inB(),
        this_mma->init(),
        this_mma->options());

    // Remaining reduction that can be scheduled cross
    //  warp or cta.
    IrBuilder::create<ReductionOp>(
        BinaryOpType::Add, this_mma->init(), consumer, producer);
  } else {
    TORCH_INTERNAL_ASSERT(false, "RFactor: unsupported tensor definition");
  }
  return producer;
}

TensorView* TensorView::multiOutputRfactorHelper(
    TensorView* tv,
    const std::vector<int>& axes) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  // Hack:
  // Semantically we should always keep the outputs of multi reduction ops
  // scheduled the same but the user end cannot guarantee that. In order to
  // guarantee that the rFactor is defined meaningfully the scheduling of the
  // output TV that got the rfactor call is force replayed towards the other two

  if (!sameAs(tv)) {
    auto root = tv->getRootDomain();
    auto this_root = getRootDomain();

    // construct a trivial root domain map
    std::unordered_map<IterDomain*, IterDomain*> id_map;
    for (const auto i : c10::irange(root.size())) {
      id_map[this_root[i]] = root[i];
    }

    // replay on the target tv
    ReplayTransformations replay(domain()->domain(), id_map);

    // construct the new tensor domain
    std::vector<IterDomain*> new_id;
    for (auto id : domain()->domain()) {
      TORCH_INTERNAL_ASSERT(
          replay.getReplay().count(id), "Multi-output reduction replay failed");
      new_id.push_back(replay.getReplay().at(id));
    }

    std::vector<bool> new_contig(
        tv->domain()->contiguity().begin(), tv->domain()->contiguity().end());
    // replace tensor domain of target tv
    tv->setDomain(IrBuilder::create<TensorDomain>(
        tv->getRootDomain(), new_id, new_contig));
  }

  // Split tensor view into 2 parts
  auto domain_pair = tv->domain()->rFactor(axes);
  // Producer in the pair
  auto producer_domain = domain_pair.first;
  // Consumer in the pair
  auto consumer_domain = domain_pair.second;

  // This domain will be the consumer, so create the producer
  TensorView* producer =
      IrBuilder::create<TensorView>(producer_domain, tv->getDataType().value());

  // Set domain of consumer
  tv->setDomain(consumer_domain);

  return producer;
}

std::vector<TensorView*> TensorView::rFactor(
    const std::vector<int>& axes,
    const std::vector<TensorView*>& tvs) {
  TORCH_CHECK(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  TORCH_CHECK(nDims() > 0, "Tried to rFactor a 0-dim TensorView");
  FusionGuard fg(fusion());
  TORCH_CHECK(
      definition() != nullptr && ir_utils::isReductionOp(definition()),
      "Error rfactoring multi-output reduction op ",
      this,
      " its definition is either a nullptr or not a GroupedReductionOp or a multi-output reduction op.");

  TORCH_CHECK(
      !domain()->hasRFactor(), "Cannot call rfactor on the same view twice.");

  TORCH_CHECK(
      definition()->outputs().size() == tvs.size(),
      "Rfactor of a multi-output reduction not used correctly");

  for (const auto i : c10::irange(tvs.size())) {
    TORCH_CHECK(
        definition()->output(i) == tvs.at(i),
        "Rfactor of a multi-output reduction not used correctly");
  }

  // Currently grouping of welford is only supported through
  // ParallelType::Group, so GroupedWelfordOp is only created during
  // the lowering time. As rFactor is done before lowering, there
  // should be no GroupedWelfordOp at this point.
  TORCH_INTERNAL_ASSERT(
      !definition()->isA<GroupedWelfordOp>(),
      "GroupedWelfordOp found: ",
      definition()->toString());

  std::vector<TensorView*> rf_tvs(tvs.size());

  // Make sure this gets rfactored last so everybody gets
  //  replayed correctly
  for (const auto i : c10::irange(tvs.size())) {
    if (this != tvs.at(i)) {
      rf_tvs.at(i) = multiOutputRfactorHelper(tvs.at(i), axes);
    }
  }

  for (const auto i : c10::irange(tvs.size())) {
    if (this == tvs.at(i)) {
      rf_tvs.at(i) = multiOutputRfactorHelper(tvs.at(i), axes);
    }
  }

  if (auto wop = dynamic_cast<WelfordOp*>(definition())) {
    TensorView* producer_avg = rf_tvs.at(0);
    TensorView* producer_var = rf_tvs.at(1);
    TensorView* producer_n = rf_tvs.at(2);

    // Setup dependency chain, inserting producer before this op.
    // Expr* producer_definition =
    IrBuilder::create<WelfordOp>(
        producer_avg,
        producer_var,
        producer_n,
        wop->inAvg(),
        wop->inVar(),
        wop->inN(),
        wop->initAvg(),
        wop->initVar(),
        wop->initN());

    // Expr* consumer_definition =
    IrBuilder::create<WelfordOp>(
        wop->outAvg(),
        wop->outVar(),
        wop->outN(),
        producer_avg,
        producer_var,
        producer_n,
        wop->initAvg(),
        wop->initVar(),
        wop->initN());
  } else if (
      auto grouped_rop = dynamic_cast<GroupedReductionOp*>(definition())) {
    IrBuilder::create<GroupedReductionOp>(
        grouped_rop->getReductionOpTypes(),
        grouped_rop->initVals(),
        std::vector<Val*>{rf_tvs.begin(), rf_tvs.end()},
        grouped_rop->inputs());

    IrBuilder::create<GroupedReductionOp>(
        grouped_rop->getReductionOpTypes(),
        grouped_rop->initVals(),
        grouped_rop->outputs(),
        std::vector<Val*>{rf_tvs.begin(), rf_tvs.end()});
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Invalid definition: ", definition()->toString());
  }

  return rf_tvs;
}

TensorView* TensorView::cacheBefore(c10::optional<LoadStoreOpType> cache_op) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  TORCH_CHECK(
      definition() != nullptr && !isFusionInput(),
      "Error adding cacheBefore ",
      this,
      " its definition is a nullptr and we restrict using cacheBefore on an input.");

  // Previously, caching computed-at tensors was allowed but was never
  // really robust. Make it an error unless it is really needed.
  TORCH_CHECK(
      !hasComputeAt(),
      "Caching computed-at tensors is not allowed. Apply caching before computeAt");

  // It also did additional transformation when a producer tensor has computeAt.
  // Make sure we no longer rely on that behavior.
  if (definition() != nullptr) {
    for (TensorView* producer_of_producer :
         ir_utils::filterByType<TensorView>(definition()->inputs())) {
      TORCH_CHECK(
          !producer_of_producer->hasComputeAt(),
          "Potentially invalid computeAt and caching detected. Apply caching before computeAt.");
    }
  }

  // Create Producer Domain
  // This domain will be the consumer which needs a new domain, so replace the
  // producers domain with this domain.

  TensorView* producer = IrBuilder::create<TensorView>(
      container(),
      IrBuilder::create<TensorDomain>(
          container(),
          domain()->getRootDomain(),
          domain()->getRFactorDomain(),
          domain()->domain(),
          domain()->contiguity()),
      getDataType().value());

  // Set domain of consumer
  TensorView* consumer = this;

  size_t i = 0;
  auto no_reduction_root_domain =
      TensorDomain::noReductions(getMaybeRFactorDomain());
  std::vector<IterDomain*> new_root_domain(no_reduction_root_domain.size());
  for (const auto& dom : no_reduction_root_domain) {
    new_root_domain[i++] = dom->cloneWithoutRFactor();
  }

  consumer->setDomain(IrBuilder::create<TensorDomain>(
      container(),
      new_root_domain,
      std::vector<bool>(new_root_domain.size(), true)));

  // Insert producer - Cache_Before (CB) - before this TV.
  // Before: Prev TV -> [Definition Op] -> This TV
  // After:  Prev TV -> [Definition Op] -> New CB TV -> [Set Op] -> This TV

  // Get inputs for origin expression
  auto expr_inputs = definition()->inputs();
  // Expr* producer_definition =
  ir_utils::replaceValInExpr(definition(), this, producer);

  // Expr* producer_uses =
  if (cache_op.has_value()) {
    IrBuilder::create<LoadStoreOp>(
        container(), cache_op.value(), consumer, producer);
  } else {
    IrBuilder::create<UnaryOp>(
        container(), UnaryOpType::Set, consumer, producer);
  }

  // definition_ is no longer valid
  // setDefinition(nullptr);

  auto replayed_consumer_pair =
      TransformReplay::replayCasP(consumer, producer, -1);
  consumer->setDomain(replayed_consumer_pair.first);

  return producer;
}

TensorView* TensorView::cacheFork() {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  // Before: [Expr] -> This TV (Global Output) -> [Usage Expr]
  // After:  [Expr] -> This TV (Local) -> [Usage Expr] > Next TV
  //                            (Fork) -> [Set Expr]   -> New TV (Global Output)

  TORCH_CHECK(
      this->isFusionOutput() && !this->uses().empty(),
      "Error adding cacheFork ",
      this,
      " this TensorView must be an output with subsequent uses");

  // Previously, caching computed-at tensors was allowed but was never
  // really robust. Make it an error unless it is really needed.
  TORCH_CHECK(
      !hasComputeAt(),
      "Caching computed-at tensors is not allowed. Apply caching before computeAt");

  // This domain will be the producer, so create the consumer
  auto root_domain = TensorDomain::noReductions(getMaybeRFactorDomain());
  TensorView* new_output = IrBuilder::create<TensorView>(
      container(),
      IrBuilder::create<TensorDomain>(
          container(),
          IterDomain::clone(root_domain),
          std::vector<bool>(root_domain.size(), true)),
      getDataType().value());

  // Create write operation from this TV to new output
  IrBuilder::create<UnaryOp>(container(), UnaryOpType::Set, new_output, this);

  // The new TV becomes an output.
  // New TV has global memory type.
  // This TV has local memory type.
  fusion()->replaceOutput(this, new_output);

  // Transform new output according to this TV
  auto replayed_output_pair = TransformReplay::replayCasP(new_output, this, -1);
  new_output->setDomain(replayed_output_pair.first);

  return new_output;
}

TensorView* TensorView::cacheAfter(c10::optional<LoadStoreOpType> cache_op) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  // Get all the uses for this Tensorview
  TORCH_CHECK(
      !isFusionOutput(),
      "Error adding cacheAfter ",
      this,
      " we restrict using cacheAfter on an output.");

  // Previously, caching computed-at tensors was allowed but was never
  // really robust. Make it an error unless it is really needed.
  TORCH_CHECK(
      !hasComputeAt(),
      "Caching computed-at tensors is not allowed. Apply caching before computeAt.");

  // It also did additional transformation when this tensor is an
  // input and the outputs of its consumers have computeAt. Make sure
  // we no longer rely on that behavior.
  if (isFusionInput()) {
    for (const auto& expr : uses()) {
      for (TensorView* output :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        TORCH_CHECK(
            !output->hasComputeAt(),
            "Potentially invalid computeAt and caching detected. Apply caching before computeAt.");
      }
    }
  }

  // Create Consumer Domain
  // Keep Broadcast Axis (Permanent)
  // Remove Reduction Axis
  size_t i = 0;
  auto no_reduction_root_domain =
      TensorDomain::noReductions(getMaybeRFactorDomain());
  std::vector<IterDomain*> new_root_domain(no_reduction_root_domain.size());
  for (const auto& dom : no_reduction_root_domain) {
    new_root_domain[i++] = dom->cloneWithoutRFactor();
  }

  // This domain will be the producer, so create the consumer
  TensorView* consumer = IrBuilder::create<TensorView>(
      container(),
      IrBuilder::create<TensorDomain>(
          container(),
          new_root_domain,
          std::vector<bool>(new_root_domain.size(), true)),
      getDataType().value());

  // Set domain of producer - No Change
  TensorView* producer = this;

  // Insert consumer - Cache_After (CA) - after this TV.
  // Before: This TV -> [Use Op] -> Next TV
  // After:  This TV -> [Set Op] -> New CA TV -> [Use Op] -> Next TV

  // Expr* consumer_uses =
  for (auto expr : fusion()->unordered_uses(this)) {
    ir_utils::replaceValInExpr(expr, this, consumer);
  }

  // Expr* consumer_definition =
  if (cache_op.has_value()) {
    IrBuilder::create<LoadStoreOp>(
        container(), cache_op.value(), consumer, producer);
  } else {
    IrBuilder::create<UnaryOp>(
        container(), UnaryOpType::Set, consumer, producer);
  }

  return consumer;
}

void TensorView::setMemoryType(MemoryType mt) {
  memory_type_ = mt;
  if (isFusionInput() || isFusionOutput()) {
    TORCH_INTERNAL_ASSERT(
        mt == MemoryType::Global,
        "Tried to set an input or output to the fusion to a non-global memory type.");
  }
}

void TensorView::clearReductionIterDomains() {
  TORCH_INTERNAL_ASSERT(
      !domain()->hasRFactor(),
      "should not call clearReductionIterDomains on rfactor tv");

  TORCH_INTERNAL_ASSERT(
      domain()->domain() == getRootDomain(),
      "should not call clearReductionIterDomains on already transformed TensorDomains");

  std::vector<IterDomain*> new_root;
  std::vector<bool> new_contig;
  for (const auto i : c10::irange(getRootDomain().size())) {
    if (!getRootDomain()[i]->isReduction()) {
      new_root.push_back(getRootDomain()[i]);
      new_contig.push_back(domain()->contiguity()[i]);
    }
  }

  setDomain(IrBuilder::create<TensorDomain>(container(), new_root, new_contig));
}

void TensorView::doubleBuffer() {
  // Early correctness checking. May miss eventual errors as the
  // checks depend on memory types and parallelization, which may not
  // be finalized until lowering.
  validateDoubleBufferedTensor(this);
  is_double_buffered_ = true;
}

void TensorView::circularBuffer(unsigned int stage) {
  // Early correctness checking. May miss eventual errors as the
  // checks depend on memory types and parallelization, which may not
  // be finalized until lowering.
  TORCH_INTERNAL_ASSERT(stage > 1, "Unsupported stage number");
  if (stage == 2) {
    // Re-direct to double buffer interface if stage is 2;
    doubleBuffer();
    return;
  }
  validateDoubleBufferedTensor(this);
  is_circular_buffered_ = true;
  circular_buffer_stage_ = stage;
}

bool TensorView::isEmptyTensor() const {
  auto& root_domain = getMaybeRFactorDomain();
  return std::all_of(
      root_domain.begin(), root_domain.end(), [](IterDomain* id) {
        return id->extent()->isZeroInt();
      });
}

void TensorView::applyMmaSwizzle(MmaOptions options) {
  switch (options.operand) {
    case MmaOptions::Operand::Accumulator:
      mma_util::WarpMmaSwizzler::scheduleMmaWarpOutput(this, options);
      break;
    case MmaOptions::Operand::A:
    case MmaOptions::Operand::B:
      mma_util::WarpMmaSwizzler::scheduleOperandRead(this, options);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown operand flag");
      break;
  }
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

TensorViewBuilder& TensorViewBuilder::shape(const std::vector<int64_t>& shape) {
  TORCH_CHECK(shape_.empty(), "Attempting to reset shape");
  if (!shape.empty()) {
    TORCH_CHECK(ndims_ == 0 || ndims_ == shape.size());
    ndims_ = shape.size();
  }
  shape_.clear();
  shape_.reserve(shape.size());
  for (int64_t i : shape) {
    if (i == -1) {
      shape_.emplace_back(IrBuilder::create<Int>());
    } else {
      TORCH_CHECK(
          i >= 0,
          "Invalid extent value. ",
          "For a tensor representing a single scalar use ndims = 0 with no sizes set.");
      shape_.emplace_back(IrBuilder::create<Int>(i));
    }
  }
  return *this;
}

TensorViewBuilder& TensorViewBuilder::shape(std::vector<Val*> shape) {
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
  for (const auto i : c10::irange(ndims_)) {
    if (shape_.empty()) {
      domain[i] =
          IterDomainBuilder(
              FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>())
              .build();
    } else {
      if (shape_[i]->isOneInt()) {
        // If size is known to be 1, assume it needs to be broadcasted.
        domain[i] = IterDomainBuilder(
                        FusionGuard::getCurFusion()->zeroVal(),
                        FusionGuard::getCurFusion()->oneVal())
                        .iter_type(IterType::Broadcast)
                        .build();
      } else {
        domain[i] =
            IterDomainBuilder(FusionGuard::getCurFusion()->zeroVal(), shape_[i])
                .build();
      }
    }
  }

  // Create the final TensorView
  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(domain, contiguity_), dtype_);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
