#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/compute_at.h>
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
      memory_type_(mtype) {
  // Don't do this after transforms
  if (domain_->domain() == domain_->getRootDomain()) {
    // Mark the size-1 axes as broadcast to support implicit broadcast semantic
    for (auto* id : domain_->domain()) {
      if (!id->isBroadcast() && !id->isReduction() && !id->isGather() &&
          id->extent()->isOneInt()) {
        id->convertToBroadcast();
      }
    }
  }
}

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
      sizes.push_back(IrBuilder::create<IterDomain>(
          passkey.ir_container_->zeroVal(),
          passkey.ir_container_->oneVal(),
          ParallelType::Serial,
          IterType::BroadcastWithStride));
    } else {
      sizes.push_back(IrBuilder::create<IterDomain>(
          passkey.ir_container_->zeroVal(), IrBuilder::create<Int>()));
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
            new_root_domain[idx] = IrBuilder::create<IterDomain>(
                container(),
                id->start(),
                replacement_extents[idx],
                id->stopOffset(),
                id->getParallelType(),
                id->getIterType());
            ++idx;
          } else {
            TORCH_INTERNAL_ASSERT(!id->isRFactorProduct());
            new_root_domain[idx++] = id->clone();
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
      cpu_scalar_(src->cpu_scalar_) {
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
  TensorView* producer =
      IrBuilder::create<TensorView>(producer_domain, getDataType().value());

  // Set domain of consumer
  setDomain(consumer_domain);
  TensorView* consumer = this;

  // Setup dependency chain, inserting producer before this op.
  // Expr* producer_definition =
  IrBuilder::create<ReductionOp>(
      this_definition->getReductionOpType(),
      this_definition->init(),
      producer,
      this_definition->in());

  // Expr* consumer_definition =
  IrBuilder::create<ReductionOp>(
      this_definition->getReductionOpType(),
      this_definition->init(),
      consumer,
      producer);

  return producer;
}

TensorView* TensorView::welfordRfactorHelper(
    TensorView* tv,
    const std::vector<int>& axes) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
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
    for (const auto i : c10::irange(root.size())) {
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

WelfordResult TensorView::rFactor(
    const std::vector<int>& axes,
    TensorView* avg,
    TensorView* var,
    TensorView* n) {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
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

  std::vector<std::pair<TensorView*, TensorView*>> tv2rf{
      {avg, nullptr}, {var, nullptr}, {n, nullptr}};

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

  TensorView* producer_avg = tv2rf[0].second;
  TensorView* producer_var = tv2rf[1].second;
  TensorView* producer_n = tv2rf[2].second;

  // Setup dependency chain, inserting producer before this op.
  // Expr* producer_definition =
  IrBuilder::create<WelfordOp>(
      producer_avg,
      producer_var,
      producer_n, /*out var/avg/count */
      wop->initAvg(),
      wop->initVar(),
      wop->initN(), /*init var/avg/count */
      wop->inAvg(),
      wop->inVar(),
      wop->inN());

  // Expr* consumer_definition =
  IrBuilder::create<WelfordOp>(
      avg,
      var,
      n,
      wop->initAvg(),
      wop->initVar(),
      wop->initN(),
      producer_avg,
      producer_var,
      producer_n);

  return WelfordResult(producer_avg, producer_var, producer_n);
}

TensorView* TensorView::cache_before() {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  TORCH_CHECK(
      definition() != nullptr && !isFusionInput(),
      "Error adding cache_before ",
      this,
      " its definition is a nullptr and we restrict using cache_before on an input.");

  TORCH_CHECK(
      isFusionOutput() ||
          definition()->getExprType() != ExprType::ReductionOp ||
          definition()->getExprType() != ExprType::WelfordOp,
      "Error adding cache_before ",
      this,
      " its definition is a reduction and it is not an output, instead please use cache_after.");

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
    new_root_domain[i++] = dom->clone();
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
  IrBuilder::create<UnaryOp>(container(), UnaryOpType::Set, consumer, producer);

  // definition_ is no longer valid
  // setDefinition(nullptr);

  auto replayed_consumer_pair =
      TransformReplay::replayCasP(consumer, producer, -1);
  consumer->setDomain(replayed_consumer_pair.first);

  return producer;
}

TensorView* TensorView::cache_fork() {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  // Before: [Expr] -> This TV (Global Output) -> [Usage Expr]
  // After:  [Expr] -> This TV (Local) -> [Usage Expr] > Next TV
  //                            (Fork) -> [Set Expr]   -> New TV (Global Output)

  TORCH_CHECK(
      this->isFusionOutput() && !this->uses().empty(),
      "Error adding cache_fork ",
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

TensorView* TensorView::cache_after() {
  TORCH_INTERNAL_ASSERT(
      !container()->isA<kir::Kernel>(),
      "Function invalid for kernel container.");
  FusionGuard fg(fusion());

  // Get all the uses for this Tensorview
  TORCH_CHECK(
      !isFusionOutput(),
      "Error adding cache_after ",
      this,
      " we restrict using cache_after on an output.");

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
    new_root_domain[i++] = dom->clone();
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
  IrBuilder::create<UnaryOp>(container(), UnaryOpType::Set, consumer, producer);

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

bool TensorView::isEmptyTensor() const {
  auto& root_domain = getMaybeRFactorDomain();
  return std::all_of(
      root_domain.begin(), root_domain.end(), [](IterDomain* id) {
        return id->extent()->isZeroInt();
      });
}

void TensorView::applyMmaSwizzle(MmaOptions options) {
  switch (options.operand) {
    case MmaOptions::Operand::NotOperand:
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
  for (const auto i : c10::irange(ndims_)) {
    if (shape_.empty() || shape_[i] == -1) {
      domain[i] = IrBuilder::create<IterDomain>(
          FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>());
    } else {
      TORCH_CHECK(
          shape_[i] >= 0,
          "Invalid extent value. ",
          "For a tensor representing a single scalar use ndims = 0 with no sizes set.");
      if (shape_[i] == 1) {
        // If size is known to be 1, assume it needs to be broadcasted.
        domain[i] = IrBuilder::create<IterDomain>(
            FusionGuard::getCurFusion()->zeroVal(),
            FusionGuard::getCurFusion()->oneVal(),
            ParallelType::Serial,
            IterType::BroadcastWithStride);
      } else {
        domain[i] = IrBuilder::create<IterDomain>(
            FusionGuard::getCurFusion()->zeroVal(),
            IrBuilder::create<Int>(shape_[i]));
      }
    }
  }

  // Create the final TensorView
  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(domain, contiguity_), dtype_);
}

void TensorView::configureMma(MmaOptions options) {
  TORCH_CHECK(definition(), "configureMma: invalid for input tensor ", this);
  auto mma = dynamic_cast<MmaOp*>(definition());
  TORCH_CHECK(mma, "configureMma: invalid for non-mma output: ", this);
  mma->configureOptions(options);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
