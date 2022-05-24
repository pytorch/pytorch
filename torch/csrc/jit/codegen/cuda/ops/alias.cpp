#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ops/alias.h>
#include <torch/csrc/jit/codegen/cuda/transform_view.h>
#include <torch/csrc/jit/codegen/cuda/type_promotion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Transform TensorView according to keep, merge, and split transformations.
//! Trivial reduction and broadcast transformations are handled separately.
//! It is recommend to use the composite ops view function, which will call
//! the analyzeView function to generate the appropriate transformations.
//!
//! For example:
//! original sizes = [2, 10, 40]
//! new_size = [2, 10, 2, 20]
//! auto analysis = analyzeView(TV0, original_sizes, new_sizes)
//! auto TV1 = TV0->view(analysis.transforms);
//!
//! Transforms = [(Keep I0), (Keep I1), (Split I2 by 2)]
//! Before: TV0[I0, I1, I2]
//! After: TV0[I0, I1, 2, ceilDiv(I2, 2)]
//!
TensorView* applyViewTransforms(
    TensorView* tv,
    const std::vector<std::shared_ptr<ViewTransform>>& transforms) {
  TORCH_INTERNAL_ASSERT(
      !tv->hasComputeAt(),
      "Cannot modify rfactor domain after compute at has been set.");

  TORCH_INTERNAL_ASSERT(tv->nDims() > 0, "Tried to view a 0-dim TensorView");

  TORCH_CHECK(
      !tv->domain()->hasRFactor(),
      "Cannot call view on the same TensorView twice.");

  TORCH_INTERNAL_ASSERT(!transforms.empty());

  TensorView* consumer = IrBuilder::create<TensorView>(
      tv->container(),
      tv->domain()->view(transforms),
      tv->getDataType().value());

  IrBuilder::create<ViewOp>(tv->container(), consumer, tv);

  return consumer;
}

} // namespace

TensorView* view(TensorView* x, DataType dtype) {
  if (x->getDataType() == dtype) {
    return x;
  }

  auto input_type = x->getDataType().value();
  auto input_size = dataTypeSize(input_type);
  auto newsize = dataTypeSize(dtype);

  if (input_size == newsize) {
    return bitCastOp(dtype, x);
  }
  // TODO: support view(dtype) for dtypes where input_size != newsize
  TORCH_INTERNAL_ASSERT(false, "Unsupported reinterpret casting view");
}

TensorView* view(
    TensorView* x,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  TORCH_INTERNAL_ASSERT(
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size() ==
      original_sizes.size());

  auto analyze_view = analyzeView(x, original_sizes, new_sizes);

  auto reduction = (!analyze_view.trivial_reduction_axes.empty())
      ? sum(x,
            analyze_view.trivial_reduction_axes,
            false /* keep_dim */,
            x->getDataType().value())
      : x;

  auto view = (!analyze_view.transforms.empty())
      ? applyViewTransforms(reduction, analyze_view.transforms)
      : reduction;

  return (analyze_view.has_broadcast)
      ? broadcast(view, analyze_view.broadcast_axes)
      : view;
}

TensorView* flatten(TensorView* x, int64_t start_dim, int64_t end_dim) {
  if (start_dim < 0) {
    start_dim += x->nDims();
  }
  if (end_dim < 0) {
    end_dim += x->nDims();
  }
  TORCH_CHECK(
      start_dim >= 0 && start_dim < x->nDims(),
      "Invalid start_dim ",
      start_dim);
  TORCH_CHECK(
      end_dim >= 0 && end_dim < x->nDims(), "Invalid end_dim ", end_dim);
  TORCH_CHECK(start_dim <= end_dim, "start_dim must be <= end_dim");

  if (start_dim == end_dim) {
    return x;
  }

  auto out = IrBuilder::create<TensorView>(
      x->container(),
      x->domain()->flatten(start_dim, end_dim),
      x->getDataType().value());

  IrBuilder::create<ViewOp>(out, x);
  return out;
}

TensorView* squeeze(TensorView* x, const std::vector<int64_t>& sizes) {
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_INTERNAL_ASSERT(
      ndims == sizes.size(),
      "Invalid sizes for squeeze: ",
      sizes,
      ". Input tensor: ",
      x->toString());

  std::vector<int> trivial_reduction_axes;
  for (const auto idx : c10::irange(sizes.size())) {
    if (sizes[idx] == 1) {
      trivial_reduction_axes.push_back(idx);
    }
  }
  return (trivial_reduction_axes.empty()) ? x
                                          : sum(x,
                                                trivial_reduction_axes,
                                                false /* keep_dim */,
                                                x->getDataType().value());
}

TensorView* squeeze(TensorView* x, const std::vector<int64_t>& sizes, int dim) {
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_INTERNAL_ASSERT(
      ndims == sizes.size(),
      "Invalid sizes for squeeze: ",
      sizes,
      ". Input tensor: ",
      x->toString());

  if (dim < 0) {
    dim = ndims + dim;
  }

  TORCH_INTERNAL_ASSERT(
      dim >= 0 && dim < ndims,
      "Invalid position to squeeze: ",
      dim,
      ". Input tensor: ",
      x->toString());

  if (sizes[dim] == 1) {
    return sum(x, {dim}, false /* keep_dim */, x->getDataType().value());
  } else {
    return set(x);
  }
}

TensorView* unsqueeze(TensorView* x, int dim) {
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  if (dim < 0) {
    dim = ndims + dim + 1;
  }

  TORCH_INTERNAL_ASSERT(
      dim >= 0 && dim <= ndims,
      "Invalid position to unsqueeze: ",
      dim,
      ". Input tensor: ",
      x->toString());

  std::vector<bool> broadcast_axes(ndims + 1, false);
  broadcast_axes[dim] = true;
  return broadcast(x, broadcast_axes);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
