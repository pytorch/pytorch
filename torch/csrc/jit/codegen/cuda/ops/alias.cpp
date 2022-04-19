#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ops/alias.h>
#include <torch/csrc/jit/codegen/cuda/transform_view.h>

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

  // TODO: support view(dtype) for dtypes of different size.
  TORCH_INTERNAL_ASSERT(
      dataTypeSize(x->getDataType().value()) == dataTypeSize(dtype),
      "Currently, aten::view only supports viewing the data as a type with the same size.");

  std::vector<IterDomain*> out_domain;
  auto inp_domain = TensorDomain::noReductions(x->getMaybeRFactorDomain());
  out_domain.reserve(inp_domain.size());
  for (auto d : inp_domain) {
    out_domain.push_back(d->clone());
  }
  auto out = IrBuilder::create<TensorView>(
      x->container(),
      IrBuilder::create<TensorDomain>(
          out_domain, std::vector<bool>(out_domain.size(), true)),
      dtype);

  IrBuilder::create<ViewDtypeOp>(x->container(), out, x, dtype);
  return out;
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
      ? sum(x, analyze_view.trivial_reduction_axes)
      : x;

  auto view = (!analyze_view.transforms.empty())
      ? applyViewTransforms(reduction, analyze_view.transforms)
      : reduction;

  return (analyze_view.has_broadcast)
      ? broadcast(view, analyze_view.broadcast_axes)
      : view;
}

TensorView* squeeze(TensorView* x, const std::vector<int64_t>& sizes) {
  TORCH_INTERNAL_ASSERT(x->nDims() == sizes.size());

  std::vector<int> trivial_reduction_axes;
  for (const auto idx : c10::irange(sizes.size())) {
    if (sizes[idx] == 1) {
      trivial_reduction_axes.push_back(idx);
    }
  }
  return (trivial_reduction_axes.empty()) ? x : sum(x, trivial_reduction_axes);
}

TensorView* squeeze(TensorView* x, const std::vector<int64_t>& sizes, int dim) {
  TORCH_INTERNAL_ASSERT(x->nDims() == sizes.size());
  if (dim < 0) {
    dim = (int)(x->nDims()) + dim;
  }
  if (dim >= 0 && dim < x->nDims() && sizes[dim] == 1) {
    return sum(x, {dim});
  } else {
    return set(x);
  }
}

TensorView* unsqueeze(TensorView* x, int dim) {
  if (dim < 0) {
    dim = (int)(x->nDims()) + dim + 1;
  }
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim <= x->nDims());

  std::vector<bool> broadcast_axes(x->nDims() + 1, false);
  broadcast_axes[dim] = true;
  return broadcast(x, broadcast_axes);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
