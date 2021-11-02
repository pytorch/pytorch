#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <ATen/TensorGeometry.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <memory>

namespace torch { namespace autograd {

struct TORCH_API CopyBackwards : public Node {
  variable_list apply(variable_list&& grads) override;

  at::TensorOptions src_options;
};

// Note [View + Inplace update for base tensor]
// Performs grad_view = fn(grad_view), but out-of-place.
// view_fn_ is an optional lambda function saved in DifferentiableViewMeta
// from forward pass, so that we can recover we when as_strided is not supported.
// It preserves the invariants:
//   view = view_fn_(base)
//   grad_view = view_fn_(grad_base)
//
// When as_strided is supported (e.g. strided CPU/CUDA Tensors), view_fn_
// is empty and we save TensorGeometry(view) instead.
// With the TensorGeometry information we can use `as_strided` call which
// is more efficient to recover views in backward.
//
// For example:
//   view_1 = view_op_1(base)
//   view_2 = view_op_2(view_1)
//   ...
//   view_n = view_op_n(view_n-1)
//   view_n = inplace_op(view_n)
//
// In CPU/CUDA case where we support efficient as_strided implementation,
// grad_view_n can be calculated through 1 step.
//
//   grad_view_n = grad_base.as_strided(view_sizes, view_strides, view_offset);
//
// But in XLA backend where we don't have full support of as_strided,
// it has to save a chained lambda function view_fn_, to exactly
// replay how the view was done in forward.
//
//   view_fn_ = view_op_n(...(view_op_2(view_op_1())))
//   grad_view_n = view_fn_(grad_base)
//
// This chain view_fn_ works as long as forward view ops are implemented,
// e.g XLA simulates view without a real Storage behind Tensor, but it's less
// efficient than the as_strided one so we should be careful to only use it when
// necessary.
//
// What do we use in CopySlices backward?
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//   - For CPU/CUDA we save TensorGeometry of both base and view tensors,
//     That's all we need to pass into as_strided.
//     E.g. int[] sizes, int[] strides, and int storage_offset.
//   - For XLA we use view_fn_, which captures all forward view op arguments
//     by **value**.
//     E.g for at::narrow, int dim, int start, in length are saved.
//
// Theorectically we could also save Tensor `view` in CopySlices Node, but
// it's far more expensive than what we currently save.
//   1. We cannot afford keeping large tensors alive to recover views only.
//   2. There are inplace checks when Tensors are loaded back to make sure
//      they haven't been changed (including size metadata).
// So saving metadata like TensorGeometry/view arguments is much better
// because it is minimal information needed to recover views, as well as it
// allows the user to modify the original Tensor without preventing the
// backward pass from running.
//
// When an in-place operation is done on a differentiable view, the base's
// grad_fn is updated to become a `CopySlice` wrapping the backward of the
// in-place operation.

// See Note [View + Inplace update for view tensor] for what we do to view
// tensor when an in-place operation happens.
struct TORCH_API CopySlices : public Node {
  CopySlices(
      const Variable& base_var,
      at::TensorGeometry view_,
      std::function<at::Tensor(const at::Tensor&)> view_fn_,
      std::shared_ptr<Node> fn_);

  variable_list apply(variable_list&& inputs) override;
  void release_variables() override;

  at::TensorGeometry base;
  // view and view_fn are redundant and view_fn will be used if available.
  // See Note [View + Inplace update for base tensor] for details.
  at::TensorGeometry view;
  std::function<at::Tensor(const at::Tensor&)> view_fn;
  std::shared_ptr<Node> fn;
};

}}
