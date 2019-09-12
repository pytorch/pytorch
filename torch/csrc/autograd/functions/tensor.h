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
  at::Device src_device = at::kCPU;
};

// Performs grad[idx] = fn(grad[idx]), but out-of-place. The slicing operation
// grad[idx] is defined by the relative sizes, strides, and offset of base and
// view.
// When an in-place operation is done on a differentiable view, the base's
// grad_fn is updated to become a `CopySlice` wrapping the backward of the
// in-place operation.
// See NOTE [ Autograd View Variables ].
struct TORCH_API CopySlices : public Node {
  CopySlices(
      const Variable& base_var,
      at::TensorGeometry view_,
      std::shared_ptr<Node> fn_);

  variable_list apply(variable_list&& inputs) override;
  void release_variables() override;

  at::TensorGeometry base;
  at::TensorGeometry view;
  std::shared_ptr<Node> fn;
};

}}
