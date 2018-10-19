#pragma once

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

#include <ATen/TensorGeometry.h>
#include "ATen/Type.h"
#include "c10/util/Optional.h"

#include <cstdint>
#include <memory>

namespace torch { namespace autograd {

struct CopyBackwards : public Function {
  variable_list apply(variable_list&& grads) override;

  at::Type *src_type;
  int32_t src_device = -1;
};

// Performs grad[idx] = fn(grad[idx]), but out-of-place. The slicing operation
// grad[idx] is defined by the relative sizes, strides, and offset of base and
// view.
struct CopySlices : public Function {
  CopySlices(
      const Variable& base_var,
      at::TensorGeometry view_,
      std::shared_ptr<Function> fn_);

  variable_list apply(variable_list&& inputs) override;
  void release_variables() override;

  at::TensorGeometry base;
  at::TensorGeometry view;
  std::shared_ptr<Function> fn;
};

}}
