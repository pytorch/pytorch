#pragma once

#include <Python.h>
#include <memory>
#include "ATen/Type.h"

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

#include <ATen/TensorGeometry.h>

namespace torch { namespace autograd {

struct CopyBackwards : public Function {
  virtual variable_list apply(const variable_list& inputs) override;

  at::Type *src_type;
  int64_t src_device;
};

// Performs grad[idx] = fn(grad[idx]), but out-of-place. The slicing operation
// grad[idx] is defined by the relative sizes, strides, and offset of base and
// view.
struct CopySlices : public Function {
  CopySlices(const Variable& base, at::TensorGeometry view, std::shared_ptr<Function> fn);

  virtual variable_list apply(const variable_list& grads) override;
  virtual void releaseVariables() override;

  at::TensorGeometry base;
  at::TensorGeometry view;
  std::shared_ptr<Function> fn;
};

}}
