#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/THP_export.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using c10::optional;
using c10::fmap;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [](const SavedVariable& x) {
    return static_cast<Tensor>(x.unpack());
  });
}

struct TypeAndSize {
  TypeAndSize() : options(at::TensorOptions()) {}
  /* implicit */
  TypeAndSize(const Tensor & t)
    : sizes(t.sizes().vec())
    , options(t.options()) {}

  Tensor zeros() { return at::zeros(sizes, options); }

private:
  std::vector<int64_t> sizes;
  at::TensorOptions options;
};

${autograd_function_declarations}

}}} // namespace torch::autograd::generated
