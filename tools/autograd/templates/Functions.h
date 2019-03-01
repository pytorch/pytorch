#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>
#include <c10/core/TensorOptions.h>

#include "torch/csrc/THP_export.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"

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
  TypeAndSize() : backend(at::Backend::Undefined), dtype(caffe2::TypeMeta()) {}
  /* implicit */
  TypeAndSize(const Tensor & t)
    : sizes(t.sizes().vec())
    , backend(t.type().backend())
    , dtype(t.dtype()) {}

  Tensor zeros() { return at::zeros(sizes, c10::TensorOptions(backend).dtype(dtype)); }

private:
  std::vector<int64_t> sizes;
  at::Backend backend;
  caffe2::TypeMeta dtype;
};

${autograd_function_declarations}

}}} // namespace torch::autograd::generated
