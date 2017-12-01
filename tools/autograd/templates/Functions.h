#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntList;
using at::Type;
using at::TensorGeometry;

struct TypeAndSize {
  TypeAndSize() : type(nullptr) {}
  /* implicit */
  TypeAndSize(const Tensor & t)
    : sizes(t.sizes())
    , type(&t.type()) {}

  Tensor zeros() { return type->zeros(sizes); }

private:
  std::vector<int64_t> sizes;
  Type* type;
};

std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    const Tensor & input,
    const Tensor & gamma,
    const Tensor & ggI,
    const Tensor & ggG,
    const Tensor & ggB,
    const Tensor & gO,
    double eps,
    const Tensor & save_mean, // not Variable
    const Tensor & save_std, // not Variable
    const Tensor & running_mean, // not Variable
    const Tensor & running_var, // not Variable
    bool training);

${autograd_function_declarations}

}}} // namespace torch::autograd::generated
