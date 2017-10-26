#pragma once

// ${generated_comment}

#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/utils/tensor_geometry.h"

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntList;
using at::Type;

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

${autograd_function_declarations}

}}} // namespace torch::autograd::generated
