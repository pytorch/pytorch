#pragma once

// ${generated_comment}

#include <torch/library.h>
#include <torch/csrc/autograd/variable.h>
#include <c10/core/SymIntArrayRef.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
$ops_headers
#endif

namespace torch::autograd::generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::ScalarType;
using c10::optional;
using c10::fmap;

${view_func_declarations}

} // namespace torch::autograd::generated
