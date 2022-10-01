#pragma once
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <ATen/core/IListRef.h>

namespace at {
namespace native {
// This file contains non-symbolic signatures for ops that we have sym-intified the signature of.
// However, in certain cases (such as static runtime), we call the native versions of the ops directly.
// In those cases, we will duplicate the signature here with non-symbolic ints, and also duplicate the C++ implementation.
TORCH_API at::Tensor reshape(const at::Tensor& self, at::IntArrayRef proposed_shape);
}}
