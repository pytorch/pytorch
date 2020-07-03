#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

using _compute_linear_combination_fn = void(*)(
  TensorIterator& iter,
  int32_t in_stride,
  int32_t coeff_stride,
  int32_t num_summations
);

DECLARE_DISPATCH(_compute_linear_combination_fn, _compute_linear_combination_stub);

}} // namespace at::native
