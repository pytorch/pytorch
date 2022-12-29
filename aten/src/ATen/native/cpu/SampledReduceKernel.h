#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReductionType.h>

namespace at { namespace native {

using sampled_reduce_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, BinaryReductionType);
using sampled_reduce_backward_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, BinaryReductionType);

DECLARE_DISPATCH(sampled_reduce_fn, sampled_reduce_stub);
DECLARE_DISPATCH(sampled_reduce_backward_fn, sampled_reduce_backward_stub);

}} // at::native
