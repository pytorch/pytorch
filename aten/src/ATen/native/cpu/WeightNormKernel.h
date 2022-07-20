#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using weight_norm_fn = void(*)(Tensor&, Tensor&, const Tensor&, const Tensor&, int64_t);
using weight_norm_backward_fn = void(*)(
    Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, int64_t);

DECLARE_DISPATCH(weight_norm_fn, weight_norm_stub);
DECLARE_DISPATCH(weight_norm_backward_fn, weight_norm_backward_stub);

}}  // namespace at::native
