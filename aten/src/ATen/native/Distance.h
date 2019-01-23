#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using pdist_forward_fn = void(*)(Tensor&, const Tensor&, const double p);
using pdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p, const Tensor&);

DECLARE_DISPATCH(pdist_forward_fn, pdist_forward_stub);
DECLARE_DISPATCH(pdist_backward_fn, pdist_backward_stub);

}} // namespace at::native
