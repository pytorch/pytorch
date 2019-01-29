#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using pdist_forward_fn = void(*)(Tensor&, const Tensor&, const double p);
using pdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p, const Tensor&);
using cdist_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p);

DECLARE_DISPATCH(pdist_forward_fn, pdist_forward_stub);
DECLARE_DISPATCH(pdist_backward_fn, pdist_backward_stub);
DECLARE_DISPATCH(cdist_fn, cdist_stub);

}} // namespace at::native
