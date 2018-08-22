#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using pdist_fn = void(*)(Tensor &, const Tensor &, const double);
DECLARE_DISPATCH(pdist_fn, pdist_kernel);

using pdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double, const Tensor&);
DECLARE_DISPATCH(pdist_backward_fn, pdist_backward_kernel);

}} // namespace at::native
