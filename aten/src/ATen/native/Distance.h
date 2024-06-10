#pragma once

#include <ATen/native/DispatchStub.h>

namespace at {
class Tensor;

namespace native {

using pdist_forward_fn = void(*)(Tensor&, const Tensor&, const double p);
using pdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p, const Tensor&);
using cdist_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p);
using cdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, const double p, const Tensor&);

DECLARE_DISPATCH(pdist_forward_fn, pdist_forward_stub);
DECLARE_DISPATCH(pdist_backward_fn, pdist_backward_stub);
DECLARE_DISPATCH(cdist_fn, cdist_stub);
DECLARE_DISPATCH(cdist_backward_fn, cdist_backward_stub);

}} // namespace at::native
