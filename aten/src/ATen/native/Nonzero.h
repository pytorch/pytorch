#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using nonzero_fn = void(*)(Tensor&, const Tensor&);

DECLARE_DISPATCH(nonzero_fn, nonzero_stub);

}} // namespace at::native
