#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using cross_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const int64_t d);

DECLARE_DISPATCH(cross_fn, cross_stub);

}} // namespace at::native
