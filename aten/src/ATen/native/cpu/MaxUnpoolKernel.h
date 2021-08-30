#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/DispatchStub.h>

#pragma once

namespace at { namespace native {

using max_unpooling_fn = void(*)(Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(max_unpooling_fn, max_unpool2d_kernel);
DECLARE_DISPATCH(max_unpooling_fn, max_unpool2d_backward_kernel);
DECLARE_DISPATCH(max_unpooling_fn, max_unpool3d_kernel);
DECLARE_DISPATCH(max_unpooling_fn, max_unpool3d_backward_kernel);

}} // at::native
