#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/DispatchStub.h>

#pragma once

namespace at { namespace native {

using pixel_shuffle_fn = void(*)(Tensor&, const Tensor&, int64_t);
DECLARE_DISPATCH(pixel_shuffle_fn, pixel_shuffle_kernel);
DECLARE_DISPATCH(pixel_shuffle_fn, pixel_unshuffle_kernel);

}} // at::native
