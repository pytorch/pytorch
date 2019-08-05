#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using topk_fn = void(*)(Tensor&, Tensor&, const Tensor&, int64_t, int64_t, bool, bool);

DECLARE_DISPATCH(topk_fn, topk_stub);

}} // at::native
