#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using sort_fn = void(*)(Tensor& values, Tensor& indices, int64_t dim, bool descending);
using topk_fn = void(*)(Tensor&, Tensor&, const Tensor&, int64_t, int64_t, bool, bool);

DECLARE_DISPATCH(sort_fn, sort_stub);
DECLARE_DISPATCH(topk_fn, topk_stub);

}} // at::native
