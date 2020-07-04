#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

using reduce_minmax_fn =
    void (*)(Tensor&, Tensor&, const Tensor&, int64_t, bool);

DECLARE_DISPATCH(reduce_minmax_fn, max_stub);
DECLARE_DISPATCH(reduce_minmax_fn, min_stub);

using where_fn = void (*)(TensorIterator &, ScalarType);
DECLARE_DISPATCH(where_fn, where_kernel);

}} // namespace at::native
