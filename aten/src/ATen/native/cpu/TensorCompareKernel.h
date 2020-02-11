#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

using reduce_fn =
    void (*)(Tensor&, Tensor&, const Tensor&, c10::optional<int64_t>);

DECLARE_DISPATCH(reduce_fn, max_kernel);
DECLARE_DISPATCH(reduce_fn, min_kernel);

using where_fn = void (*)(TensorIterator &, ScalarType);
DECLARE_DISPATCH(where_fn, where_kernel);

}} // namespace at::native
