#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include "c10/util/Optional.h"

namespace at { namespace native {

using reduce_fn = void (*)(Tensor&, const Tensor&, c10::optional<int64_t>);

DECLARE_DISPATCH(reduce_fn, sum_kernel);
DECLARE_DISPATCH(reduce_fn, prod_kernel);

using reduce_norm_fn =
    void (*)(Tensor&, const Tensor&, Scalar, c10::optional<int64_t>);
DECLARE_DISPATCH(reduce_norm_fn, norm_kernel);

}} // namespace at::native
