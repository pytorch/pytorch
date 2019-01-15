#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/Optional.h>

namespace at {
  struct TensorIterator;
}

namespace at { namespace native {

using reduce_fn = void(*)(TensorIterator &);

DECLARE_DISPATCH(reduce_fn, sum_stub);
DECLARE_DISPATCH(reduce_fn, prod_stub);
DECLARE_DISPATCH(reduce_fn, mean_stub);

using reduce_std_var_function =
  void (*)(TensorIterator&, bool unbiased, bool take_sqrt);
DECLARE_DISPATCH(reduce_std_var_function, std_var_stub);

using reduce_norm_fn =
    void (*)(Tensor&, const Tensor&, Scalar, c10::optional<int64_t>);
DECLARE_DISPATCH(reduce_norm_fn, norm_kernel);

}} // namespace at::native
