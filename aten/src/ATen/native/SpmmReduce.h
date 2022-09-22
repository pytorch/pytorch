#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using spmm_sum_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const c10::optional<Tensor>&, const Tensor&);
DECLARE_DISPATCH(spmm_sum_fn, spmm_sum_stub);

}} // at::native
