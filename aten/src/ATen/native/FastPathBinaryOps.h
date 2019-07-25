#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

bool all_cpu_fast_path_conds_satisfied(const Tensor& t1, const Tensor& t2);
bool all_cpu_fast_path_conds_satisfied(const Tensor& t);

using fast_path_binary_fn_alpha = void(*)(Tensor& result, const Tensor& self,
    const Tensor& other, Scalar alpha);
using fast_path_binary_fn = void(*)(Tensor& result, const Tensor& self,
    const Tensor& other);

DECLARE_DISPATCH(fast_path_binary_fn_alpha, fast_path_add_stub);
DECLARE_DISPATCH(fast_path_binary_fn_alpha, fast_path_sub_stub);
DECLARE_DISPATCH(fast_path_binary_fn, fast_path_mul_stub);
DECLARE_DISPATCH(fast_path_binary_fn, fast_path_div_stub);

}} // namespace at::native
