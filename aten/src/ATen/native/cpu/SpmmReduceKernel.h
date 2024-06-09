#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReductionType.h>

namespace at::native {

using spmm_reduce_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_arg_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_backward_input_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_backward_input_arg_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_backward_other_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);

DECLARE_DISPATCH(spmm_reduce_fn, spmm_reduce_stub);
DECLARE_DISPATCH(spmm_reduce_arg_fn, spmm_reduce_arg_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_fn, spmm_reduce_backward_input_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_arg_fn, spmm_reduce_backward_input_arg_stub);
DECLARE_DISPATCH(spmm_reduce_backward_other_fn, spmm_reduce_backward_other_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_arg_fn, spmm_reduce_backward_other_arg_stub);

} // at::native
