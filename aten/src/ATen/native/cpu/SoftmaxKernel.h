#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using forward_fn = void (*)(const Tensor&, const Tensor&);
using backward_fn = void(*)(const Tensor &, const Tensor &, const Tensor&);

DECLARE_DISPATCH(forward_fn, softmax_lastdim_kernel);
DECLARE_DISPATCH(forward_fn, log_softmax_lastdim_kernel);
DECLARE_DISPATCH(backward_fn, softmax_backward_lastdim_kernel);
DECLARE_DISPATCH(backward_fn, log_softmax_backward_lastdim_kernel);

using forward_fn_with_dim = void(*)(const Tensor &, const Tensor &, const int64_t);
DECLARE_DISPATCH(forward_fn_with_dim, softmax_kernel);

}
}
