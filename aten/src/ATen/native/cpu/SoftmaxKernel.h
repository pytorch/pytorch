#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using forward_fn = void(*)(Tensor &, const Tensor &);
using backward_fn = void(*)(Tensor &, const Tensor &, const Tensor&);

DECLARE_DISPATCH(forward_fn, softmax_lastdim_kernel);
DECLARE_DISPATCH(forward_fn, log_softmax_lastdim_kernel);
DECLARE_DISPATCH(backward_fn, softmax_backward_lastdim_kernel);
DECLARE_DISPATCH(backward_fn, log_softmax_backward_lastdim_kernel);

}
}
