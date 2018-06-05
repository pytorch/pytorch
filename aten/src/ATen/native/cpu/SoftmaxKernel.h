#pragma once

#include <ATen/ATen.h>
#include "CapabilityDispatch.h"

namespace at {
namespace native {

using forward_fn = void(*)(Tensor &, const Tensor &);
using backward_fn = void(*)(Tensor &, const Tensor &, const Tensor&);

extern DispatchStub<forward_fn> softmax_lastdim_kernel;
extern DispatchStub<forward_fn> log_softmax_lastdim_kernel;
extern DispatchStub<backward_fn> softmax_backward_lastdim_kernel;
extern DispatchStub<backward_fn> log_softmax_backward_lastdim_kernel;

}
}
