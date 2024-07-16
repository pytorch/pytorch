#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {

namespace native {

using fused_adagrad_fn = void (*)(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& state_sum,
    const at::Tensor& state_step,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const float* grad_scale_ptr);

DECLARE_DISPATCH(fused_adagrad_fn, fused_adagrad_stub);

}
}
