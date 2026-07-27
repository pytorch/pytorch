#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using fused_sgd_fn = void (*)(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& momentum_buffer,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr);

DECLARE_DISPATCH(fused_sgd_fn, fused_sgd_stub)

} // namespace at::native
