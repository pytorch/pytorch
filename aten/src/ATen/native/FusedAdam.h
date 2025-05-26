#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

enum class ADAM_MODE : uint8_t { ORIGINAL = 0, ADAMW = 1 };

using fused_adam_fn = void (*)(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& max_exp_avg_sq,
    const at::Tensor& state_step,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const float* grad_scale_ptr,
    const ADAM_MODE);

DECLARE_DISPATCH(fused_adam_fn, fused_adam_stub)

} // namespace at::native
