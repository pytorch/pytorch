#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/AmpKernels.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale_native.h>
#include <ATen/ops/_amp_update_scale.h>
#include <ATen/ops/_amp_update_scale_native.h>
#endif

namespace at::native {

void _amp_foreach_non_finite_check_and_unscale_cpu_(
    TensorList scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
    _amp_foreach_non_finite_check_and_unscale_cpu_stub(
        found_inf.device().type(), scaled_grads, found_inf, inv_scale);
}

at::Tensor& _amp_update_scale_cpu_ (
    at::Tensor& current_scale,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
    return _amp_update_scale_cpu_stub(
        growth_tracker.device().type(), current_scale, growth_tracker,
        found_inf, growth_factor, backoff_factor, growth_interval);
}

DEFINE_DISPATCH(_amp_foreach_non_finite_check_and_unscale_cpu_stub);
DEFINE_DISPATCH(_amp_update_scale_cpu_stub);

} // namespace at::native
