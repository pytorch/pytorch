#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/core/ATen_fwd.h>

namespace at {
class Tensor;

namespace native {

using _amp_foreach_non_finite_check_and_unscale_cpu__fn = void (*)(
    TensorList,
    Tensor&,
    const Tensor&);

using _amp_update_scale_cpu__fn = Tensor& (*)(
    Tensor&,
    Tensor&,
    const Tensor&,
    double,
    double,
    int64_t);

DECLARE_DISPATCH(_amp_foreach_non_finite_check_and_unscale_cpu__fn, _amp_foreach_non_finite_check_and_unscale_cpu_stub)
DECLARE_DISPATCH(_amp_update_scale_cpu__fn, _amp_update_scale_cpu_stub)

} // namespace native
} // namespace at
