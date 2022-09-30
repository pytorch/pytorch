#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/Utils.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace linear {

Tensor linear_eltwise_run(
    const Tensor& input,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    std::string attr,
    std::vector<c10::optional<at::Scalar>> scalars,
    c10::optional<std::string> algorithm);

} // namespace linear
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
