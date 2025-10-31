#pragma once

#include <ATen/Config.h>
#include <ATen/core/Tensor.h>

#if AT_ZENDNN_ENABLED()
namespace at::native {

TORCH_API void zendnn_baddbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    float beta,
    float alpha);

} // namespace at::native

#endif // AT_ZENDNN_ENABLED()
