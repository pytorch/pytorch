#pragma once

#include <ATen/Tensor.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

namespace at::native {
C10_API Tensor mkldnn_linear_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string_view attr,
    c10::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm);

C10_API Tensor mkldnn_linear_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string_view attr);

#if AT_MKL_ENABLED()

C10_API Tensor mkl_linear(
    const Tensor& self,
    const Tensor& mkl_weight_t,
    const Tensor& origin_weight_t,
    const std::optional<Tensor>& bias_opt,
    const int64_t prepack_batch_size);

#endif// AT_MKL_ENABLED

} // namespace at

#endif // AT_MKLDNN_ENABLED()

namespace at::native {
Tensor&
mkldnn_scaled_mm(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out);
} // namespace at::native
