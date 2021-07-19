#pragma once

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <ATen/ATen.h>
#include <vector>
#include <cpuinfo.h>


namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace = false);

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode);
};

inline bool mkldnn_bf16_device_check() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bw()
      && cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512dq();
}

inline bool use_mkldnn_bf16_gemm(
    const Tensor& mat1,
    const Tensor& mat2,
    const c10::optional<Tensor>& result_opt) {
  c10::MaybeOwned<Tensor> result_maybe_owned = at::borrow_from_optional_tensor(result_opt);
  const Tensor& result = *result_maybe_owned;

  const int64_t mkldnn_gemm_min_size = 16 * 16 * 16;
  // if dim = 2, mat1's size = (m * n), mat2's size = (n * k)
  // else dim = 3, mat1's size = (b * m * n), mat2's size = (b * n * k)
  // only m * n * k are large enough we can get benefit from mkldnn optimized gemm kernel
  // if some cases pytorch dose not have default impl for bf16 (such as "dot"), will use mkldnn impl anyway
  int64_t m = mat1.dim() == 2? mat1.size(0) : mat1.size(1);
  int64_t n = mat1.dim() == 2? mat1.size(1) : mat1.size(2);
  int64_t k = mat2.dim() == 2? mat2.size(1) : mat2.size(2);
  return (
    m * n * k >= mkldnn_gemm_min_size &&
    mkldnn_bf16_device_check() &&
    mat1.scalar_type() == kBFloat16 &&
    mat2.scalar_type() == kBFloat16 &&
    (!result.defined() || result.scalar_type() == kBFloat16) &&
    mat1.numel() != 0 &&
    mat2.numel() != 0);
}

}
