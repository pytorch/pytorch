#pragma once

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <vector>
#include <cpuinfo.h>

namespace at { namespace native {

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

inline bool mkldnn_bf16_gemm_usable_check(const Tensor& mat1, const Tensor& mat2, const Tensor& result) {
  return (
    mkldnn_bf16_device_check() &&
    mat1.scalar_type() == kBFloat16 &&
    mat2.scalar_type() == kBFloat16 &&
    result.scalar_type() == kBFloat16 &&
    mat1.numel() != 0 &&
    mat2.numel() != 0);
}

}
