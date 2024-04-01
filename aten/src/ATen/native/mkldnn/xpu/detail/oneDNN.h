#pragma once

#include <ATen/ATen.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>


namespace at::native::onednn{

TORCH_API sycl::event matmul(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& b_raw,
    bool m2_trans,
    Attr attr,
    const std::vector<sycl::event>& deps = {});

} // namespace at::native::onednn
