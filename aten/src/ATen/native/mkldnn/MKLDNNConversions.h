#pragma once

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor mkldnn_reorder_conv2d_input(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups);

#endif // AT_MKLDNN_ENABLED()

}}
