#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::batch_normalization_forward_inference::compute (the
// overload taking mean/variance).
void batch_normalization_forward_inference(
    const ideep::tensor& src,
    const ideep::tensor& mean,
    const ideep::tensor& variance,
    const ideep::tensor& scale,
    const ideep::tensor& shift,
    ideep::tensor& dst,
    float epsilon);

// Ported from ideep::batch_normalization_forward_training::compute (the
// overload without running stats).
void batch_normalization_forward_training(
    const ideep::tensor& src,
    const ideep::tensor& scale,
    const ideep::tensor& shift,
    ideep::tensor& dst,
    ideep::tensor& mean,
    ideep::tensor& variance,
    float epsilon);

// Ported from ideep::batch_normalization_backward::compute.
void batch_normalization_backward(
    const ideep::tensor& src,
    const ideep::tensor& mean,
    const ideep::tensor& variance,
    const ideep::tensor& diff_dst,
    const ideep::tensor& scale,
    ideep::tensor& diff_src,
    ideep::tensor& diff_scale,
    ideep::tensor& diff_shift,
    float epsilon);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
