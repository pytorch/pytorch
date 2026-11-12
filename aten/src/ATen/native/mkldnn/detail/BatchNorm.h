#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::batch_normalization_forward_inference::compute (the
// overload taking mean/variance).
void batch_normalization_forward_inference(
    const ITensor& src,
    const ITensor& mean,
    const ITensor& variance,
    const ITensor& scale,
    const ITensor& shift,
    ITensor& dst,
    float epsilon);

// Ported from ideep::batch_normalization_forward_training::compute (the
// overload without running stats).
void batch_normalization_forward_training(
    const ITensor& src,
    const ITensor& scale,
    const ITensor& shift,
    ITensor& dst,
    ITensor& mean,
    ITensor& variance,
    float epsilon);

// Ported from ideep::batch_normalization_backward::compute.
void batch_normalization_backward(
    const ITensor& src,
    const ITensor& mean,
    const ITensor& variance,
    const ITensor& diff_dst,
    const ITensor& scale,
    ITensor& diff_src,
    ITensor& diff_scale,
    ITensor& diff_shift,
    float epsilon);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
