#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native::mkldnn {

// Ported from ideep::batch_normalization_forward_inference::compute (the
// overload taking mean/variance).
void batch_normalization_forward_inference(
    const MkldnnMemory& src,
    const MkldnnMemory& mean,
    const MkldnnMemory& variance,
    const MkldnnMemory& scale,
    const MkldnnMemory& shift,
    MkldnnMemory& dst,
    float epsilon);

// Ported from ideep::batch_normalization_forward_training::compute (the
// overload without running stats).
void batch_normalization_forward_training(
    const MkldnnMemory& src,
    const MkldnnMemory& scale,
    const MkldnnMemory& shift,
    MkldnnMemory& dst,
    MkldnnMemory& mean,
    MkldnnMemory& variance,
    float epsilon);

// Ported from ideep::batch_normalization_backward::compute.
void batch_normalization_backward(
    const MkldnnMemory& src,
    const MkldnnMemory& mean,
    const MkldnnMemory& variance,
    const MkldnnMemory& diff_dst,
    const MkldnnMemory& scale,
    MkldnnMemory& diff_src,
    MkldnnMemory& diff_scale,
    MkldnnMemory& diff_shift,
    float epsilon);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
