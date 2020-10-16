#pragma once

#include <ATen/native/vulkan/Vulkan.h>
#include <ATen/native/vulkan/VulkanConvolution.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {
namespace vulkan {
namespace detail {

void upsample_nearest2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    int64_t IH,
    int64_t IW,
    int64_t OH,
    int64_t OW,
    int64_t N,
    int64_t C,
    float scaleH,
    float scaleW);

void adaptive_avg_pool2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const int64_t IH,
    const int64_t IW,
    const int64_t OH,
    const int64_t OW,
    const int64_t IN,
    const int64_t IC);

void max_pool2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const int iH,
    const int iW,
    const int oH,
    const int oW,
    const int _n,
    const int _c,
    const int kH,
    const int kW,
    const int dH,
    const int dW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW);

void avg_pool2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const int iH,
    const int iW,
    const int oH,
    const int oW,
    const int _n,
    const int _c,
    const int kH,
    const int kW,
    const int dH,
    const int dW,
    const int padH,
    const int padW);

VulkanTensor transpose(
    const VulkanTensor& input,
    const int64_t dim0,
    const int64_t dim1);

VulkanTensor slice(
    const VulkanTensor& input,
    const int64_t dim,
    const int64_t start,
    const int64_t end,
    const int64_t step);

VulkanTensor reshape_copy(
    const VulkanTensor& input,
    std::vector<int64_t> shape);

VulkanTensor cat(
    VulkanTensor& output,
    ArrayRef<VulkanTensor> inputs,
    int64_t dim);

void add(
    VulkanTensor& output,
    const VulkanTensor& input0,
    const VulkanTensor& input1,
    float alpha);

void mul(VulkanTensor& output, const VulkanTensor& input, const float s);

void add(VulkanTensor& output, const VulkanTensor& input, const float s);

void conv2d_prepack_weights(
    VulkanTensor& output,
    const float* weight,
    int64_t OC,
    int64_t C,
    int64_t KH,
    int64_t KW);

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const float* weight,
    const c10::optional<const float*> bias,
    const Conv2DParams params,
    c10::optional<float> output_min = c10::nullopt,
    c10::optional<float> output_max = c10::nullopt);

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight_prepacked,
    const c10::optional<const float*> bias,
    const Conv2DParams params,
    c10::optional<float> output_min = c10::nullopt,
    c10::optional<float> output_max = c10::nullopt);

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight_prepacked,
    const VulkanTensor& bias,
    const Conv2DParams params,
    c10::optional<float> output_min = c10::nullopt,
    c10::optional<float> output_max = c10::nullopt);

void clamp(
    VulkanTensor& output,
    const VulkanTensor& input,
    float min,
    float max);

void addmm(
    VulkanTensor& output,
    c10::optional<const VulkanTensor> t,
    const VulkanTensor& m1,
    const VulkanTensor& m2,
    float beta,
    float alpha);

void mean(VulkanTensor& output, const VulkanTensor& input);

} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
