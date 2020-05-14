#include <ATen/native/vulkan/Vulkan.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {
namespace vulkan {
namespace details {
namespace vulkan {

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

void add(
    VulkanTensor& output,
    const VulkanTensor& input0,
    const VulkanTensor& input1,
    float alpha);

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
    int64_t KH,
    int64_t KW,
    const c10::optional<float*> bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G);

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight_prepacked,
    int64_t KH,
    int64_t KW,
    const c10::optional<float*> bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G);

void conv2d(
    VulkanTensor& output,
    const VulkanTensor& input,
    const VulkanTensor& weight_prepacked,
    int64_t KH,
    int64_t KW,
    const VulkanTensor& bias,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G);

void clamp(
    VulkanTensor& output,
    const VulkanTensor& input,
    float min,
    float max);

void addmm(
    VulkanTensor& output,
    const VulkanTensor& t,
    const VulkanTensor& m1,
    const VulkanTensor& m2,
    float beta,
    float alpha);

void mean(VulkanTensor& output, const VulkanTensor& input);

} // namespace vulkan
} // namespace details
} // namespace vulkan
} // namespace native
} // namespace at
