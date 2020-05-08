#if !defined(USE_VULKAN) && !defined(USE_GLES)
#include <ATen/ATen.h>

namespace at {
namespace native {

bool _vulkan_available() {
  return false;
}

at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src) {
  AT_ERROR("vulkan_copy_: ATen not compiled with Vulkan or GLES support");
}

Tensor vulkan_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  AT_ERROR("vulkan_add: ATen not compiled with Vulkan or GLES support");
}

at::Tensor empty_vulkan(
    IntArrayRef sizes,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ERROR("empty_vulkan: ATen not compiled with Vulkan or GLES support");
}

at::Tensor vulkan_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  AT_ERROR("vulkan_convolution: ATen not compiled with Vulkan or GLES support");
}

at::Tensor upsample_nearest2d_vulkan(
    const at::Tensor& input,
    IntArrayRef outputSizes,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_ERROR(
      "upsample_nearest2d_vulkan: ATen not compiled with Vulkan or GLES support");
}

Tensor vulkan_addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  AT_ERROR("vulkan_addmm: ATen not compiled with Vulkan or GLES support");
}

Tensor vulkan_clamp(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  AT_ERROR("vulkan_clamp: ATen not compiled with Vulkan or GLES support");
}

Tensor& _clamp__vulkan(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  AT_ERROR("_clamp_vulkan: ATen not compiled with Vulkan or GLES support");
}

Tensor vulkan_hardtanh(const Tensor& self, Scalar min, Scalar max) {
  AT_ERROR("vulkan_hardtanh: ATen not compiled with Vulkan or GLES support");
}

Tensor& vulkan_hardtanh_(Tensor& self, Scalar min, Scalar max) {
  AT_ERROR("vulkan_hardtanh_: ATen not compiled with Vulkan or GLES support");
}

Tensor mean_vulkan(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  AT_ERROR("mean_vulkan: ATen not compiled with Vulkan or GLES support");
}
} // namespace native
} // namespace at
#endif
