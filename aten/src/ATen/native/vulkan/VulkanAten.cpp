#include <ATen/native/vulkan/VulkanAten.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/Vulkan.h>
#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>
#include <ATen/native/vulkan/VulkanOps.h>

namespace at {
namespace native {

bool is_vulkan_available() {
  return at::native::vulkan::detail::is_available();
}

using vulkan::detail::VulkanTensor;
using VulkanTensorImpl = VulkanOpaqueTensorImpl<VulkanTensor>;

at::Tensor new_with_vtensor_vulkan(
    VulkanTensor&& vt,
    const TensorOptions& options) {
  auto sizes = vt.sizes();
  auto strides = vt.strides();
  return detail::make_tensor<VulkanTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      options.dtype(),
      at::Device(at::kVulkan),
      std::move(vt),
      std::vector<int64_t>(sizes.begin(), sizes.end()),
      std::vector<int64_t>(strides.begin(), strides.end()));
}

VulkanTensor& vtensor_from_vulkan(const at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      tensor.is_vulkan(), "vtensor_from_vulkan expects Vulkan tensor input");
  VulkanTensorImpl* impl =
      static_cast<VulkanTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle();
}

at::Tensor empty_vulkan(
    IntArrayRef sizes,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !options.has_memory_format(),
      "'memory_format' argument is incompatible with Vulkan tensor");
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "'memory_format' argument is incompatible with Vulkan tensor");

  VulkanTensor vt{sizes.vec()};
  return new_with_vtensor_vulkan(std::move(vt), options);
}

at::Tensor empty_strided_vulkan(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  return empty_vulkan(size, options, c10::nullopt);
}

at::Tensor& copy_from_vulkan_(at::Tensor& self, const at::Tensor& src) {
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::Vulkan,
      "copy_from_vulkan input tensor's device is not Vulkan");
  TORCH_INTERNAL_ASSERT(
      self.device().type() == DeviceType::CPU,
      "copy_from_vulkan is implemented only for CPU device output");
  TORCH_INTERNAL_ASSERT(
      self.layout() == Layout::Strided,
      "copy_from_vulkan is implemented only for Strided layout output");
  TORCH_INTERNAL_ASSERT(
      self.scalar_type() == ScalarType::Float,
      "copy_from_vulkan is implemented only for float dtype output, got:",
      self.scalar_type());
  TORCH_INTERNAL_ASSERT(
      self.is_contiguous(),
      "copy_from_vulkan is implemented only for contiguous output tensor");

  VulkanTensor& vtensor = vtensor_from_vulkan(src);
  vtensor.copy_data_to_host(self.data_ptr<float>());
  return self;
}

at::Tensor& copy_to_vulkan_(at::Tensor& self, const at::Tensor& src) {
  TORCH_INTERNAL_ASSERT(
      self.device().type() == DeviceType::Vulkan,
      "copy_to_vulkan output tensor's device is not Vulkan");
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::CPU,
      "copy_to_vulkan is implemented only for CPU device input");
  TORCH_INTERNAL_ASSERT(
      src.layout() == Layout::Strided,
      "copy_to_vulkan is implemented only for Strided layout input");
  TORCH_INTERNAL_ASSERT(
      src.scalar_type() == ScalarType::Float,
      "copy_to_vulkan is implemented only for float dtype");

  auto cpu_tensor_contiguous = src.contiguous();
  VulkanTensor& vtensor = vtensor_from_vulkan(self);
  vtensor.set_data_from_host(cpu_tensor_contiguous.data_ptr<float>());
  return self;
}

at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src) {
  if (src.device().type() == at::kVulkan && self.device().type() == at::kCPU) {
    return copy_from_vulkan_(self, src);
  }
  if (src.device().type() == at::kCPU && self.device().type() == at::kVulkan) {
    return copy_to_vulkan_(self, src);
  }
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::Vulkan,
      "vulkan_copy_ is implemented only for CPU,Strided,float->Vulkan; Vulkan->CPU,Strided,float");
  return self;
}

at::Tensor upsample_nearest2d_vulkan(
    const at::Tensor& input,
    IntArrayRef outputSizes,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  VulkanTensor& x = vtensor_from_vulkan(input);
  auto inputSizes = input.sizes();
  auto in = inputSizes[0];
  auto ic = inputSizes[1];
  auto ih = inputSizes[2];
  auto iw = inputSizes[3];

  auto oh = outputSizes[0];
  auto ow = outputSizes[1];
  const float height_scale = compute_scales_value<float>(scales_h, ih, oh);
  const float width_scale = compute_scales_value<float>(scales_w, iw, ow);
  Tensor output = empty_vulkan({in, ic, oh, ow}, input.options(), {});
  VulkanTensor& y = vtensor_from_vulkan(output);
  y.allocate_storage();
  vulkan::detail::upsample_nearest2d(
      y, x, ih, iw, oh, ow, in, ic, height_scale, width_scale);
  return output;
}

Tensor vulkan_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  VulkanTensor& x = vtensor_from_vulkan(self);
  VulkanTensor& y = vtensor_from_vulkan(other);
  float a = alpha.to<float>();

  VulkanTensor output = VulkanTensor{self.sizes().vec()};
  output.allocate_storage();
  vulkan::detail::add(output, x, y, a);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

at::Tensor vulkan_convolution(
    const at::Tensor& input, // Vulkan
    const at::Tensor& weight, // CPU
    const at::Tensor& bias, // CPU
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  vulkan::Conv2DParams params{
      input.sizes(), weight.sizes(), padding, stride, dilation, groups};
  TORCH_INTERNAL_ASSERT(
      input.dim() == 4, "vulkan_convolution: Expected 4-dimensional input");
  TORCH_INTERNAL_ASSERT(
      weight.dim() == 4, "vulkan_convolution: Expected 4-dimensional weight");
  TORCH_INTERNAL_ASSERT(
      groups == 1 || groups == params.C,
      "vulkan_convolution: only nogroup or depthwise convolutions supported");

  const VulkanTensor& vinput = vtensor_from_vulkan(input);
  VulkanTensor voutput = VulkanTensor{params.output_sizes()};
  voutput.allocate_storage();

  vulkan::detail::conv2d(
      voutput,
      vinput,
      weight.data_ptr<float>(),
      bias.defined() ? c10::make_optional<const float*>(bias.data_ptr<float>())
                     : c10::nullopt,
      params);
  return new_with_vtensor_vulkan(std::move(voutput), input.options());
}

at::Tensor vulkan_convolution_prepack_weights(const at::Tensor& weight) {
  auto wsizes = weight.sizes();
  TORCH_INTERNAL_ASSERT(
      wsizes.size() == 4,
      "vulkan_convolution_prepack_weights: Expected 4-dimensional weight");

  const int64_t OC = wsizes[0];
  const int64_t C = wsizes[1];
  const int64_t KH = wsizes[2];
  const int64_t KW = wsizes[3];
  VulkanTensor voutput =
      VulkanTensor{{UP_DIV(OC, 4), UP_DIV(C, 4), KH * KW, 16}};
  voutput.allocate_storage();

  vulkan::detail::conv2d_prepack_weights(
      voutput, weight.data_ptr<float>(), OC, C, KH, KW);
  return new_with_vtensor_vulkan(
      std::move(voutput), at::device(at::kVulkan).dtype(at::kFloat));
}

at::Tensor vulkan_convolution_prepacked(
    const at::Tensor& input, // Vulkan
    IntArrayRef weightSizes,
    const at::Tensor& weight_prepacked_vulkan, // Vulkan
    const c10::optional<at::Tensor>& bias, // Vulkan|CPU
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    const float output_min,
    const float output_max) {
  TORCH_INTERNAL_ASSERT(
      input.dim() == 4, "vulkan_convolution: Expected 4-dimensional input");
  TORCH_INTERNAL_ASSERT(
      weight_prepacked_vulkan.dim() == 4,
      "vulkan_convolution: Expected 4-dimensional weight");
  vulkan::Conv2DParams params{
      input.sizes(), weightSizes, padding, stride, dilation, groups};
  TORCH_INTERNAL_ASSERT(
      groups == 1 || groups == params.C,
      "vulkan_convolution: only nogroup or depthwise convolutions supported");
  const VulkanTensor& vinput = vtensor_from_vulkan(input);
  const VulkanTensor& vweight = vtensor_from_vulkan(weight_prepacked_vulkan);
  VulkanTensor voutput =
      VulkanTensor{{params.N, params.OC, params.OH, params.OW}};
  voutput.allocate_storage();
  const bool hasBias = bias.has_value() && bias->defined();
  const bool vulkanBias = (*bias).is_vulkan();
  if (hasBias && vulkanBias) {
    const VulkanTensor& vbias = vtensor_from_vulkan(*bias);
    vulkan::detail::conv2d(
        voutput, vinput, vweight, vbias, params, output_min, output_max);
  } else {
    vulkan::detail::conv2d(
        voutput,
        vinput,
        vweight,
        hasBias ? c10::make_optional<const float*>((*bias).data_ptr<float>())
                : c10::nullopt,
        params,
        output_min,
        output_max);
  }
  return new_with_vtensor_vulkan(std::move(voutput), input.options());
}

Tensor vulkan_addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  const VulkanTensor t =
      vtensor_from_vulkan(self.is_vulkan() ? self : self.vulkan());
  const VulkanTensor m1 =
      vtensor_from_vulkan(mat1.is_vulkan() ? mat1 : mat1.vulkan());
  const VulkanTensor m2 =
      vtensor_from_vulkan(mat2.is_vulkan() ? mat2 : mat2.vulkan());
  float b = beta.to<float>();
  float a = alpha.to<float>();

  VulkanTensor output = VulkanTensor{self.sizes().vec()};
  output.allocate_storage();
  vulkan::detail::addmm(output, t, m1, m2, b, a);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

Tensor vulkan_clamp(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  VulkanTensor& x = vtensor_from_vulkan(self);
  VulkanTensor output = VulkanTensor{self.sizes().vec()};
  output.allocate_storage();
  float minValue = min.has_value() ? min.value().to<float>()
                                   : std::numeric_limits<float>::min();
  float maxValue = max.has_value() ? max.value().to<float>()
                                   : std::numeric_limits<float>::max();
  vulkan::detail::clamp(output, x, minValue, maxValue);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

Tensor& _clamp__vulkan(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  auto y = vulkan_clamp(self, min, max);
  self.copy_(y);
  return self;
}

Tensor vulkan_hardtanh(const Tensor& self, Scalar min, Scalar max) {
  return vulkan_clamp(self, min, max);
}

Tensor& vulkan_hardtanh_(Tensor& self, Scalar min, Scalar max) {
  return _clamp__vulkan(self, min, max);
}

Tensor mean_vulkan(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  TORCH_INTERNAL_ASSERT(
      self.is_vulkan(), "mean_vulkan expects Vulkan tensor input");
  TORCH_INTERNAL_ASSERT(
      self.dim() == 4 && dim.size() == 2 && dim[0] == 2 && dim[1] == 3);
  VulkanTensor& x = vtensor_from_vulkan(self);
  auto sizes = self.sizes();
  std::vector<int64_t> outputSizes{sizes[0], sizes[1]};
  VulkanTensor output = VulkanTensor{outputSizes};
  output.allocate_storage();
  vulkan::detail::mean(output, x);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

} // namespace native
} // namespace at
