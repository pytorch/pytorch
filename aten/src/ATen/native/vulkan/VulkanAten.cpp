#include <ATen/native/vulkan/VulkanAten.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/OpaqueTensorImpl.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/utils/ParamUtils.h>

#ifdef USE_VULKAN
#include <ATen/native/vulkan/Vulkan.h>
#include <ATen/native/vulkan/VulkanCommon.h>
#include <ATen/native/vulkan/VulkanOps.h>
#define VULKAN_GL vulkan
#else

#ifdef USE_GLES
#include <ATen/native/vulkan/gl/GLES.h>
#define VULKAN_GL gl
#endif

#endif

namespace at {
namespace native {

bool is_vulkan_available() {
  return at::native::vulkan::details::VULKAN_GL::is_available();
}

#ifdef USE_VULKAN
using VTensor = at::native::vulkan::details::vulkan::VulkanTensor;
#else

#ifdef USE_GLES
using VTensor = at::native::vulkan::details::gl::GLTensor;
#endif

#endif

using VulkanTensorImpl = OpaqueTensorImpl<VTensor>;

at::Tensor new_with_vtensor_vulkan(VTensor&& vt, const TensorOptions& options) {
  auto dims = vt.sizes();
  return detail::make_tensor<VulkanTensorImpl>(
      DispatchKeySet(DispatchKey::VulkanTensorId),
      options.dtype(),
      at::Device(at::kVulkan),
      std::move(vt),
      std::vector<int64_t>(dims.begin(), dims.end()));
}

VTensor& vtensor_from_vulkan(const at::Tensor& tensor) {
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

  VTensor vt{sizes.vec()};
  return new_with_vtensor_vulkan(std::move(vt), options);
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

  VTensor& vtensor = vtensor_from_vulkan(src);
  vtensor.copy_data_to_host(self.template data_ptr<float>());
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

  auto cpu_tensor_cont = src.contiguous();
  VTensor& vtensor = vtensor_from_vulkan(self);
  vtensor.set_data_from_host(cpu_tensor_cont.template data_ptr<float>());
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
  VTensor& x = vtensor_from_vulkan(input);
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
  VTensor& y = vtensor_from_vulkan(output);
  y.allocate_storage();
  at::native::vulkan::details::VULKAN_GL::upsample_nearest2d(
      y, x, ih, iw, oh, ow, in, ic, height_scale, width_scale);
  return output;
}

Tensor vulkan_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  VTensor& x = vtensor_from_vulkan(self);
  VTensor& y = vtensor_from_vulkan(other);
  float a = alpha.to<float>();

  VTensor output = VTensor{self.sizes().vec()};
  output.allocate_storage();
  at::native::vulkan::details::VULKAN_GL::add(output, x, y, a);
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
  auto isizes = input.sizes();
  TORCH_INTERNAL_ASSERT(
      isizes.size() == 4, "vulkan_convolution: Expected 4-dimensional input");
  auto wsizes = weight.sizes();
  TORCH_INTERNAL_ASSERT(
      wsizes.size() == 4, "vulkan_convolution: Expected 4-dimensional weight");
  int64_t C = isizes[1];
  TORCH_INTERNAL_ASSERT(
      groups == 1 || groups == C,
      "vulkan_convolution: only nogroup or depthwise convolutions supported");

  int64_t N = isizes[0];
  int64_t H = isizes[2];
  int64_t W = isizes[3];

  int64_t OC = wsizes[0];
  int64_t KH = wsizes[2];
  int64_t KW = wsizes[3];

  int64_t PY = padding[0];
  int64_t PX = padding[1];

  int64_t SY = stride[0];
  int64_t SX = stride[1];

  int64_t DY = dilation[0];
  int64_t DX = dilation[1];

  const int KWE = (KW - 1) * DX + 1;
  const int KHE = (KH - 1) * DY + 1;
  const int64_t OW = ((W - KWE + 2 * PX) / SX) + 1;
  const int64_t OH = ((H - KHE + 2 * PY) / SY) + 1;

  auto osizes = std::vector<int64_t>{N, OC, OH, OW};

  const VTensor& vinput = vtensor_from_vulkan(input);
  VTensor voutput = VTensor{osizes};
  voutput.allocate_storage();

  float* biasData{};
  if (bias.defined()) {
    biasData = bias.template data_ptr<float>();
  } else {
    biasData = (float*)std::malloc(sizeof(float) * OC);
    std::memset(biasData, 0, sizeof(float) * OC);
  }
  float* weightData = weight.template data_ptr<float>();
  at::native::vulkan::details::VULKAN_GL::conv2d(
      voutput,
      vinput,
      weightData,
      KH,
      KW,
      biasData,
      SY,
      SX,
      PY,
      PX,
      DY,
      DX,
      groups);
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
  VTensor voutput = VTensor{{OC, C, KH, KW}};
  voutput.allocate_storage();

  at::native::vulkan::details::VULKAN_GL::conv2d_prepack_weights(
      voutput, weight.template data_ptr<float>(), OC, C, KH, KW);
  return new_with_vtensor_vulkan(
      std::move(voutput), at::device(at::kCPU).dtype(at::kFloat));
}

at::Tensor vulkan_convolution_prepacked(
    const at::Tensor& input, // Vulkan
    const at::Tensor& weight_prepacked_vulkan, // Vulkan
    const c10::optional<at::Tensor>& bias, // Vulkan|CPU
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  auto isizes = input.sizes();
  TORCH_INTERNAL_ASSERT(
      isizes.size() == 4, "vulkan_convolution: Expected 4-dimensional input");
  auto wsizes = weight_prepacked_vulkan.sizes();
  TORCH_INTERNAL_ASSERT(
      wsizes.size() == 4, "vulkan_convolution: Expected 4-dimensional weight");
  vulkan::Conv2DParams c2ds{input.sizes(),
                            weight_prepacked_vulkan.sizes(),
                            padding,
                            stride,
                            dilation,
                            groups};
  TORCH_INTERNAL_ASSERT(
      groups == 1 || groups == c2ds.C,
      "vulkan_convolution: only nogroup or depthwise convolutions supported");
  const VTensor& vinput = vtensor_from_vulkan(input);
  const VTensor& vweight = vtensor_from_vulkan(weight_prepacked_vulkan);
  VTensor voutput = VTensor{{c2ds.N, c2ds.OC, c2ds.OH, c2ds.OW}};
  voutput.allocate_storage();
  const bool hasBias = bias.has_value() && (*bias).defined();
  const bool vulkanBias = (*bias).is_vulkan();
  if (hasBias && vulkanBias) {
    const VTensor& vbias = vtensor_from_vulkan(*bias);
    at::native::vulkan::details::VULKAN_GL::conv2d(
        voutput,
        vinput,
        vweight,
        c2ds.KH,
        c2ds.KW,
        vbias,
        c2ds.SY,
        c2ds.SX,
        c2ds.PY,
        c2ds.PX,
        c2ds.DY,
        c2ds.DX,
        groups);
  } else {
    at::native::vulkan::details::VULKAN_GL::conv2d(
        voutput,
        vinput,
        vweight,
        c2ds.KH,
        c2ds.KW,
        hasBias ? c10::make_optional((*bias).template data_ptr<float>())
                : c10::nullopt,
        c2ds.SY,
        c2ds.SX,
        c2ds.PY,
        c2ds.PX,
        c2ds.DY,
        c2ds.DX,
        groups);
  }
  return new_with_vtensor_vulkan(std::move(voutput), input.options());
}

Tensor vulkan_addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  VTensor& t = vtensor_from_vulkan(self);
  VTensor& m1 = vtensor_from_vulkan(mat1);
  VTensor& m2 = vtensor_from_vulkan(mat2);
  float b = beta.to<float>();
  float a = alpha.to<float>();

  VTensor output = VTensor{self.sizes().vec()};
  output.allocate_storage();
  at::native::vulkan::details::VULKAN_GL::addmm(output, t, m1, m2, b, a);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

Tensor vulkan_clamp(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  VTensor& x = vtensor_from_vulkan(self);
  VTensor output = VTensor{self.sizes().vec()};
  output.allocate_storage();
  float minValue = min.has_value() ? min.value().to<float>()
                                   : std::numeric_limits<float>::min();
  float maxValue = max.has_value() ? max.value().to<float>()
                                   : std::numeric_limits<float>::max();
  at::native::vulkan::details::VULKAN_GL::clamp(output, x, minValue, maxValue);
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
  VTensor& x = vtensor_from_vulkan(self);
  auto sizes = self.sizes();
  std::vector<int64_t> outputSizes{sizes[0], sizes[1]};
  VTensor output = VTensor{outputSizes};
  output.allocate_storage();
  at::native::vulkan::details::VULKAN_GL::mean(output, x);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

} // namespace native
} // namespace at
