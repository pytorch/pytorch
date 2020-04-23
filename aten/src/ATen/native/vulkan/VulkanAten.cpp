#include <ATen/ATen.h>

#if defined(USE_VULKAN) || defined(USE_GLES)

#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/OpaqueTensorImpl.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/utils/ParamUtils.h>

#ifdef USE_VULKAN
#include <ATen/native/vulkan/Vulkan.h>
#else

#ifdef USE_GLES
#include <ATen/native/vulkan/GLES.h>
#endif

#endif

namespace at {
namespace native {

template <typename T>
struct CAFFE2_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
 private:
  T target_;

 public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target) : target_(target) {}
  IntrusivePtrTargetWrapper(T&& target) : target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

#ifdef USE_VULKAN
using VTensor = at::native::vulkan::details::vulkan::VulkanTensor;
#else

#ifdef USE_GLES
using VTensor = at::native::vulkan::details::gl::GLTensor;
#endif

#endif

using VTensorWrapper = IntrusivePtrTargetWrapper<VTensor>;
using VTensorWrapperPtr = c10::intrusive_ptr<VTensorWrapper>;
using VulkanTensorImpl = OpaqueTensorImpl<VTensorWrapperPtr>;
using VulkanTensor = at::Tensor;

at::Tensor new_with_vtensor_vulkan(VTensor&& vt, const TensorOptions& options) {
  auto dims = vt.sizes();
  VTensorWrapperPtr handle = c10::make_intrusive<VTensorWrapper>(std::move(vt));
  return detail::make_tensor<VulkanTensorImpl>(
      DispatchKeySet(DispatchKey::VulkanTensorId),
      options.dtype(),
      at::Device(at::kVulkan),
      handle,
      std::vector<int64_t>(dims.begin(), dims.end()));
}

VTensor& vtensor_from_vulkan(const VulkanTensor& vulkan_tensor) {
  AT_ASSERTM(
      vulkan_tensor.is_vulkan(), "vulkan_to_dense expects Vulkan tensor input");
  VulkanTensorImpl* impl =
      static_cast<VulkanTensorImpl*>(vulkan_tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle()->get_target();
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

at::Tensor vulkan_to_dense(const at::Tensor& vulkan_tensor) {
  VTensor& vtensor = vtensor_from_vulkan(vulkan_tensor);
  auto dims = vtensor.sizes();
  Tensor cpu_tensor = at::empty(
      std::vector<int64_t>(dims.begin(), dims.end()),
      vulkan_tensor.options()
          .device(at::Device(at::kCPU))
          .layout(c10::kStrided));
  float* tensorOutputData = cpu_tensor.template data_ptr<float>();
  vtensor.copyDataToHost(tensorOutputData);
  return cpu_tensor;
}

at::Tensor dense_to_vulkan(const at::Tensor& cpu_tensor) {
  AT_ASSERTM(
      cpu_tensor.device().type() == DeviceType::CPU,
      "dense_to_vulkan expects CPU tensor input");
  AT_ASSERTM(
      cpu_tensor.layout() == Layout::Strided,
      "dense_to_vulkan_expects strided tensor input");
  AT_ASSERTM(
      cpu_tensor.scalar_type() == ScalarType::Float,
      "dense_to_vulkan expects float tensor input");
  AT_ASSERTM(cpu_tensor.dim() == 4, "dense_to_vulkan expects tensor dim == 4");

  auto cpu_tensor_cont = cpu_tensor.contiguous();
  Tensor vulkan_tensor =
      empty_vulkan(cpu_tensor_cont.sizes(), cpu_tensor_cont.options(), {});

  VTensor& vtensor = vtensor_from_vulkan(vulkan_tensor);
  vtensor.setDataFromHost(cpu_tensor_cont.template data_ptr<float>());
  return vulkan_tensor;
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
  y.allocateStorage();

#ifdef USE_GLES
  at::native::vulkan::details::gl::upsample_nearest2d(
      y, x, ih, iw, oh, ow, in, ic, height_scale, width_scale);
#else
  // XXX Not implemented
  AT_ERROR("upsample_nearest2d_vulkan: Not implemented yet for Vulkan");
#endif
  return output;
}

Tensor vulkan_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  VTensor& x = vtensor_from_vulkan(self);
  VTensor& y = vtensor_from_vulkan(other);
  float a = alpha.to<float>();

  VTensor output = VTensor{self.sizes().vec()};
  output.allocateStorage();

#ifdef USE_GLES
  at::native::vulkan::details::gl::add(output, x, y, a);
#else
  // XXX Not implemented
  AT_ERROR("vulkan_add: Not implemented yet for Vulkan");
#endif

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
  TORCH_INTERNAL_ASSERT(
      groups == 1,
      "vulkan_convolution: group convolutions are not supported yet");

  int64_t N = isizes[0];
  int64_t C = isizes[1];
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
  voutput.allocateStorage();

  float* biasData{};
  if (bias.defined()) {
    biasData = bias.template data_ptr<float>();
  } else {
    biasData = (float*)std::malloc(sizeof(float) * OC);
    std::memset(biasData, 0, sizeof(float) * OC);
  }

  float* weightData = weight.template data_ptr<float>();

#ifdef USE_GLES
  at::native::vulkan::details::gl::conv2d(
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
#else
  // XXX Not implemented
  AT_ERROR("vulkan_convolution: Not implemented yet for Vulkan");
#endif
  return new_with_vtensor_vulkan(std::move(voutput), input.options());
}

} // namespace native
} // namespace at

#else
// if not (defined(USE_VULKAN) || defined(USE_GLES))

namespace at {
namespace native {

at::Tensor vulkan_to_dense(const at::Tensor& vulkan_tensor) {
  AT_ERROR("vulkan_to_dense: ATen not compiled with Vulkan or GLES support");
}

at::Tensor dense_to_vulkan(const at::Tensor& cpu_tensor) {
  AT_ERROR("dense_to_vulkan: ATen not compiled with Vulkan or GLES support");
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

} // namespace native
} // namespace at
#endif
