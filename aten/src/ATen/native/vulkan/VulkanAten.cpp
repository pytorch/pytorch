#include <ATen/native/vulkan/VulkanAten.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InferSize.h>
#include <ATen/native/Pool.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/Vulkan.h>
#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>
#include <ATen/native/vulkan/VulkanOps.h>
#include <ATen/vulkan/Context.h>

namespace at {
namespace native {
namespace vulkan {
namespace aten {
using at::native::vulkan::detail::VulkanTensor;
using VulkanTensorImpl = VulkanOpaqueTensorImpl<VulkanTensor>;

namespace {
int64_t normalize_dim(int64_t d, int64_t n) {
  return (d % n + n) % n;
}
} // namespace

Tensor new_with_vtensor_vulkan(
    VulkanTensor&& vt,
    const TensorOptions& options) {
  auto sizes = vt.sizes();
  auto strides = vt.strides();
  return at::detail::make_tensor<VulkanTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      options.dtype(),
      at::Device(at::kVulkan),
      std::move(vt),
      std::vector<int64_t>(sizes.begin(), sizes.end()),
      std::vector<int64_t>(strides.begin(), strides.end()));
}

const VulkanTensor& vtensor_from_vulkan(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      tensor.is_vulkan(), "vtensor_from_vulkan expects Vulkan tensor input");
  VulkanTensorImpl* const impl =
      static_cast<VulkanTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle();
}

VulkanTensor& vtensor_from_vulkan(Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      tensor.is_vulkan(), "vtensor_from_vulkan expects Vulkan tensor input");
  VulkanTensorImpl* const impl =
      static_cast<VulkanTensorImpl*>(tensor.unsafeGetTensorImpl());
  return impl->unsafe_opaque_handle();
}

Tensor empty(
    IntArrayRef size,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory,
    const optional<MemoryFormat> memory_format) {
  TORCH_CHECK(
      !pin_memory.has_value(),
      "'pin_memory' argument is incompatible with Vulkan tensor");
  TORCH_CHECK(
      !memory_format.has_value(),
      "'memory_format' argument is incompatible with Vulkan tensor");
  VulkanTensor vt{size.vec()};
  return new_with_vtensor_vulkan(
      std::move(vt), at::device(at::kVulkan).dtype(dtype));
}

Tensor empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory) {
  return vulkan::aten::empty(
      size, dtype, layout, device, pin_memory, c10::nullopt);
}

Tensor upsample_nearest2d(
    const Tensor& input,
    const IntArrayRef outputSizes,
    const c10::optional<double> scales_h,
    const c10::optional<double> scales_w) {
  const auto& x = vtensor_from_vulkan(input);
  const auto inputSizes = input.sizes();
  const auto in = inputSizes[0];
  const auto ic = inputSizes[1];
  const auto ih = inputSizes[2];
  const auto iw = inputSizes[3];

  const auto oh = outputSizes[0];
  const auto ow = outputSizes[1];
  const float height_scale = compute_scales_value<float>(scales_h, ih, oh);
  const float width_scale = compute_scales_value<float>(scales_w, iw, ow);
  VulkanTensor output{{in, ic, oh, ow}};
  vulkan::detail::upsample_nearest2d(
      output, x, ih, iw, oh, ow, in, ic, height_scale, width_scale);
  return new_with_vtensor_vulkan(std::move(output), input.options());
}

Tensor adaptive_avg_pool2d(const at::Tensor& input, IntArrayRef outputSize) {
  TORCH_INTERNAL_ASSERT(
      input.dim() == 4,
      "vulkan_adaptive_avg_pool2d expects 4-dimensional input");
  const auto& x = vtensor_from_vulkan(input);
  const auto inputSize = input.sizes();
  const auto in = inputSize[0];
  const auto ic = inputSize[1];
  const auto ih = inputSize[2];
  const auto iw = inputSize[3];

  const auto oh = outputSize[0];
  const auto ow = outputSize[1];
  VulkanTensor output{{in, ic, oh, ow}};
  vulkan::detail::adaptive_avg_pool2d(output, x, ih, iw, oh, ow, in, ic);
  return new_with_vtensor_vulkan(std::move(output), input.options());
}

Tensor avg_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int>(kernel_size[0]);
  const int kW =
      kernel_size.size() == 1 ? kH : safe_downcast<int>(kernel_size[1]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int>(stride[0]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int>(padding[1]);

  const auto& x = vtensor_from_vulkan(self);
  auto inputSize = self.sizes();
  const int64_t iN = inputSize[0];
  const int64_t iC = inputSize[1];
  const int64_t iH = inputSize[2];
  const int64_t iW = inputSize[3];

  const int64_t oH =
      pooling_output_shape<int64_t>(iH, kH, padH, dH, 1, ceil_mode);
  const int64_t oW =
      pooling_output_shape<int64_t>(iW, kW, padW, dW, 1, ceil_mode);

  pool2d_shape_check(
      self, kH, kW, dH, dW, padH, padW, 1, 1, iC, iH, iW, oH, oW, self.suggest_memory_format());

  VulkanTensor y{{iN, iC, oH, oW}};
  vulkan::detail::avg_pool2d(
      y, x, iH, iW, oH, oW, iN, iC, kH, kW, dH, dW, padH, padW);
  return new_with_vtensor_vulkan(std::move(y), self.options());
}

Tensor max_pool2d(
    const at::Tensor& self,
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "Vulkan max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int>(kernel_size[0]);
  const int kW =
      kernel_size.size() == 1 ? kH : safe_downcast<int>(kernel_size[1]);
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "Vulkan max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int>(stride[0]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "Vulkan max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "Vulkan max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int>(dilation[0]);
  const int dilationW =
      dilation.size() == 1 ? dilationH : safe_downcast<int>(dilation[1]);
  TORCH_CHECK(
      self.dim() == 4, "Vulkan max_pool2d is implemented for 4-dim input");

  const auto& x = vtensor_from_vulkan(self);
  const auto inputSize = self.sizes();
  const int64_t iN = inputSize[0];
  const int64_t iC = inputSize[1];
  const int64_t iH = inputSize[2];
  const int64_t iW = inputSize[3];

  const int64_t oH =
      pooling_output_shape<int64_t>(iH, kH, padH, dH, dilationH, ceil_mode);
  const int64_t oW =
      pooling_output_shape<int64_t>(iW, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(
      self,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      iC,
      iH,
      iW,
      oH,
      oW,
      self.suggest_memory_format());

  VulkanTensor y{{iN, iC, oH, oW}};
  vulkan::detail::max_pool2d(
      y,
      x,
      iH,
      iW,
      oH,
      oW,
      iN,
      iC,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW);
  return new_with_vtensor_vulkan(std::move(y), self.options());
}

Tensor reshape(at::Tensor const& input, IntArrayRef shape) {
  return new_with_vtensor_vulkan(
      vulkan::detail::reshape_copy(vtensor_from_vulkan(input), shape.vec()),
      input.options());
}

Tensor cat(const TensorList tensors, int64_t dim) {
  const auto norm_dim = normalize_dim(dim, 4);
  TORCH_INTERNAL_ASSERT(
      norm_dim == 0 || norm_dim == 1,
      "Vulkan cat is implemented only for batch and channels dimensions");
  at::Tensor tensor = tensors[0];
  int64_t cat_dim_size = 0;

  std::vector<VulkanTensor> vTensors{};
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    TORCH_INTERNAL_ASSERT(
        t.dim() == 4, "Vulkan cat expects 4 dimensional inputs");
    TORCH_INTERNAL_ASSERT(t.is_vulkan(), "Vulkan cat expects Vulkan inputs");

    for (int d = 0; d < 4; ++d) {
      if (d == dim) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          t.size(d) == tensor.size(d),
          "Vulkan cat inputs must have matching sizes except concatenated dimension");
    }
    vTensors.push_back(vtensor_from_vulkan(t));
    cat_dim_size += t.size(dim);
  }

  auto result_size = tensor.sizes().vec();
  result_size[dim] = cat_dim_size;

  VulkanTensor output{result_size};

  vulkan::detail::cat(output, vTensors, dim);
  return new_with_vtensor_vulkan(std::move(output), tensor.options());
}

Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  return new_with_vtensor_vulkan(
      vulkan::detail::transpose(vtensor_from_vulkan(self), dim0, dim1),
      self.options());
}

Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  auto& x = vtensor_from_vulkan(self);
  x = vulkan::detail::transpose(x, dim0, dim1);
  return self;
}

Tensor view(const Tensor& self, IntArrayRef size) {
  return new_with_vtensor_vulkan(
      vulkan::detail::reshape_copy(
          vtensor_from_vulkan(self), at::infer_size(size, self.numel())),
      self.options());
}

Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  return self;
}

Tensor slice(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  return new_with_vtensor_vulkan(
      vulkan::detail::slice(vtensor_from_vulkan(self), dim, start, end, step),
      self.options());
}

Tensor add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto xt = self.is_vulkan() ? self : self.vulkan();
  const auto& x = vtensor_from_vulkan(xt);
  auto yt = other.is_vulkan() ? other : other.vulkan();
  const auto& y = vtensor_from_vulkan(yt);
  const float a = alpha.to<float>();

  VulkanTensor output{self.sizes().vec()};
  vulkan::detail::add(output, x, y, a);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

VulkanTensor& vtensor(Tensor& t) {
  if (t.is_vulkan()) {
    return vtensor_from_vulkan(t);
  }
  auto tv = t.vulkan();
  return vtensor_from_vulkan(tv);
}

const VulkanTensor& vtensor(const Tensor& t) {
  if (t.is_vulkan()) {
    return vtensor_from_vulkan(t);
  }
  const auto tv = t.vulkan();
  return vtensor_from_vulkan(tv);
}

Tensor& add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto& x = vtensor(self);
  const auto& y = vtensor(other);
  float a = alpha.to<float>();

  VulkanTensor output{self.sizes().vec()};
  vulkan::detail::add(output, x, y, a);
  x = std::move(output);
  return self;
}

Tensor add_scalar(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  const auto& x = vtensor_from_vulkan(self);
  const float s = other.to<float>();
  const float a = alpha.to<float>();
  VulkanTensor output{self.sizes().vec()};
  vulkan::detail::add(output, x, s * a);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

Tensor mul_scalar(const Tensor& self, const Scalar& other) {
  const auto& x = vtensor_from_vulkan(self);
  const float s = other.to<float>();
  VulkanTensor output{self.sizes().vec()};
  vulkan::detail::mul(output, x, s);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  auto sliced = vulkan::aten::slice(self, dim, index, index + 1, 1);
  auto sizes = self.sizes().vec();
  sizes.erase(sizes.begin() + dim);
  return vulkan::aten::reshape(sliced, sizes);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  auto sizes = self.sizes().vec();
  sizes.insert(sizes.begin() + dim, 1);
  return vulkan::aten::reshape(self, sizes);
}

Tensor convolution(
    const Tensor& input, // Vulkan
    const Tensor& weight, // CPU
    const c10::optional<Tensor>& bias, // CPU
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef output_padding,
    const int64_t groups) {
  const vulkan::Conv2DParams params{
      input.sizes(), weight.sizes(), padding, stride, dilation, groups};
  TORCH_INTERNAL_ASSERT(
      input.dim() == 4, "convolution: Expected 4-dimensional input");
  TORCH_INTERNAL_ASSERT(
      weight.dim() == 4, "convolution: Expected 4-dimensional weight");
  TORCH_INTERNAL_ASSERT(
      groups == 1 || groups == params.C,
      "convolution: only nogroup or depthwise convolutions supported");
  TORCH_INTERNAL_ASSERT(!transposed, "convolution: transposed not supported");

  const VulkanTensor& vinput = vtensor_from_vulkan(input);
  VulkanTensor voutput = VulkanTensor{params.output_sizes()};

  vulkan::detail::conv2d(
      voutput,
      vinput,
      weight.data_ptr<float>(),
      (bias.has_value() && bias->defined())
          ? c10::make_optional<const float*>(bias->data_ptr<float>())
          : c10::nullopt,
      params);
  return new_with_vtensor_vulkan(std::move(voutput), input.options());
}

Tensor addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  const VulkanTensor t =
      vtensor_from_vulkan(self.is_vulkan() ? self : self.vulkan());
  const VulkanTensor m1 =
      vtensor_from_vulkan(mat1.is_vulkan() ? mat1 : mat1.vulkan());
  const VulkanTensor m2 =
      vtensor_from_vulkan(mat2.is_vulkan() ? mat2 : mat2.vulkan());
  const float b = beta.to<float>();
  const float a = alpha.to<float>();

  VulkanTensor output = VulkanTensor{self.sizes().vec()};
  vulkan::detail::addmm(output, t, m1, m2, b, a);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  TORCH_INTERNAL_ASSERT(
      self.dim() == 2 && mat2.dim() == 2,
      "vulkan_mm expects 2-dimensional tensors");
  const auto m1Sizes = self.sizes();
  const auto m2Sizes = mat2.sizes();
  TORCH_INTERNAL_ASSERT(
      m1Sizes[1] == m2Sizes[0],
      "vulkan_mm expects self.sizes[1] equal mat2.sizes[0]");

  const auto& m1 = vtensor_from_vulkan(self.is_vulkan() ? self : self.vulkan());
  const auto& m2 = vtensor_from_vulkan(mat2.is_vulkan() ? mat2 : mat2.vulkan());

  VulkanTensor output{{m1Sizes[0], m2Sizes[1]}};
  vulkan::detail::addmm(output, c10::nullopt, m1, m2, 0.f, 1.f);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

Tensor clamp(
    const Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  const auto& x = vtensor_from_vulkan(self);
  VulkanTensor output{self.sizes().vec()};
  vulkan::detail::clamp(
      output,
      x,
      min ? min.value().to<float>() : -std::numeric_limits<float>::infinity(),
      max ? max.value().to<float>() : std::numeric_limits<float>::infinity());
  return vulkan::aten::new_with_vtensor_vulkan(
      std::move(output), self.options());
}

Tensor& clamp_(
    Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  auto& x = vtensor_from_vulkan(self);
  VulkanTensor output{self.sizes().vec()};
  vulkan::detail::clamp(
      output,
      x,
      min ? min.value().to<float>() : -std::numeric_limits<float>::infinity(),
      max ? max.value().to<float>() : std::numeric_limits<float>::infinity());
  x = std::move(output);
  return self;
}

Tensor hardtanh(const Tensor& self, const Scalar& min, const Scalar& max) {
  return vulkan::aten::clamp(self, min, max);
}

Tensor& hardtanh_(Tensor& self, const Scalar& min, const Scalar& max) {
  return vulkan::aten::clamp_(self, min, max);
}

Tensor& relu_(Tensor& self) {
  return vulkan::aten::clamp_(self, 0, nullopt);
}

Tensor mean(
    const Tensor& self,
    const IntArrayRef dim,
    const bool keepdim,
    const optional<ScalarType> dtype) {
  TORCH_INTERNAL_ASSERT(!keepdim, "keepdim not implemented for Vulkan mean");
  TORCH_INTERNAL_ASSERT(self.is_vulkan(), "mean expects Vulkan tensor input");

  // Mean is implemented only for HW dimensions of 4-d tensor
  TORCH_INTERNAL_ASSERT(self.dim() == 4);
  static const std::unordered_set<int64_t> expected_dims_set({2, 3});
  std::unordered_set<int64_t> dims_set;
  for (const auto& d : dim) {
    dims_set.insert(normalize_dim(d, 4));
  }
  TORCH_INTERNAL_ASSERT(expected_dims_set == dims_set);

  const auto& x = vtensor_from_vulkan(self);
  const auto sizes = self.sizes();
  VulkanTensor output{std::vector<int64_t>{sizes[0], sizes[1]}};
  vulkan::detail::mean(output, x);
  return new_with_vtensor_vulkan(std::move(output), self.options());
}

#ifndef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("slice.Tensor", TORCH_FN(at::native::vulkan::aten::slice));
  m.impl("view", TORCH_FN(at::native::vulkan::aten::reshape));
  m.impl("select.int", TORCH_FN(at::native::vulkan::aten::select));
  m.impl("transpose.int", TORCH_FN(at::native::vulkan::aten::transpose));
  m.impl("transpose_", at::native::vulkan::aten::transpose_);
  m.impl("view", TORCH_FN(at::native::vulkan::aten::view));
  m.impl("unsqueeze", TORCH_FN(at::native::vulkan::aten::unsqueeze));
  m.impl("empty.memory_format", at::native::vulkan::aten::empty);
  m.impl("empty_strided", TORCH_FN(at::native::vulkan::aten::empty_strided));
  m.impl("add.Tensor", TORCH_FN(at::native::vulkan::aten::add));
  m.impl("clamp", TORCH_FN(at::native::vulkan::aten::clamp));
  m.impl("mean.dim", TORCH_FN(at::native::vulkan::aten::mean));
  m.impl("mm", TORCH_FN(at::native::vulkan::aten::mm));
  m.impl("addmm", TORCH_FN(at::native::vulkan::aten::addmm));
  m.impl(
      "upsample_nearest2d",
      TORCH_FN(at::native::vulkan::aten::upsample_nearest2d));
  m.impl(
      "_adaptive_avg_pool2d",
      TORCH_FN(at::native::vulkan::aten::adaptive_avg_pool2d));
  m.impl("avg_pool2d", TORCH_FN(at::native::vulkan::aten::avg_pool2d));
  m.impl("max_pool2d", TORCH_FN(at::native::vulkan::aten::max_pool2d));
  m.impl("_cat", TORCH_FN(at::native::vulkan::aten::cat));
  m.impl("mul.Scalar", TORCH_FN(at::native::vulkan::aten::mul_scalar));
  m.impl("add.Scalar", TORCH_FN(at::native::vulkan::aten::add_scalar));
  m.impl(
      "convolution_overrideable", at::native::vulkan::aten::convolution);
  m.impl("hardtanh_", at::native::vulkan::aten::hardtanh_);
  m.impl("relu_", at::native::vulkan::aten::relu_);
  m.impl("add_.Tensor", at::native::vulkan::aten::add_);
}

#endif /* USE_VULKAN_API */

Tensor& copy_from_vulkan_(Tensor& self, const Tensor& src) {
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::Vulkan,
      "copy_from_vulkan input tensor's device is not Vulkan");
  TORCH_INTERNAL_ASSERT(
      self.device().is_cpu(),
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

  const auto& vtensor = vtensor_from_vulkan(src);
  vtensor.copy_data_to_host(self.data_ptr<float>());
  return self;
}

Tensor& copy_to_vulkan_(Tensor& self, const Tensor& src) {
  TORCH_INTERNAL_ASSERT(
      self.device().type() == DeviceType::Vulkan,
      "copy_to_vulkan output tensor's device is not Vulkan");
  TORCH_INTERNAL_ASSERT(
      src.device().is_cpu(),
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

Tensor& vulkan_copy_impl_(Tensor& self, const Tensor& src) {
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

struct VulkanImpl final : public at::vulkan::VulkanImplInterface {
  bool is_vulkan_available() const override {
    return at::native::vulkan::detail::is_available();
  }

  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return vulkan_copy_impl_(self, src);
  }
};
static at::vulkan::VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());

} // namespace aten

using detail::VulkanTensor;
Tensor convolution_prepack_weights(const Tensor& weight) {
  const auto wsizes = weight.sizes();
  TORCH_INTERNAL_ASSERT(
      wsizes.size() == 4,
      "convolution_prepack_weights: Expected 4-dimensional weight");

  const int64_t OC = wsizes[0];
  const int64_t C = wsizes[1];
  const int64_t KH = wsizes[2];
  const int64_t KW = wsizes[3];
  VulkanTensor voutput =
      VulkanTensor{{UP_DIV(OC, 4), UP_DIV(C, 4), KH * KW, 16}};

  vulkan::detail::conv2d_prepack_weights(
      voutput, weight.data_ptr<float>(), OC, C, KH, KW);
  return aten::new_with_vtensor_vulkan(
      std::move(voutput), at::device(at::kVulkan).dtype(at::kFloat));
}

Tensor convolution_prepacked(
    const Tensor& input, // Vulkan
    const IntArrayRef weightSizes,
    const Tensor& weight_prepacked_vulkan, // Vulkan
    const c10::optional<Tensor>& bias, // Vulkan|CPU
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    int64_t groups,
    const float output_min,
    const float output_max) {
  TORCH_INTERNAL_ASSERT(
      input.dim() == 4, "Vulkan convolution: Expected 4-dimensional input");
  TORCH_INTERNAL_ASSERT(
      weight_prepacked_vulkan.dim() == 4,
      "Vulkan convolution: Expected 4-dimensional weight");
  vulkan::Conv2DParams params{
      input.sizes(), weightSizes, padding, stride, dilation, groups};
  TORCH_INTERNAL_ASSERT(
      groups == 1 || groups == params.C,
      "Vulkan convolution: only nogroup or depthwise convolutions supported");
  const VulkanTensor& vinput = aten::vtensor_from_vulkan(input);
  const VulkanTensor& vweight =
      aten::vtensor_from_vulkan(weight_prepacked_vulkan);
  VulkanTensor voutput =
      VulkanTensor{{params.N, params.OC, params.OH, params.OW}};
  const bool hasBias = bias.has_value() && bias->defined();
  if (hasBias && bias->is_vulkan()) {
    const VulkanTensor& vbias = aten::vtensor_from_vulkan(*bias);
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
  return aten::new_with_vtensor_vulkan(std::move(voutput), input.options());
}

} // namespace vulkan
} // namespace native
} // namespace at
