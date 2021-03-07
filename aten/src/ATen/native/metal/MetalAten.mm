#import <ATen/native/metal/MetalTensor.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNOps.h>

#include <ATen/metal/Context.h>
#include <torch/script.h>

namespace at {
namespace native {
namespace metal {

at::Tensor& copy_from_metal_(at::Tensor& dst, const at::Tensor& src) {
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::Metal,
      "copy_from_metal input tensor's device is not metal");
  TORCH_INTERNAL_ASSERT(
      dst.device().is_cpu(),
      "copy_from_metal is implemented only for CPU device output");
  TORCH_INTERNAL_ASSERT(
      dst.layout() == Layout::Strided,
      "copy_from_metal is implemented only for Strided layout output");
  TORCH_INTERNAL_ASSERT(
      dst.scalar_type() == ScalarType::Float,
      "copy_from_metal is implemented only for float dtype output, got:",
      dst.scalar_type());
  TORCH_INTERNAL_ASSERT(
      dst.is_contiguous(),
      "copy_from_metal is implemented only for contiguous output tensor");

  MetalTensor& mtensor = MetalTensor::fromTensor(src);
  mtensor.copy_data_to_host(dst.data_ptr<float>());
  return dst;
}

at::Tensor& copy_to_metal_(at::Tensor& dst, const at::Tensor& src) {
  TORCH_INTERNAL_ASSERT(
      dst.device().type() == DeviceType::Metal,
      "copy_to_metal_ output tensor's device is not metal");
  TORCH_INTERNAL_ASSERT(
      src.device().is_cpu(),
      "copy_to_metal_ is implemented only for CPU device input");
  TORCH_INTERNAL_ASSERT(
      src.layout() == Layout::Strided,
      "copy_to_metal_ is implemented only for Strided layout input");
  TORCH_INTERNAL_ASSERT(
      src.scalar_type() == ScalarType::Float,
      "copy_to_metal_ is implemented only for float dtype");
  auto cpu_tensor_contiguous = src.contiguous();
  MetalTensor& mtensor = MetalTensor::fromTensor(dst);
  mtensor.set_data_from_host(cpu_tensor_contiguous.data_ptr<float>());
  return dst;
}

at::Tensor& metal_copy_impl_(at::Tensor& dst, const at::Tensor& src) {
  if (src.device().type() == at::kMetal && dst.device().type() == at::kCPU) {
    return copy_from_metal_(dst, src);
  }
  if (src.device().type() == at::kCPU && dst.device().type() == at::kMetal) {
    return copy_to_metal_(dst, src);
  }
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::Metal,
      "metal_copy_ is implemented only for CPU,Strided,float->Metal; Metal->CPU,Strided,float");
  return dst;
}

#pragma mark - ATen Ops

Tensor empty(
    IntArrayRef size,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory,
    c10::optional<MemoryFormat> memory_format) {
  TORCH_CHECK(
      !pin_memory.has_value(),
      "'pin_memory' argument is incompatible with Metal tensor");
  TORCH_CHECK(
      !memory_format.has_value(),
      "'memory_format' argument is incompatible with Metal tensor");
  MetalTensor mt{size.vec()};
  return MetalTensor::toTensor(
      std::move(mt), at::device(at::kMetal).dtype(dtype));
};

at::Tensor empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory) {
  TORCH_CHECK(
      !pin_memory.has_value() || !pin_memory.value(),
      "'pin_memory' argument is incompatible with Metal tensor");
  MetalTensor mt{size.vec(), stride.vec()};
  return MetalTensor::toTensor(
      std::move(mt), at::device(at::kMetal).dtype(dtype));
}

Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2);
  TORCH_CHECK(beta.toFloat() == 1.0f);
  TORCH_CHECK(alpha.toFloat() == 1.0f);
  auto&& sizes = weight.sizes();
  at::Tensor transposedWeight = weight.t().contiguous();
  at::Tensor mWeight =
      transposedWeight.view({sizes[1], sizes[0], 1, 1}).contiguous();
  return mpscnn::addmm(bias, input, mWeight);
}

Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(input.is_metal());
  Conv2DParams params{
      input.sizes(), weight.sizes(), padding, stride, dilation, groups};
  TORCH_INTERNAL_ASSERT(input.dim() == 4, "Expected 4-dimensional input");
  TORCH_INTERNAL_ASSERT(weight.dim() == 4, "Expected 4-dimensional weight");
  TORCH_CHECK(weight.device().type() == kCPU);
  return mpscnn::conv2d(input, weight, bias, params);
}

Tensor log_softmax_int(
    const Tensor& input,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  TORCH_CHECK(dim == 1);
  return mpscnn::log_softmax_int(input);
}

Tensor max_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(
      dilation[0] == dilation[1] == 1, "dilation is not supported on MPSCNN");
  return mpscnn::max_pool2d(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}

Tensor relu(const Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::relu(input);
}

Tensor& relu_(Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::relu_(input);
}

Tensor sigmoid(const Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::sigmoid(input);
}

Tensor& hardsigmoid_(Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::hardsigmoid_(input);
}

Tensor& hardswish_(Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::hardswish_(input);
}

Tensor t(const Tensor& input) {
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(input.dim() == 2);
  return mpscnn::t(input);
}

Tensor view(const Tensor& input, IntArrayRef size) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::view(input, size);
}

Tensor cat(const TensorList inputs, int64_t dim) {
  return mpscnn::cat(inputs, dim);
}

Tensor upsample_nearest2d_vec(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::upsample_nearest2d_vec(input, output_size, scale_factors);
}

Tensor add_Tensor(const Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  TORCH_CHECK(input1.sizes()[2] == input2.sizes()[2]);
  TORCH_CHECK(input1.sizes()[3] == input2.sizes()[3]);
  return mpscnn::add(input1, input2.is_metal() ? input2 : input2.metal());
}

Tensor& add__Tensor(Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  TORCH_CHECK(input1.sizes()[2] == input2.sizes()[2]);
  TORCH_CHECK(input1.sizes()[3] == input2.sizes()[3]);
  return mpscnn::add_(input1, input2.is_metal() ? input2 : input2.metal());
}

Tensor sub_Tensor(const Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  TORCH_CHECK(input2.sizes()[2] == input2.sizes()[3] == 1);
  return mpscnn::sub(input1, input2.is_metal() ? input2 : input2.metal());
}

Tensor mul_Tensor(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  TORCH_CHECK(input2.sizes()[2] == input2.sizes()[3] == 1);
  return mpscnn::mul(input1, input2.is_metal() ? input2 : input2.metal());
}

Tensor adaptive_avg_pool2d(const Tensor& input, IntArrayRef output_size) {
  // averages across the width and height, and outputs a 1x1xC image.
  TORCH_CHECK(output_size[0] == 1 && output_size[1] == 1);
  TORCH_CHECK(input.is_metal());
  return mpscnn::global_avg_pool2d(input, output_size);
}

Tensor& hardtanh_(Tensor& input, const Scalar& min_val, const Scalar& max_val) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::hardtanh_(input, min_val, max_val);
}

Tensor reshape(const Tensor& input, IntArrayRef shape) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::reshape(input, shape);
}

Tensor flatten_using_ints(
    const Tensor& input,
    int64_t start_dim,
    int64_t end_dim) {
  TORCH_CHECK(input.is_metal());
  return mpscnn::flatten_using_ints(input, start_dim, end_dim);
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("conv2d", TORCH_FN(conv2d));
  m.impl("add.Tensor", TORCH_FN(add_Tensor));
  m.impl("add_.Tensor", TORCH_FN(add__Tensor));
  m.impl("addmm", TORCH_FN(addmm));
  m.impl("empty.memory_format", empty);
  m.impl("empty_strided", TORCH_FN(empty_strided));
  m.impl("log_softmax.int", TORCH_FN(log_softmax_int));
  m.impl("max_pool2d", TORCH_FN(max_pool2d));
  m.impl("mul.Tensor", TORCH_FN(mul_Tensor));
  m.impl("relu", TORCH_FN(relu));
  m.impl("relu_", TORCH_FN(relu_));
  m.impl("sigmoid", TORCH_FN(sigmoid));
  m.impl("hardsigmoid_", TORCH_FN(hardsigmoid_));
  m.impl("hardswish_", TORCH_FN(hardswish_));
  m.impl("sub.Tensor", TORCH_FN(sub_Tensor));
  m.impl("upsample_nearest2d.vec", TORCH_FN(upsample_nearest2d_vec));
  m.impl("view", TORCH_FN(view));
  m.impl("_cat", TORCH_FN(cat));
  m.impl("adaptive_avg_pool2d", TORCH_FN(adaptive_avg_pool2d));
  m.impl("hardtanh_", TORCH_FN(hardtanh_));
  m.impl("reshape", TORCH_FN(reshape));
  m.impl("flatten.using_ints", TORCH_FN(flatten_using_ints));
}

} // namespace metal
} // namespace native

struct MetalImpl : public at::metal::MetalInterface {
  bool is_metal_available() const override {
#if defined(USE_PYTORCH_METAL)
    return [[MPSCNNContext sharedInstance] available];
#else
    return false;
#endif
  }
  at::Tensor& metal_copy_(at::Tensor& input, const at::Tensor& src)
      const override {
    TORCH_CHECK(
        is_metal_available(), "Metal is not available on the current device");
    return native::metal::metal_copy_impl_(input, src);
  }
};
#if defined(USE_PYTORCH_METAL)
static at::metal::MetalImplRegistrar g_metal_impl(new MetalImpl());
#endif

} // namespace at
