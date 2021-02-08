#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor view(
    const Tensor& self_arg,
    const IntArrayRef shape) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    shape,
    self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    command_buffer.copy(
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_self.buffer(
            command_buffer,
            vTensor::Stage::Transfer),
        // Write-only access bypasses synchronization but inserts appropriate
        // barriers if necessary.
        v_output.buffer(
            command_buffer,
            vTensor::Stage::Transfer,
            vTensor::Access::Write));
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return convert(v_output);
}

Tensor as_strided(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  auto t_cpu = self.cpu();
  auto t_out_cpu = t_cpu.as_strided(size, stride, storage_offset_);
  return t_out_cpu.vulkan();
}

Tensor sub_Tensor(const Tensor& input1, const Tensor& input2, Scalar alpha) {
  auto input1_cpu = input1.cpu();
  auto input2_cpu = input2.cpu();
  return at::sub(input1_cpu, input2_cpu, alpha).vulkan();
}

Tensor div_Tensor(const Tensor& self, const Tensor& other) {
  return at::div(self.cpu(), other.cpu()).vulkan();
}

Tensor upsample_bilinear2d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  return at::upsample_bilinear2d(
      input.cpu(),
      output_size,
      align_corners,
      scale_factors).vulkan();
}

Tensor upsample_bilinear2d_2(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  return at::upsample_bilinear2d(
      input.cpu(),
      output_size,
      align_corners,
      scales_h, scales_w).vulkan();
}

Tensor new_full(
    const Tensor & self,
    IntArrayRef size,
    Scalar fill_value,
    c10::optional<ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  //TODO: fill with fill_value
  return at::empty(size, self.options().device(at::kCPU).dtype(dtype)).vulkan();
}

Tensor mul_Tensor(const Tensor& input1, const Tensor& input2) {
  return (input1.cpu() * input2.cpu()).vulkan();
}

Tensor& hardswish_(Tensor& self) {
  auto self_cpu = self.cpu();
  auto out_vulkan = at::hardswish_(self_cpu).vulkan();
  std::swap(out_vulkan, self);
  return self;
}

Tensor& hardsigmoid_(Tensor& self) {
  auto self_cpu = self.cpu();
  auto out_vulkan = at::hardsigmoid_(self_cpu).vulkan();
  std::swap(out_vulkan, self);
  return self;
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode) {
    auto out_cpu = at::max_pool2d_with_indices(input.cpu(), kernel_size, stride, padding, dilation, ceil_mode);
    return std::tuple<Tensor, Tensor>(
        std::get<0>(out_cpu).vulkan(),
        std::get<1>(out_cpu)
    );
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_vulkan(
    const Tensor& self,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    bool train, double momentum, double eps) {
        auto out_cpu = at::native_batch_norm(
            self,
            weight ? weight->cpu() : weight,
            bias ? bias->cpu() : bias,
            running_mean ? running_mean->cpu() : running_mean,
            running_var ? running_var->cpu() : running_var,
            train, momentum, eps);
        return std::tuple<Tensor, Tensor, Tensor>(
            std::get<0>(out_cpu).vulkan(),
            std::get<1>(out_cpu).vulkan(),
            std::get<2>(out_cpu).vulkan()
        );
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("view", TORCH_FN(view));
  m.impl("as_strided", TORCH_FN(as_strided));
  m.impl("sub.Tensor", TORCH_FN(sub_Tensor));
  m.impl("div.Tensor", TORCH_FN(div_Tensor));
  m.impl("mul.Tensor", TORCH_FN(mul_Tensor));
  m.impl("upsample_bilinear2d.vec", TORCH_FN(upsample_bilinear2d));
  m.impl("new_full", TORCH_FN(new_full));
  m.impl("hardswish_", TORCH_FN(hardswish_));
  m.impl("hardsigmoid_", TORCH_FN(hardsigmoid_));
  m.impl("max_pool2d_with_indices", TORCH_FN(max_pool2d_with_indices));
  m.impl("native_batch_norm", TORCH_FN(batch_norm_vulkan));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
