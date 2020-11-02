#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor adaptive_avg_pool2d(const at::Tensor& input_arg, IntArrayRef output_size) {
  TORCH_INTERNAL_ASSERT(
      input_arg.dim() == 4,
      "vulkan_adaptive_avg_pool2d expects 4-dimensional input");

  api::Context* const context = api::context();
  const vTensor& v_input = convert(input_arg);
  vTensor v_output{
    context,
    {input_arg.sizes()[0], input_arg.sizes()[1], output_size[0], output_size[1]},
    input_arg.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_input.has_image()) {
      const struct {
        uint32_t input_width, input_height, output_width, output_height;
      } block {
        input_arg.sizes()[3],
        input_arg.sizes()[2],
        output_size[1],
        output_size[0],
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(adaptive_avg_pool2d),
          v_output.extents(),
          v_output.image(command_buffer, vTensor::Access::Write),
          v_input.image(command_buffer),
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
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
  const int kernel_height = safe_downcast<int>(kernel_size[0]);
  const int kernel_width =
      kernel_size.size() == 1 ? kernel_height : safe_downcast<int>(kernel_size[1]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kernel_height : safe_downcast<int>(stride[0]);
  const int dW = stride.empty()
      ? kernel_width
      : stride.size() == 1 ? dH : safe_downcast<int>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int>(padding[1]);

  const int64_t input_batch = self.sizes()[0];
  const int64_t input_channels = self.sizes()[1];
  const int64_t input_height = self.sizes()[2];
  const int64_t input_width = self.sizes()[3];

  const int64_t output_height =
      pooling_output_shape<int64_t>(input_height, kernel_height, padH, dH, 1, ceil_mode);
  const int64_t output_width =
      pooling_output_shape<int64_t>(input_width, kernel_width, padW, dW, 1, ceil_mode);

  pool2d_shape_check(
      self, kernel_height, kernel_width, dH, dW, padH, padW, 1, 1, input_channels, input_height, input_width, output_height, output_width);

  api::Context* const context = api::context();

  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    {input_batch, input_channels, output_height, output_width},
    self.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_self.has_image()) {
      const struct {
        uint32_t input_width, input_height, input_channels, input_size_stub;
        uint32_t output_width, output_height, output_channels, output_size_stub;
        uint32_t kernel_width, kernel_height;
        uint32_t stride_x, stride_y;
        uint32_t padding_x, padding_y;
        uint32_t dilate_x, dilate_y;
      } block {
        input_width, input_height, input_batch * input_channels, 0u,
        output_width, output_height, input_batch * input_channels, 0u,
        kernel_width, kernel_height,
        dW, dH,
        padW, padH,
        1u, 1u
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(avg_pool2d),
          v_output.extents(),
          v_output.image(command_buffer, vTensor::Access::Write),
          v_self.image(command_buffer),
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);

}
#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("_adaptive_avg_pool2d", TORCH_FN(adaptive_avg_pool2d));
  m.impl("avg_pool2d", TORCH_FN(avg_pool2d));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
