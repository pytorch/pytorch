#include <ATen/native/vulkan/ops/McLarenEncoderBlock.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

McLarenEncoderBlockOpContext::McLarenEncoderBlockOpContext(
      const Tensor& weight_1,
      const c10::optional<Tensor>& bias_1,
      IntArrayRef stride_1,
      IntArrayRef padding_1,
      IntArrayRef dilation_1,
      int64_t groups_1,
      const Tensor& weight_2,
      const c10::optional<Tensor>& bias_2,
      IntArrayRef stride_2,
      IntArrayRef padding_2,
      IntArrayRef dilation_2,
      int64_t groups_2)
  : packed_{
      pack_conv2d_weights(weight_1, Conv2dSlidingWindow),
      pack_conv2d_biases(bias_1, weight_1),
      pack_conv2d_filter(weight_1, expand_param_if_needed(dilation_1, "dilation", 2)),
      pack_conv2d_params(expand_param_if_needed(stride_1, "stride", 2)),
      pack_conv2d_params(expand_param_if_needed(padding_1, "padding", 2)),
      pack_conv2d_params(expand_param_if_needed(dilation_1, "dilation", 2)),
      safe_downcast<int32_t>(groups_1),
      pack_conv2d_weights(weight_2, Conv2dSlidingWindow),
      pack_conv2d_biases(bias_2, weight_2),
      pack_conv2d_filter(weight_2, expand_param_if_needed(dilation_2, "dilation", 2)),
      pack_conv2d_params(expand_param_if_needed(stride_2, "stride", 2)),
      pack_conv2d_params(expand_param_if_needed(padding_2, "padding", 2)),
      pack_conv2d_params(expand_param_if_needed(dilation_2, "dilation", 2)),
      safe_downcast<int32_t>(groups_2),
    },
    unpacked_{
      weight_1,
      bias_1,
      weight_1.sizes().vec(),
      stride_1.vec(),
      padding_1.vec(),
      dilation_1.vec(),
      groups_1,
      weight_2,
      bias_2,
      weight_2.sizes().vec(),
      stride_2.vec(),
      padding_2.vec(),
      dilation_2.vec(),
      groups_2,
    } {
}

McLarenEncoderBlockOpContext McLarenEncoderBlockOpContext::create(
      const Tensor& weight_1,
      const c10::optional<Tensor>& bias_1,
      IntArrayRef stride_1,
      IntArrayRef padding_1,
      IntArrayRef dilation_1,
      int64_t groups_1,
      const Tensor& weight_2,
      const c10::optional<Tensor>& bias_2,
      IntArrayRef stride_2,
      IntArrayRef padding_2,
      IntArrayRef dilation_2,
      int64_t groups_2) {
  // Pass in the originals
  return McLarenEncoderBlockOpContext{
    weight_1,
    bias_1,
    stride_1,
    padding_1,
    dilation_1,
    groups_1,
    weight_2,
    bias_2,
    stride_2,
    padding_2,
    dilation_2,
    groups_2,
  };
}

void McLarenEncoderBlockOpContext::conv2d_sliding_window(
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    std::array<int64_t, 4> filter,
    std::vector<int64_t> orig_filter,
    std::array<int64_t, 2> stride,
    std::array<int64_t, 2> padding,
    std::array<int64_t, 2> dilation) const {
  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    const struct Block final {
      uvec3 extents;
      int32_t ic4;
      ivec4 kernel;
      ivec2 ikernel;
      ivec2 stride;
      ivec2 padding;
      ivec2 dilate;
      vec2 clamp;
      ivec4 src_filter;
    } block {
      v_output.extents(),
      safe_downcast<int32_t>(filter[Layout::Filter::input]),
      {
        safe_downcast<int32_t>(filter[Layout::Filter::width]),
        safe_downcast<int32_t>(filter[Layout::Filter::height]),
        safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::width]),
        safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::height]),
      },
      {
        safe_downcast<int32_t>(orig_filter[Layout::Filter::width]),
        safe_downcast<int32_t>(orig_filter[Layout::Filter::height]),
      },
      {
        safe_downcast<int32_t>(stride[Layout::Parameter::width]),
        safe_downcast<int32_t>(stride[Layout::Parameter::height]),
      },
      {
        safe_downcast<int32_t>(padding[Layout::Parameter::width]),
        safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      },
      {
        safe_downcast<int32_t>(dilation[Layout::Parameter::width]),
        safe_downcast<int32_t>(dilation[Layout::Parameter::height]),
      },
      {
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()
      },
    };

    uvec3 global_size = v_output.extents();

    context->dispatch(
        command_buffer,
        {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        VK_KERNEL(conv2d),
        global_size,
        adaptive_work_group_size(global_size),
        // Write-only access bypasses synchronization but inserts appropriate
        // barriers if necessary.
        v_output.image(
            command_buffer,
            vTensor::Stage::Compute,
            vTensor::Access::Write),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_input.image(
            command_buffer,
            vTensor::Stage::Compute),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_weight.image(
            command_buffer,
            vTensor::Stage::Compute),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_bias.image(
            command_buffer,
            vTensor::Stage::Compute),
        // Object lifetime is managed by the resource pool.
        // It is OK not to keep track of the handle.
        context->resource().pool.uniform(block).object);
  }
  command_pool.submit(context->gpu().queue, command_buffer);
}

Tensor McLarenEncoderBlockOpContext::run(const Tensor& input_arg_1, const Tensor& input_arg_2) const {
  api::Context* const context = api::context();

  const Tensor input_1 = input_arg_1.is_vulkan() ? input_arg_1 : input_arg_1.vulkan();
  const vTensor& v_input_1 = convert(input_1);
  const Tensor input_2 = input_arg_2.is_vulkan() ? input_arg_2 : input_arg_2.vulkan();
  const vTensor& v_input_2 = convert(input_2);

  std::vector<int64_t> input_1_size = input_1.sizes().vec();
  std::vector<int64_t> input_2_size = input_2.sizes().vec();
  TORCH_CHECK(input_1_size.size() == 4, "McLarenEncoderBlock: first input tensor must have exactly 4 dims")
  TORCH_CHECK(input_1_size.size() == 4, "McLarenEncoderBlock: second input tensor must have exactly 4 dims")
  TORCH_CHECK(input_1_size[2] == 1, "McLarenEncoderBlock: first input tensor must have height of exactly 1")
  TORCH_CHECK(input_2_size[2] == 1, "McLarenEncoderBlock: second input tensor must have height of exactly 1")
  std::vector<int64_t> conv_input_size(input_1_size);
  conv_input_size[2] = 2;
  std::vector<int64_t> output_size = conv_output_size(
    conv_input_size,
    unpacked_.filter_1,
    packed_.padding_1,
    packed_.stride_1,
    packed_.dilation_1
  );

  vTensor v_output{
    context,
    output_size,
    input_1.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    const struct Block final {
      uvec3 extents;
      int32_t ic4;
      ivec4 kernel;
      ivec2 ikernel;
      ivec2 stride;
      ivec2 padding;
      ivec2 dilate;
      vec2 clamp;
      ivec4 src_filter;
    } block {
      v_output.extents(),
      safe_downcast<int32_t>(packed_.filter_1[Layout::Filter::input]),
      {
        safe_downcast<int32_t>(packed_.filter_1[Layout::Filter::width]),
        safe_downcast<int32_t>(packed_.filter_1[Layout::Filter::height]),
        safe_downcast<int32_t>(conv_input_size[Layout::Activation4D::width]),
        safe_downcast<int32_t>(conv_input_size[Layout::Activation4D::height]),
      },
      {
        safe_downcast<int32_t>(unpacked_.filter_1[Layout::Filter::width]),
        safe_downcast<int32_t>(unpacked_.filter_1[Layout::Filter::height]),
      },
      {
        safe_downcast<int32_t>(packed_.stride_1[Layout::Parameter::width]),
        safe_downcast<int32_t>(packed_.stride_1[Layout::Parameter::height]),
      },
      {
        safe_downcast<int32_t>(packed_.padding_1[Layout::Parameter::width]),
        safe_downcast<int32_t>(packed_.padding_1[Layout::Parameter::height]),
      },
      {
        safe_downcast<int32_t>(packed_.dilation_1[Layout::Parameter::width]),
        safe_downcast<int32_t>(packed_.dilation_1[Layout::Parameter::height]),
      },
      {
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()
      },
    };

    uvec3 global_size = v_output.extents();

    context->dispatch(
        command_buffer,
        {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        VK_KERNEL(mclaren_encoder_block),
        global_size,
        adaptive_work_group_size(global_size),
        v_output.image(
            command_buffer,
            vTensor::Stage::Compute,
            vTensor::Access::Write),
        v_input_1.image(
            command_buffer,
            vTensor::Stage::Compute),
        v_input_2.image(
            command_buffer,
            vTensor::Stage::Compute),
        packed_.v_weight_1.image(
            command_buffer,
            vTensor::Stage::Compute),
        packed_.v_bias_1.image(
            command_buffer,
            vTensor::Stage::Compute),
        packed_.v_weight_2.image(
            command_buffer,
            vTensor::Stage::Compute),
        packed_.v_bias_2.image(
            command_buffer,
            vTensor::Stage::Compute),
        context->resource().pool.uniform(block).object);
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return convert(v_output);
}

McLarenEncoderBlockOpContext::State McLarenEncoderBlockOpContext::unpack() const {
  return McLarenEncoderBlockOpContext::State{
    unpacked_.weight_1,
    unpacked_.bias_1,
    unpacked_.stride_1,
    unpacked_.padding_1,
    unpacked_.dilation_1,
    unpacked_.groups_1,
    unpacked_.weight_2,
    unpacked_.bias_2,
    unpacked_.stride_2,
    unpacked_.padding_2,
    unpacked_.dilation_2,
    unpacked_.groups_2,
  };
}

c10::intrusive_ptr<McLarenEncoderBlockOpContext> mclaren_encoder_block_prepack(
    Tensor&& weight_1,
    c10::optional<Tensor>&& bias_1,
    std::vector<int64_t>&& stride_1,
    std::vector<int64_t>&& padding_1,
    std::vector<int64_t>&& dilation_1,
    const int64_t groups_1,
    Tensor&& weight_2,
    c10::optional<Tensor>&& bias_2,
    std::vector<int64_t>&& stride_2,
    std::vector<int64_t>&& padding_2,
    std::vector<int64_t>&& dilation_2,
    const int64_t groups_2) {
  return c10::make_intrusive<McLarenEncoderBlockOpContext>(
      McLarenEncoderBlockOpContext::create(
          std::move(weight_1),
          std::move(bias_1),
          std::move(stride_1),
          std::move(padding_1),
          std::move(dilation_1),
          groups_1,
          std::move(weight_2),
          std::move(bias_2),
          std::move(stride_2),
          std::move(padding_2),
          std::move(dilation_2),
          groups_2));
}

Tensor mclaren_encoder_block_run(
    const Tensor& input_1,
    const Tensor& input_2,
    const c10::intrusive_ptr<McLarenEncoderBlockOpContext>& context) {
  return context->run(input_1, input_2);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
