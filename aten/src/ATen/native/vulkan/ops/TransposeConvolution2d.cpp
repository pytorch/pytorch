#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/TransposeConvolution2d.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

vTensor pack_weights_2d_reverse(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    const Tensor& weight,
    bool reversed) {
  /* Source */
  const IntArrayRef src_filter = weight.sizes();
  const float* const src_weight_ptr = weight.data_ptr<float>();

  const int64_t src_kw_sz = src_filter[Layout::Filter::width];
  const int64_t src_kh_sz = src_filter[Layout::Filter::height];
  const int64_t src_kernel_sz = src_kw_sz * src_kh_sz;
  const int64_t src_block_sz = src_kernel_sz * src_filter[Layout::Filter::input];

  const int64_t num_stacks = div_up(src_filter[Layout::Filter::output], INT64_C(4));
  const int64_t stack_depth = api::utils::align_up(src_filter[Layout::Filter::input], INT64_C(4));

  /* Destination */
  const int64_t dst_kw_sz = src_kw_sz * stack_depth;
  const int64_t dst_kh_sz = src_kh_sz * num_stacks;
  const int64_t dst_kernel_sz = dst_kw_sz * dst_kh_sz;

  vTensor v_weight{
      context,
      {
          4,
          dst_kh_sz,
          dst_kw_sz,
      },
      weight.options(),
  };

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
  Future::Payload v_weight_payload = v_weight_future.wait();

  float* const dst_weight_ptr = v_weight_payload.get();
  memset(dst_weight_ptr, 0, v_weight.nbytes());

  for (const auto src_oc : c10::irange(src_filter[Layout::Filter::output])) {
    /* Source */
    const float* const src_weight_oc_ptr = src_weight_ptr + src_oc * src_block_sz;

    /* Destination */
    const int64_t dst_oh = src_oc / 4;
    const int64_t dst_c = src_oc % 4;

    float* const dst_weight_c_ptr = dst_weight_ptr + dst_c * dst_kernel_sz;

    for (const auto src_ic : c10::irange(src_filter[Layout::Filter::input])) {
      for (const auto src_ih : c10::irange(src_kh_sz)) {
        const int64_t dst_h = reversed ? (src_kh_sz - 1 - src_ih) : src_ih;
        for (const auto src_iw : c10::irange(src_kw_sz)) {
          const int64_t dst_w = reversed ? (src_kw_sz - 1 - src_iw) : src_iw;
          const int64_t dst_w_offset = dst_w * stack_depth;
          memcpy(
              dst_weight_c_ptr + (dst_oh * src_kh_sz + dst_h) * dst_kw_sz + src_ic + dst_w_offset,
              src_weight_oc_ptr + src_ic * src_kernel_sz + src_ih * src_kw_sz + src_iw,
              sizeof(float));
        }
      }
    }
  }

  return v_weight;
}

vTensor pack_weights(const Tensor& weight_arg) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();  // Don't collect the timestamp since the command buffer doesn't record anything

  const Tensor weight = at::permute(weight_arg, {1, 0, 2, 3}).contiguous();

  return pack_weights_2d_reverse(
      context,
      command_buffer,
      weight,
      true);
}

vTensor pack_biases(
    const c10::optional<Tensor>& bias,
    const Tensor& weight) {
  if (bias && bias->is_vulkan()) {
    return convert(*bias);
  }

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();  // Don't collect the timestamp since the command buffer doesn't record anything

  const int64_t src_w = weight.size(Layout::TransposedFilter::output);
  const int64_t packed_w = div_up(src_w, INT64_C(4));
  vTensor v_bias{
    context,
    {
      4,
      1,
      packed_w,
    },
    weight.options(),
  };

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  Future v_bias_future = v_bias.host<float, vTensor::Access::Write>(command_buffer);
  Future::Payload v_bias_payload = v_bias_future.wait();

  if (bias) {
    const float* const src_bias_ptr = bias->contiguous().data_ptr<float>();
    float* const dst_bias_ptr = v_bias_payload.get();

    memset(dst_bias_ptr, 0, v_bias.nbytes());
    for (const auto i : c10::irange(src_w)) {
      const int64_t c = i % 4;
      const int64_t x = i / 4;
      dst_bias_ptr[c * packed_w + x] = src_bias_ptr[i];
    }
  }
  else {
    memset(
        v_bias_payload.get(),
        // 2's complement integers and IEEE-754 floating point numbers both
        // have identical bit representations for 0, so can use memset which
        // only accepts uint8_t parameter.
        0,
        v_bias.nbytes());
  }

  return v_bias;
}

std::array<int64_t, 4> pack_filter(
    const Tensor& weight,
    const IntArrayRef dilation) {
  const IntArrayRef filter = weight.sizes();

  const auto effective = [](const int64_t k, const int64_t d) {
    return k + (k - 1) * (d - 1);
  };

  return {
    align_up(filter[Layout::TransposedFilter::output], INT64_C(4)),
    align_up(filter[Layout::TransposedFilter::input], INT64_C(4)),
    effective(
        filter[Layout::Filter::height],
        dilation[Layout::Parameter::height]),
    effective(
        filter[Layout::Filter::width],
        dilation[Layout::Parameter::width]),
  };
}

std::array<int64_t, 2> pack_params(const std::vector<int64_t>& vector) {
  TORCH_INTERNAL_ASSERT(2u == vector.size(), "Invalid usage!");

  return {
    vector[0],
    vector[1],
  };
}

bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef /* output_padding */,
    const IntArrayRef dilation,
    const bool transposed,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return api::available() &&
         // Weight
         (4 == weight.ndimension()) &&
         (weight.size(Layout::Filter::height) > 0) &&
         (weight.size(Layout::Filter::width) > 0) &&
         ((weight.device().is_cpu()) ||
          (c10::DeviceType::Vulkan == weight.device().type())) &&
         (kFloat == weight.scalar_type()) &&
         // Bias
         ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                       ((bias->device().is_cpu()) ||
                                        (c10::DeviceType::Vulkan == bias->device().type())) &&
                                       (kFloat == bias->scalar_type()) &&
                                       (transposed ? (weight.size(Layout::TransposedFilter::output) ==
                                                          bias->size(Layout::Filter::output))
                                                   : (weight.size(Layout::Filter::output) ==
                                                          bias->size(Layout::Filter::output))))
                                    : true) &&
         // Stride
         (stride[Layout::Parameter::height] > 0) &&
         (stride[Layout::Parameter::width] > 0) &&
         // Padding
         (padding[Layout::Parameter::height] >= 0) &&
         (padding[Layout::Parameter::width] >= 0) &&
         // Dilation
         (transposed ? (dilation[Layout::Parameter::height] == 1) &&
                       (dilation[Layout::Parameter::width] == 1)
                     : (dilation[Layout::Parameter::height] > 0) &&
                       (dilation[Layout::Parameter::width] > 0)) &&
         // Groups
         (groups > 0) &&
         // Input
         (weight.size(Layout::Filter::input) > 0) &&
         // Output
         (weight.size(Layout::Filter::output) > 0) &&
         // Output - Groups
         ((weight.size(Layout::Filter::output) % groups) == 0) &&
         // Output Min / Max
         (!output_min || output_min->isFloatingPoint()) &&
         (!output_max || output_max->isFloatingPoint()) &&
         true;
}

bool usable(const Tensor& input) {
         // Input
  return (4 == input.ndimension()) &&
         (c10::DeviceType::Vulkan == input.device().type()) &&
         (kFloat == input.scalar_type()) &&
         (input.size(Layout::Activation4D::batch) >= 0) &&
         (input.size(Layout::Activation4D::channels) > 0) &&
         (input.size(Layout::Activation4D::height) > 0) &&
         (input.size(Layout::Activation4D::width) > 0) &&
         !input.requires_grad() &&
         true;
}

static inline std::vector<int64_t> get_conv_transpose_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding,
    IntArrayRef stride, IntArrayRef dilation = IntArrayRef()) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_input_channels_dim];
  for (const auto d : c10::irange(2, dim)) {
    output_size[d] = stride[d - 2] * (input_size[d] - 1) + weight_size[d] - 2 * padding[d - 2] + output_padding[d - 2];
  }
  return output_size;
}

} // namespace

TransposeConv2dOpContext::TransposeConv2dOpContext(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef output_padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max)
  : packed_{
      pack_weights(weight),
      pack_biases(bias, weight),
      pack_filter(weight, expand_param_if_needed(dilation, "dilation", 2)),
      pack_params(expand_param_if_needed(stride, "stride", 2)),
      pack_params(expand_param_if_needed(padding, "padding", 2)),
      pack_params(expand_param_if_needed(output_padding, "output_padding", 2)),
      pack_params(expand_param_if_needed(dilation, "dilation", 2)),
      safe_downcast<int32_t>(groups),
      output_min ? output_min->template to<float>() : -std::numeric_limits<float>::infinity(),
      output_max ? output_max->template to<float>() : +std::numeric_limits<float>::infinity(),
    },
    unpacked_{
      weight,
      bias,
      weight.sizes().vec(),
      stride.vec(),
      padding.vec(),
      output_padding.vec(),
      dilation.vec(),
      groups,
      output_min,
      output_max,
    } {
}

TransposeConv2dOpContext TransposeConv2dOpContext::create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef output_padding_arg,
    const IntArrayRef dilation_arg,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  const auto stride = expand_param_if_needed(stride_arg, "stride", 2);
  const auto padding = expand_param_if_needed(padding_arg, "padding", 2);
  const auto dilation = expand_param_if_needed(dilation_arg, "dilation", 2);
  const auto output_padding = expand_param_if_needed(output_padding_arg, "output_padding", 2);

  TORCH_CHECK(
      available(
          weight,
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          true,
          groups,
          output_min,
          output_max),
      "Vulkan::convolution not available! "
      "Reason: The provided (weight, bias, stride, padding, dilation, groups, "
      "transposed, output_padding, output_min, output_max) parameters are either "
      "invalid individually or their combination is not supported by Vulkan impl.");

  // Pass in the originals
  return TransposeConv2dOpContext{
    weight,
    bias,
    stride_arg,
    padding_arg,
    output_padding_arg,
    dilation_arg,
    groups,
    output_min,
    output_max,
  };
}

void TransposeConv2dOpContext::conv2d_transpose_sliding_window(
    const api::Shader::Descriptor& shader,
    vTensor& v_output,
    const vTensor& v_input) const {
  bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && packed_.v_weight.has_image());
  TORCH_CHECK(valid, "Not Implemented!")

  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), "prepacked::conv2d_transpose_clamp_run (conv2d_transpose_sliding_window)");

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
      safe_downcast<int32_t>(packed_.filter[Layout::Filter::input]), /* this is aligned up */
      {
        safe_downcast<int32_t>(packed_.filter[Layout::Filter::width]),
        safe_downcast<int32_t>(packed_.filter[Layout::Filter::height]),
        safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::width]),
        safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::height]),
      },
      {
        safe_downcast<int32_t>(unpacked_.filter[Layout::Filter::width]),
        safe_downcast<int32_t>(unpacked_.filter[Layout::Filter::height]),
      },
      {
        safe_downcast<int32_t>(packed_.stride[Layout::Parameter::width]),
        safe_downcast<int32_t>(packed_.stride[Layout::Parameter::height]),
      },
      {
        safe_downcast<int32_t>(packed_.padding[Layout::Parameter::width]),
        safe_downcast<int32_t>(packed_.padding[Layout::Parameter::height]),
      },
      {
        safe_downcast<int32_t>(packed_.dilation[Layout::Parameter::width]),
        safe_downcast<int32_t>(packed_.dilation[Layout::Parameter::height]),
      },
      {
        packed_.output_min,
        packed_.output_max,
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
        shader,
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
        packed_.v_weight.image(
            command_buffer,
            vTensor::Stage::Compute),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        packed_.v_bias.image(
            command_buffer,
            vTensor::Stage::Compute),
        // Object lifetime is managed by the resource pool.
        // It is OK not to keep track of the handle.
        context->resource().pool.uniform(block).object);
  }
  command_pool.submit(context->gpu().queue, command_buffer);
}

Tensor TransposeConv2dOpContext::run(const Tensor& input_arg) const {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  TORCH_CHECK(
      usable(input),
      "Vulkan Convolution not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by Vulkan impl.");

  vTensor v_output{
    context,
    get_conv_transpose_output_size(
        v_input.sizes(),
        unpacked_.filter,
        packed_.padding,
        packed_.output_padding,
        packed_.stride,
        packed_.dilation),
    input.options(),
  };

  conv2d_transpose_sliding_window(VK_KERNEL(conv_transpose2d), v_output, v_input);

  return convert(v_output);
}

TransposeConv2dOpContext::State TransposeConv2dOpContext::unpack() const {
  return TransposeConv2dOpContext::State{
    unpacked_.weight,
    unpacked_.bias,
    unpacked_.stride,
    unpacked_.padding,
    unpacked_.output_padding,
    unpacked_.dilation,
    unpacked_.groups,
    unpacked_.output_min,
    unpacked_.output_max,
  };
}

c10::intrusive_ptr<TransposeConv2dOpContext> conv2d_transpose_clamp_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return c10::make_intrusive<TransposeConv2dOpContext>(
      TransposeConv2dOpContext::create(
          std::move(weight),
          std::move(bias),
          std::move(stride),
          std::move(padding),
          std::move(output_padding),
          std::move(dilation),
          groups,
          output_min,
          output_max));
}

Tensor conv2d_transpose_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<TransposeConv2dOpContext>& context) {
  return context->run(input);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
