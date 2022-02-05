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

vTensor pack_combined_weights(const Tensor& weight_1, const Tensor& weight_2) {
  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();

  /* Source */
  const IntArrayRef src_filter = weight_1.sizes();
  const float* const src_weight_1_ptr = weight_1.data_ptr<float>();
  const float* const src_weight_2_ptr = weight_2.data_ptr<float>();

  const int64_t src_kw_sz = src_filter[Layout::Filter::width];
  const int64_t src_kh_sz = src_filter[Layout::Filter::height];
  const int64_t src_kernel_sz = src_kw_sz * src_kh_sz;
  const int64_t src_block_sz = src_kernel_sz * src_filter[Layout::Filter::input];

  const int64_t num_stacks = div_up(src_filter[Layout::Filter::output], INT64_C(4));
  const int64_t stack_depth = api::utils::align_up(src_filter[Layout::Filter::input], INT64_C(4));

  /* Destination */
  const int64_t dst_kw_sz = src_kw_sz * stack_depth;
  const int64_t dst_kh_sz = 2 * src_kh_sz * num_stacks;
  const int64_t dst_kernel_sz = dst_kw_sz * dst_kh_sz;

  vTensor v_weight{
      context,
      {
          4,
          dst_kh_sz,
          dst_kw_sz,
      },
      weight_1.options(),
  };

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
  Future::Payload v_weight_payload = v_weight_future.wait();

  float* const dst_weight_ptr = v_weight_payload.get();
  memset(dst_weight_ptr, 0, v_weight.nbytes());

  for (int64_t src_oc = 0; src_oc < src_filter[Layout::Filter::output]; ++src_oc) {
    /* Source */
    const float* const src_weight_1_oc_ptr = src_weight_1_ptr + src_oc * src_block_sz;
    const float* const src_weight_2_oc_ptr = src_weight_2_ptr + src_oc * src_block_sz;

    /* Destination */
    const int64_t dst_oh = src_oc / 4;
    const int64_t dst_c = src_oc % 4;

    float* const dst_weight_c_ptr = dst_weight_ptr + dst_c * dst_kernel_sz;

    for (int64_t src_ic = 0; src_ic < src_filter[Layout::Filter::input]; ++src_ic) {
      const int64_t dst_ic4 = src_ic / 4;

      for (const auto src_ih : c10::irange(src_kh_sz)) {
        for (const auto src_iw : c10::irange(src_kw_sz)) {
          memcpy(
              dst_weight_c_ptr + (2*(dst_oh * src_kh_sz + src_ih)) * dst_kw_sz +
                dst_ic4 * src_kw_sz * 4 + src_iw * 4 + src_ic % 4,
              src_weight_1_oc_ptr + src_ic * src_kernel_sz + src_ih * src_kw_sz + src_iw,
              sizeof(float));
          memcpy(
              dst_weight_c_ptr + (2*(dst_oh * src_kh_sz + src_ih)+1) * dst_kw_sz +
                dst_ic4 * src_kw_sz * 4 + src_iw * 4 + src_ic % 4,
              src_weight_2_oc_ptr + src_ic * src_kernel_sz + src_ih * src_kw_sz + src_iw,
              sizeof(float));
        }
      }
    }
  }

  return v_weight;
}

vTensor pack_combined_weights_reverse(const Tensor& weight_arg_1, const Tensor& weight_arg_2) {
  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();

  /* Source */
  const Tensor weight_1 = at::permute(weight_arg_1, {1, 0, 2, 3}).contiguous();
  const IntArrayRef src_filter = weight_1.sizes();
  const float* const src_weight_1_ptr = weight_1.data_ptr<float>();
  const Tensor weight_2 = at::permute(weight_arg_2, {1, 0, 2, 3}).contiguous();
  const float* const src_weight_2_ptr = weight_2.data_ptr<float>();

  const int64_t src_kw_sz = src_filter[Layout::Filter::width];
  const int64_t src_kh_sz = src_filter[Layout::Filter::height];
  const int64_t src_kernel_sz = src_kw_sz * src_kh_sz;
  const int64_t src_block_sz = src_kernel_sz * src_filter[Layout::Filter::input];

  const int64_t num_stacks = div_up(src_filter[Layout::Filter::output], INT64_C(4));
  const int64_t stack_depth = api::utils::align_up(src_filter[Layout::Filter::input], INT64_C(4));

  /* Destination */
  const int64_t dst_kw_sz = src_kw_sz * stack_depth;
  const int64_t dst_kh_sz = 2 * src_kh_sz * num_stacks;
  const int64_t dst_kernel_sz = dst_kw_sz * dst_kh_sz;

  vTensor v_weight{
      context,
      {
          4,
          dst_kh_sz,
          dst_kw_sz,
      },
      weight_1.options(),
  };

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
  Future::Payload v_weight_payload = v_weight_future.wait();

  float* const dst_weight_ptr = v_weight_payload.get();
  memset(dst_weight_ptr, 0, v_weight.nbytes());

  for (int64_t src_oc = 0; src_oc < src_filter[Layout::Filter::output]; ++src_oc) {
    /* Source */
    const float* const src_weight_1_oc_ptr = src_weight_1_ptr + src_oc * src_block_sz;
    const float* const src_weight_2_oc_ptr = src_weight_2_ptr + src_oc * src_block_sz;

    /* Destination */
    const int64_t dst_oh = src_oc / 4;
    const int64_t dst_c = src_oc % 4;

    float* const dst_weight_c_ptr = dst_weight_ptr + dst_c * dst_kernel_sz;

    for (int64_t src_ic = 0; src_ic < src_filter[Layout::Filter::input]; ++src_ic) {
      for (int64_t src_ih = 0; src_ih < src_kh_sz; ++src_ih) {
        const int64_t dst_h = src_kh_sz - 1 - src_ih;
        for (int64_t src_iw = 0; src_iw < src_kw_sz; ++src_iw) {
          const int64_t dst_w = src_kw_sz - 1 - src_iw;
          const int64_t dst_w_offset = dst_w * stack_depth;
          memcpy(
              dst_weight_c_ptr + (2*(dst_oh * src_kh_sz + dst_h)) * dst_kw_sz + src_ic + dst_w_offset,
              src_weight_1_oc_ptr + src_ic * src_kernel_sz + src_ih * src_kw_sz + src_iw,
              sizeof(float));
          memcpy(
              dst_weight_c_ptr + (2*(dst_oh * src_kh_sz + dst_h)+1) * dst_kw_sz + src_ic + dst_w_offset,
              src_weight_2_oc_ptr + src_ic * src_kernel_sz + src_ih * src_kw_sz + src_iw,
              sizeof(float));
        }
      }
    }
  }

  return v_weight;
}

vTensor pack_combined_biases(const c10::optional<Tensor>& bias_1,
                             const c10::optional<Tensor>& bias_2) {
  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();

  int64_t src_w = 1;
  if (bias_1) {
    for (const auto i : bias_1->sizes().vec()) {
      if (i > 1) src_w = i;
    }
  }
  const int64_t packed_w = div_up(src_w, INT64_C(4));
  vTensor v_bias{
    context,
    {
      4,
      2,
      packed_w,
    },
    bias_1->options(),
  };

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  Future v_bias_future = v_bias.host<float, vTensor::Access::Write>(command_buffer);
  Future::Payload v_bias_payload = v_bias_future.wait();
  float* const dst_bias_ptr = v_bias_payload.get();

  memset(dst_bias_ptr, 0, v_bias.nbytes());
  if (bias_1) {
    const float* const src_bias_ptr = bias_1->contiguous().data_ptr<float>();

    for (const auto i : c10::irange(src_w)) {
      const int64_t c = i % 4;
      const int64_t x = i / 4;
      dst_bias_ptr[c * 2 * packed_w + x] = src_bias_ptr[i];
    }
  }
  if (bias_2) {
    const float* const src_bias_ptr = bias_2->contiguous().data_ptr<float>();

    for (const auto i : c10::irange(src_w)) {
      const int64_t c = i % 4;
      const int64_t x = i / 4;
      dst_bias_ptr[c * 2 * packed_w + packed_w + x] = src_bias_ptr[i];
    }
  }

  return v_bias;
}

static inline std::vector<int64_t> get_conv_transpose_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding,
    IntArrayRef stride, IntArrayRef dilation = IntArrayRef()) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_input_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    output_size[d] = stride[d - 2] * (input_size[d] - 1) + weight_size[d] - 2 * padding[d - 2] + output_padding[d - 2];
  }
  return output_size;
}

McLarenEncoderBlockOpContext::McLarenEncoderBlockOpContext(
      const Tensor& weight_1,
      const c10::optional<Tensor>& bias_1,
      IntArrayRef stride_1,
      IntArrayRef padding_1,
      IntArrayRef output_padding_1,
      IntArrayRef dilation_1,
      const int64_t groups_1,
      const Tensor& weight_2,
      const c10::optional<Tensor>& bias_2,
      IntArrayRef stride_2,
      IntArrayRef padding_2,
      IntArrayRef output_padding_2,
      IntArrayRef dilation_2,
      const int64_t groups_2,
      const bool transposed)
  : packed_{
      transposed ? pack_combined_weights_reverse(weight_1, weight_2) : pack_combined_weights(weight_1, weight_2),
      pack_combined_biases(bias_1, bias_2),
      pack_conv2d_filter(weight_1, expand_param_if_needed(dilation_1, "dilation", 2)),
      pack_conv2d_params(expand_param_if_needed(stride_1, "stride", 2)),
      pack_conv2d_params(expand_param_if_needed(padding_1, "padding", 2)),
      pack_conv2d_params(expand_param_if_needed(output_padding_1, "output_padding", 2)),
      pack_conv2d_params(expand_param_if_needed(dilation_1, "dilation", 2)),
    },
    unpacked_{
      weight_1,
      bias_1,
      weight_1.sizes().vec(),
      stride_1.vec(),
      padding_1.vec(),
      output_padding_1.vec(),
      dilation_1.vec(),
      groups_1,
      weight_2,
      bias_2,
      weight_2.sizes().vec(),
      stride_2.vec(),
      padding_2.vec(),
      output_padding_2.vec(),
      dilation_2.vec(),
      groups_2,
      transposed,
    } {
}

McLarenEncoderBlockOpContext McLarenEncoderBlockOpContext::create(
      const Tensor& weight_1,
      const c10::optional<Tensor>& bias_1,
      IntArrayRef stride_1,
      IntArrayRef padding_1,
      IntArrayRef output_padding_1,
      IntArrayRef dilation_1,
      const int64_t groups_1,
      const Tensor& weight_2,
      const c10::optional<Tensor>& bias_2,
      IntArrayRef stride_2,
      IntArrayRef padding_2,
      IntArrayRef output_padding_2,
      IntArrayRef dilation_2,
      const int64_t groups_2,
      const bool transposed) {
  // Pass in the originals
  return McLarenEncoderBlockOpContext{
    weight_1,
    bias_1,
    stride_1,
    padding_1,
    output_padding_1,
    dilation_1,
    groups_1,
    weight_2,
    bias_2,
    stride_2,
    padding_2,
    output_padding_2,
    dilation_2,
    groups_2,
    transposed
  };
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
  TORCH_CHECK(input_2_size.size() == 4, "McLarenEncoderBlock: second input tensor must have exactly 4 dims")
  //TORCH_CHECK(input_1_size[2] == 1, "McLarenEncoderBlock: first input tensor must have height of exactly 1")
  //TORCH_CHECK(input_2_size[2] == 1, "McLarenEncoderBlock: second input tensor must have height of exactly 1")
  std::vector<int64_t> conv_input_size(input_1_size);
  conv_input_size[2] = 2;
  std::vector<int64_t> output_size = !unpacked_.transposed ? conv_output_size(
    conv_input_size,
    unpacked_.filter_1,
    /*padding=*/{0,0},
    packed_.stride,
    /*dilation=*/{1,1}
  ) : get_conv_transpose_output_size(
    conv_input_size,
    unpacked_.filter_1,
    packed_.padding,
    packed_.output_padding,
    packed_.stride,
    /*dilation=*/{1,1}
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
    } block {
      v_output.extents(),
      safe_downcast<int32_t>(packed_.filter[Layout::Filter::input]),
      {
        safe_downcast<int32_t>(packed_.filter[Layout::Filter::width]),
        safe_downcast<int32_t>(packed_.filter[Layout::Filter::height]),
        safe_downcast<int32_t>(conv_input_size[Layout::Activation4D::width]),
        safe_downcast<int32_t>(conv_input_size[Layout::Activation4D::height]),
      },
      {
        safe_downcast<int32_t>(unpacked_.filter_1[Layout::Filter::width]),
        safe_downcast<int32_t>(unpacked_.filter_1[Layout::Filter::height]),
      },
      {
        safe_downcast<int32_t>(packed_.stride[Layout::Parameter::width]),
        safe_downcast<int32_t>(packed_.stride[Layout::Parameter::height]),
      },
      {
        safe_downcast<int32_t>(packed_.padding[Layout::Parameter::width]),
        safe_downcast<int32_t>(packed_.padding[Layout::Parameter::height]),
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
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        unpacked_.transposed ? VK_KERNEL(mclaren_decoder_block) : VK_KERNEL(mclaren_encoder_block),
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
        packed_.v_weight.image(
            command_buffer,
            vTensor::Stage::Compute),
        packed_.v_bias.image(
            command_buffer,
            vTensor::Stage::Compute),
        context->resource().pool.uniform(block).object);
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return convert(v_output);
  //return convert(packed_.v_weight);
}

McLarenEncoderBlockOpContext::State McLarenEncoderBlockOpContext::unpack() const {
  return McLarenEncoderBlockOpContext::State{
    unpacked_.weight_1,
    unpacked_.bias_1,
    unpacked_.stride_1,
    unpacked_.padding_1,
    unpacked_.output_padding_1,
    unpacked_.dilation_1,
    unpacked_.groups_1,
    unpacked_.weight_2,
    unpacked_.bias_2,
    unpacked_.stride_2,
    unpacked_.padding_2,
    unpacked_.output_padding_2,
    unpacked_.dilation_2,
    unpacked_.groups_2,
    unpacked_.transposed
  };
}

c10::intrusive_ptr<McLarenEncoderBlockOpContext> mclaren_encoder_block_prepack(
    Tensor&& weight_1,
    c10::optional<Tensor>&& bias_1,
    std::vector<int64_t>&& stride_1,
    std::vector<int64_t>&& padding_1,
    std::vector<int64_t>&& output_padding_1,
    std::vector<int64_t>&& dilation_1,
    const int64_t groups_1,
    Tensor&& weight_2,
    c10::optional<Tensor>&& bias_2,
    std::vector<int64_t>&& stride_2,
    std::vector<int64_t>&& padding_2,
    std::vector<int64_t>&& output_padding_2,
    std::vector<int64_t>&& dilation_2,
    const int64_t groups_2,
    const bool transposed) {
  return c10::make_intrusive<McLarenEncoderBlockOpContext>(
      McLarenEncoderBlockOpContext::create(
          std::move(weight_1),
          std::move(bias_1),
          std::move(stride_1),
          std::move(padding_1),
          std::move(output_padding_1),
          std::move(dilation_1),
          groups_1,
          std::move(weight_2),
          std::move(bias_2),
          std::move(stride_2),
          std::move(padding_2),
          std::move(output_padding_2),
          std::move(dilation_2),
          groups_2,
          transposed));
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
