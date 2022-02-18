#include <ATen/native/vulkan/ops/GatedConv2dModule.h>
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

std::array<int64_t, 4> pack_filter(
    const Tensor& weight,
    const IntArrayRef dilation) {
  const IntArrayRef filter = weight.sizes();

  const auto effective = [](const int64_t k, const int64_t d) {
    return k + (k - 1) * (d - 1);
  };

  return {
    align_up(filter[Layout::Filter::output], INT64_C(4)),
    align_up(filter[Layout::Filter::input], INT64_C(4)),
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
          /*
          memcpy(
              dst_weight_c_ptr + (dst_oh * src_kh_sz + dst_h) * dst_kw_sz + src_ic + dst_w_offset,
              src_weight_1_oc_ptr + src_ic * src_kernel_sz + src_ih * src_kw_sz + src_iw,
              sizeof(float));
          */
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

GatedConv2dModuleOpContext::GatedConv2dModuleOpContext(
      const Tensor& weight_a,
      const c10::optional<Tensor>& bias_a,
      const Tensor& weight_b,
      const c10::optional<Tensor>& bias_b,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef output_padding,
      IntArrayRef dilation,
      const int64_t groups,
      const bool transposed)
  : packed_{
      transposed ? pack_combined_weights_reverse(weight_a, weight_b) : pack_combined_weights(weight_a, weight_b),
      pack_combined_biases(bias_a, bias_b),
      pack_filter(weight_a, expand_param_if_needed(dilation, "dilation", 2)),
      pack_params(expand_param_if_needed(stride, "stride", 2)),
      pack_params(expand_param_if_needed(padding, "padding", 2)),
      pack_params(expand_param_if_needed(output_padding, "output_padding", 2)),
      pack_params(expand_param_if_needed(dilation, "dilation", 2)),
    },
    unpacked_{
      weight_a,
      bias_a,
      weight_b,
      bias_b,
      weight_a.sizes().vec(),
      stride.vec(),
      padding.vec(),
      output_padding.vec(),
      dilation.vec(),
      groups,
      transposed
    } {
}

GatedConv2dModuleOpContext GatedConv2dModuleOpContext::create(
      const Tensor& weight_a,
      const c10::optional<Tensor>& bias_a,
      const Tensor& weight_b,
      const c10::optional<Tensor>& bias_b,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef output_padding,
      IntArrayRef dilation,
      const int64_t groups,
      const bool transposed) {
  // Pass in the originals
  return GatedConv2dModuleOpContext{
    weight_a,
    bias_a,
    weight_b,
    bias_b,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    transposed
  };
}

Tensor GatedConv2dModuleOpContext::run(const Tensor& padding_arg, const Tensor& prev_out_arg) const {
  api::Context* const context = api::context();

  const Tensor padding = padding_arg.is_vulkan() ? padding_arg : padding_arg.vulkan();
  const vTensor& v_padding = convert(padding);
  const Tensor prev_out = prev_out_arg.is_vulkan() ? prev_out_arg : prev_out_arg.vulkan();
  const vTensor& v_prev_out = convert(prev_out);

  std::vector<int64_t> padding_size = padding.sizes().vec();
  std::vector<int64_t> prev_out_size = prev_out.sizes().vec();
  TORCH_CHECK(padding_size.size() == 4, "GatedConv2dModule: first input tensor must have exactly 4 dims")
  TORCH_CHECK(prev_out_size.size() == 4, "GatedConv2dModule: second input tensor must have exactly 4 dims")
  //TORCH_CHECK(padding_size[2] == 1, "GatedConv2dModule: first input tensor must have height of exactly 1")
  //TORCH_CHECK(prev_out_size[2] == 1, "GatedConv2dModule: second input tensor must have height of exactly 1")
  std::vector<int64_t> conv_input_size(padding_size);
  conv_input_size[2] = 2;
  std::vector<int64_t> output_size = conv_output_size(
    conv_input_size,
    unpacked_.filter,
    /*padding=*/{0,0},
    packed_.stride,
    /*dilation=*/{1,1}
  );

  vTensor v_output{
    context,
    output_size,
    padding.options(),
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
        VK_KERNEL(gated_conv2d_module),
        global_size,
        adaptive_work_group_size(global_size),
        v_output.image(
            command_buffer,
            vTensor::Stage::Compute,
            vTensor::Access::Write),
        v_padding.image(
            command_buffer,
            vTensor::Stage::Compute),
        v_prev_out.image(
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
}

Tensor GatedConv2dModuleOpContext::run_transpose(
    const Tensor& padding_arg, const Tensor& prev_enc_out_arg,
    const Tensor& prev_out_arg, const Tensor& encoder_out_arg) const {
  api::Context* const context = api::context();

  const Tensor padding = padding_arg.is_vulkan() ? padding_arg : padding_arg.vulkan();
  const vTensor& v_padding = convert(padding);
  const Tensor prev_enc_out = prev_enc_out_arg.is_vulkan() ? prev_enc_out_arg : prev_enc_out_arg.vulkan();
  const vTensor& v_prev_enc_out = convert(prev_enc_out);
  const Tensor prev_out = prev_out_arg.is_vulkan() ? prev_out_arg : prev_out_arg.vulkan();
  const vTensor& v_prev_out = convert(prev_out);
  const Tensor encoder_out = encoder_out_arg.is_vulkan() ? encoder_out_arg : encoder_out_arg.vulkan();
  const vTensor& v_encoder_out = convert(encoder_out);

  std::vector<int64_t> padding_size = padding.sizes().vec();
  std::vector<int64_t> prev_enc_out_size = prev_enc_out.sizes().vec();
  std::vector<int64_t> prev_out_size = prev_out.sizes().vec();
  std::vector<int64_t> encoder_out_size = encoder_out.sizes().vec();
  TORCH_CHECK(padding_size.size() == 4, "GatedConv2dModule: padding tensor must have exactly 4 dims")
  TORCH_CHECK(prev_enc_out_size.size() == 4, "GatedConv2dModule: previous encoder output tensor must have exactly 4 dims")
  TORCH_CHECK(prev_out_size.size() == 4, "GatedConv2dModule: previous output tensor must have exactly 4 dims")
  TORCH_CHECK(encoder_out_size.size() == 4, "GatedConv2dModule: encoder output tensor must have exactly 4 dims")
  TORCH_CHECK(padding_size[2] == 1, "GatedConv2dModule: padding tensor must have height of exactly 1")
  TORCH_CHECK(prev_enc_out_size[2] == 1, "GatedConv2dModule: previous encoder output tensor must have height of exactly 1")
  TORCH_CHECK(prev_out_size[2] == 1, "GatedConv2dModule: previous output tensor must have height of exactly 1")
  TORCH_CHECK(encoder_out_size[2] == 1, "GatedConv2dModule: encoder output tensor must have height of exactly 1")
  //TODO: check number of channels match
  std::vector<int64_t> conv_input_size(padding_size);
  conv_input_size[2] = 2;
  std::vector<int64_t> output_size = get_conv_transpose_output_size(
    conv_input_size,
    unpacked_.filter,
    packed_.padding,
    packed_.output_padding,
    packed_.stride,
    /*dilation=*/{1,1}
  );

  vTensor v_output{
    context,
    output_size,
    prev_out.options(),
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
      safe_downcast<int32_t>(packed_.filter[Layout::TransposedFilter::input]),
      {
        safe_downcast<int32_t>(packed_.filter[Layout::Filter::width]),
        safe_downcast<int32_t>(packed_.filter[Layout::Filter::height]),
        safe_downcast<int32_t>(conv_input_size[Layout::Activation4D::width]),
        safe_downcast<int32_t>(conv_input_size[Layout::Activation4D::height]),
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
        VK_KERNEL(gated_conv_transpose2d_module),
        global_size,
        adaptive_work_group_size(global_size),
        v_output.image(
            command_buffer,
            vTensor::Stage::Compute,
            vTensor::Access::Write),
        v_padding.image(
            command_buffer,
            vTensor::Stage::Compute),
        v_prev_enc_out.image(
            command_buffer,
            vTensor::Stage::Compute),
        v_prev_out.image(
            command_buffer,
            vTensor::Stage::Compute),
        v_encoder_out.image(
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

  //return convert(v_output);
  return convert(v_output);
}

GatedConv2dModuleOpContext::State GatedConv2dModuleOpContext::unpack() const {
  return GatedConv2dModuleOpContext::State{
    unpacked_.weight_a,
    unpacked_.bias_a,
    unpacked_.weight_b,
    unpacked_.bias_b,
    unpacked_.stride,
    unpacked_.padding,
    unpacked_.output_padding,
    unpacked_.dilation,
    unpacked_.groups,
    unpacked_.transposed
  };
}

c10::intrusive_ptr<GatedConv2dModuleOpContext> gated_conv2d_module_prepack(
    Tensor&& weight_a,
    c10::optional<Tensor>&& bias_a,
    Tensor&& weight_b,
    c10::optional<Tensor>&& bias_b,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const bool transposed) {
  return c10::make_intrusive<GatedConv2dModuleOpContext>(
      GatedConv2dModuleOpContext::create(
          std::move(weight_a),
          std::move(bias_a),
          std::move(weight_b),
          std::move(bias_b),
          std::move(stride),
          std::move(padding),
          std::move(output_padding),
          std::move(dilation),
          groups,
          transposed));
}

Tensor gated_conv2d_module_run(
    const Tensor& padding,
    const Tensor& prev_out,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context) {
  return context->run(padding, prev_out);
}

Tensor gated_conv_transpose2d_module_run(
    const Tensor& padding,
    const Tensor& prev_enc_out,
    const Tensor& prev_out,
    const Tensor& encoder_out,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context) {
  return context->run_transpose(padding, prev_enc_out, prev_out, encoder_out);
}

Tensor gated_conv2d_module_run_cpu(
    const Tensor& padding,
    const Tensor& prev_out,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context) {
  const auto input = at::cat({padding, prev_out}, 2);
  const auto output_a_cpu = at::conv2d(
      input,
      context->unpacked_.weight_a,
      context->unpacked_.bias_a,
      context->unpacked_.stride,
      context->unpacked_.padding,
      context->unpacked_.dilation,
      context->unpacked_.groups);
  const auto output_b_cpu = at::conv2d(
      input,
      context->unpacked_.weight_b,
      context->unpacked_.bias_b,
      context->unpacked_.stride,
      context->unpacked_.padding,
      context->unpacked_.dilation,
      context->unpacked_.groups);

  return output_a_cpu * at::sigmoid(output_b_cpu);
}


Tensor gated_conv_transpose2d_module_run_cpu(
    const Tensor& padding,
    const Tensor& prev_enc_out,
    const Tensor& prev_out,
    const Tensor& encoder_out,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context) {
  const auto top_row_cpu = at::cat({padding, prev_enc_out}, 1);
  const auto bottom_row_cpu = at::cat({prev_out, encoder_out}, 1);
  const auto input_cpu = at::cat({top_row_cpu, bottom_row_cpu}, 2);
  const auto output_a_cpu = at::conv_transpose2d(
      input_cpu,
      context->unpacked_.weight_a,
      context->unpacked_.bias_a,
      context->unpacked_.stride,
      context->unpacked_.padding,
      context->unpacked_.output_padding,
      context->unpacked_.groups,
      context->unpacked_.dilation);
  const auto output_b_cpu = at::conv_transpose2d(
      input_cpu,
      context->unpacked_.weight_b,
      context->unpacked_.bias_b,
      context->unpacked_.stride,
      context->unpacked_.padding,
      context->unpacked_.output_padding,
      context->unpacked_.groups,
      context->unpacked_.dilation);
  return output_a_cpu * at::sigmoid(output_b_cpu);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
