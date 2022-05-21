#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

struct Experimentation final {
  static constexpr bool kUseWinogradConvs = false;
};

inline bool is_depthwise(
    const IntArrayRef filter,
    const int64_t groups) {
  return (filter[Layout::Filter::output] == groups) &&
         // Only K == 1 supported.
         (filter[Layout::Filter::input] == 1);
}

inline bool is_pointwise(const IntArrayRef filter) {
  return (1 == filter[Layout::Filter::height]) &&
         (1 == filter[Layout::Filter::width]);
}


bool all_lessthan(const IntArrayRef arr, const int t) {
  bool retval = true;
  for (const auto i : c10::irange(arr.size())) {
    retval = retval && (arr[i] < t);
  }
  return retval;
}

inline bool is_winograd_n_3(
    const IntArrayRef filter,
    const IntArrayRef stride,
    const IntArrayRef dilation) {
  return (3 == filter[Layout::Filter::height]) &&
         (3 == filter[Layout::Filter::width]) &&
         all_lessthan(stride, 2) &&
         all_lessthan(dilation, 2);
}

Conv2dMethod determine_method(
    const IntArrayRef filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups) {
  if (is_depthwise(filter, groups))
    return Conv2dDepthwise;
  if (is_pointwise(filter))
    return Conv2dPointwise;
  if (Experimentation::kUseWinogradConvs && is_winograd_n_3(filter, stride, dilation))
    return Conv2dWinograd_2_3;
  return Conv2dSlidingWindow;
}

vTensor pack_weights_dw(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    const Tensor& weight) {
  /* Source */
  const IntArrayRef src_filter = weight.sizes();
  const float* const src_weight_ptr = weight.data_ptr<float>();

  const int64_t src_kw_sz = src_filter[Layout::Filter::width];
  const int64_t src_kh_sz = src_filter[Layout::Filter::height];
  const int64_t src_kernel_sz = src_kw_sz * src_kh_sz;
  const int64_t src_block_sz = src_kernel_sz * src_filter[Layout::Filter::input];
  const int64_t num_stacks = div_up(src_filter[Layout::Filter::output], INT64_C(4));

  /* Destination */
  const int64_t dst_kw_sz = src_kernel_sz;
  const int64_t dst_kh_sz = num_stacks;
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

    float* const dst_weight_c_ptr = dst_weight_ptr +
                                    dst_c * dst_kernel_sz +
                                    dst_oh * dst_kw_sz;

    for (const auto src_ih : c10::irange(src_filter[Layout::Filter::height])) {
      memcpy(
          dst_weight_c_ptr + src_ih * src_kw_sz,
          src_weight_oc_ptr + src_ih * src_kw_sz,
          sizeof(float) * src_kw_sz);
    }
  }

  return v_weight;
}

vTensor pack_weights_2d(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    const Tensor& weight) {
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
      const int64_t dst_ic4 = src_ic / 4;

      for (const auto src_ih : c10::irange(src_kh_sz)) {
        for (const auto src_iw : c10::irange(src_kw_sz)) {
          memcpy(
              dst_weight_c_ptr + (dst_oh * src_kh_sz + src_ih) * dst_kw_sz +
                dst_ic4 * src_kw_sz * 4 + src_iw * 4 + src_ic % 4,
              src_weight_oc_ptr + src_ic * src_kernel_sz + src_ih * src_kw_sz + src_iw,
              sizeof(float));
        }
      }
    }
  }

  return v_weight;
}

vTensor pack_weights_2d_winograd_2_3(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    const Tensor& weight) {
  /* Source */
  const IntArrayRef src_filter = weight.sizes();

  TORCH_CHECK(
      src_filter[Layout::Filter::width] == 3 && src_filter[Layout::Filter::height] == 3,
      "Kernel size must be 3x3 for Winograd(2x2, 3x3)!");
  const int64_t src_ic_sz = src_filter[Layout::Filter::input];
  const int64_t src_oc_sz = src_filter[Layout::Filter::output];

  /* Destination */
  const int64_t dst_ow_sz = div_up(src_ic_sz, INT64_C(4));
  const int64_t dst_oh_sz = div_up(src_oc_sz, INT64_C(4));
  const int64_t dst_kw_sz = 16*dst_ow_sz;
  const int64_t dst_kh_sz = 4*dst_oh_sz;
  const int64_t dst_block_sz = dst_kw_sz * dst_kh_sz;

  vTensor v_weight{
      context,
      {
        4,
        4*dst_oh_sz,
        16*dst_ow_sz,
      },
      weight.options(),
  };

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
  Future::Payload v_weight_payload = v_weight_future.wait();

  float* const dst_weight_ptr = v_weight_payload.get();
  memset(dst_weight_ptr, 0, v_weight.nbytes());

  for (const auto src_oc : c10::irange(src_oc_sz)) {
    const int64_t dst_oh = src_oc / 4;
    const int64_t dst_iw = src_oc % 4;

    for (const auto src_ic : c10::irange(src_ic_sz)) {
      const int64_t dst_ow = src_ic / 4;
      const int64_t dst_c = src_ic % 4;

      //const float* const src_k_ptr = src_weight_ptr + src_oc * src_block_sz + src_ic * 9;
      float* const dst_k = dst_weight_ptr + dst_c * dst_block_sz;

      const float s00 = weight[src_oc][src_ic][0][0].item<float>();
      const float s01 = weight[src_oc][src_ic][0][1].item<float>();
      const float s02 = weight[src_oc][src_ic][0][2].item<float>();
      const float s10 = weight[src_oc][src_ic][1][0].item<float>();
      const float s11 = weight[src_oc][src_ic][1][1].item<float>();
      const float s12 = weight[src_oc][src_ic][1][2].item<float>();
      const float s20 = weight[src_oc][src_ic][2][0].item<float>();
      const float s21 = weight[src_oc][src_ic][2][1].item<float>();
      const float s22 = weight[src_oc][src_ic][2][2].item<float>();

      const float m00 = s00;
      const float m01 = s01;
      const float m02 = s02;
      const float m10 = (s00 + s10 + s20)/2.f;
      const float m11 = (s01 + s11 + s21)/2.f;
      const float m12 = (s02 + s12 + s22)/2.f;
      const float m20 = (s00 - s10 + s20)/2.f;
      const float m21 = (s01 - s11 + s21)/2.f;
      const float m22 = (s02 - s12 + s22)/2.f;
      const float m30 = s20;
      const float m31 = s21;
      const float m32 = s22;

      dst_k[(4*dst_oh + 0)*dst_kw_sz +  0*dst_ow_sz + 4*dst_ow + dst_iw] = m00;
      dst_k[(4*dst_oh + 0)*dst_kw_sz +  4*dst_ow_sz + 4*dst_ow + dst_iw] = (m00 + m01 + m02)/2.f;
      dst_k[(4*dst_oh + 0)*dst_kw_sz +  8*dst_ow_sz + 4*dst_ow + dst_iw] = (m00 - m01 + m02)/2.f;
      dst_k[(4*dst_oh + 0)*dst_kw_sz + 12*dst_ow_sz + 4*dst_ow + dst_iw] = m02;
      dst_k[(4*dst_oh + 1)*dst_kw_sz +  0*dst_ow_sz + 4*dst_ow + dst_iw] = m10;
      dst_k[(4*dst_oh + 1)*dst_kw_sz +  4*dst_ow_sz + 4*dst_ow + dst_iw] = (m10 + m11 + m12)/2.f;
      dst_k[(4*dst_oh + 1)*dst_kw_sz +  8*dst_ow_sz + 4*dst_ow + dst_iw] = (m10 - m11 + m12)/2.f;
      dst_k[(4*dst_oh + 1)*dst_kw_sz + 12*dst_ow_sz + 4*dst_ow + dst_iw] = m12;
      dst_k[(4*dst_oh + 2)*dst_kw_sz +  0*dst_ow_sz + 4*dst_ow + dst_iw] = m20;
      dst_k[(4*dst_oh + 2)*dst_kw_sz +  4*dst_ow_sz + 4*dst_ow + dst_iw] = (m20 + m21 + m22)/2.f;
      dst_k[(4*dst_oh + 2)*dst_kw_sz +  8*dst_ow_sz + 4*dst_ow + dst_iw] = (m20 - m21 + m22)/2.f;
      dst_k[(4*dst_oh + 2)*dst_kw_sz + 12*dst_ow_sz + 4*dst_ow + dst_iw] = m22;
      dst_k[(4*dst_oh + 3)*dst_kw_sz +  0*dst_ow_sz + 4*dst_ow + dst_iw] = m30;
      dst_k[(4*dst_oh + 3)*dst_kw_sz +  4*dst_ow_sz + 4*dst_ow + dst_iw] = (m30 + m31 + m32)/2.f;
      dst_k[(4*dst_oh + 3)*dst_kw_sz +  8*dst_ow_sz + 4*dst_ow + dst_iw] = (m30 - m31 + m32)/2.f;
      dst_k[(4*dst_oh + 3)*dst_kw_sz + 12*dst_ow_sz + 4*dst_ow + dst_iw] = m32;
    }
  }

  return v_weight;
}

vTensor pack_weights(
    const Tensor& weight_arg,
    const Conv2dMethod conv_method) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();  // Don't collect the timestamp since the command buffer doesn't record anything

  const Tensor weight = weight_arg.contiguous();

  if (conv_method == Conv2dDepthwise) {
    return pack_weights_dw(
        context,
        command_buffer,
        weight);
  }

  if (conv_method == Conv2dWinograd_2_3) {
    return pack_weights_2d_winograd_2_3(
        context,
        command_buffer,
        weight);
  }

  return pack_weights_2d(
      context,
      command_buffer,
      weight);
}

vTensor pack_biases(
    const c10::optional<Tensor>& bias,
    const Tensor& weight) {
  if (bias && bias->is_vulkan()) {
    return convert(*bias);
  }

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();  // Don't collect the timestamp since the command buffer doesn't record anything

  const int64_t src_w = weight.size(Layout::Filter::output);
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

bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef /* output_padding */,
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
                                       (transposed ? false /* to be addded in the future */
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
         (dilation[Layout::Parameter::height] > 0) &&
         (dilation[Layout::Parameter::width] > 0) &&
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

} // namespace

Conv2dOpContext::Conv2dOpContext(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool /* transposed */,
    const IntArrayRef /* output_padding */,
    const int64_t groups,
    const Conv2dMethod method,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max)
  : packed_{
      pack_weights(weight, method),
      pack_biases(bias, weight),
      pack_filter(weight, expand_param_if_needed(dilation, "dilation", 2)),
      pack_params(expand_param_if_needed(stride, "stride", 2)),
      pack_params(expand_param_if_needed(padding, "padding", 2)),
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
      dilation.vec(),
      groups,
      output_min,
      output_max,
    },
    method_(method) {
}

Conv2dOpContext Conv2dOpContext::create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool transposed,
    const IntArrayRef output_padding_arg,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  const auto stride = expand_param_if_needed(stride_arg, "stride", 2);
  const auto padding = expand_param_if_needed(padding_arg, "padding", 2);
  const auto dilation = expand_param_if_needed(dilation_arg, "dilation", 2);
  const auto output_padding = output_padding_arg; // TODO: Deconvolutions

  TORCH_CHECK(
      available(
          weight,
          bias,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          output_min,
          output_max),
      "Vulkan::convolution not available! "
      "Reason: The provided (weight, bias, stride, padding, dilation, groups, "
      "transposed, output_padding, output_min, output_max) parameters are either "
      "invalid individually or their combination is not supported by Vulkan impl.");

  const auto method = determine_method(
      weight.sizes(),
      stride,
      padding,
      dilation,
      groups);

  // Pass in the originals
  return Conv2dOpContext{
    weight,
    bias,
    stride_arg,
    padding_arg,
    dilation_arg,
    transposed,
    output_padding_arg,
    groups,
    method,
    output_min,
    output_max
  };
}

void Conv2dOpContext::conv2d_sliding_window(
    const api::Shader::Descriptor& shader,
    vTensor& v_output,
    const vTensor& v_input,
    const std::string& op_name) const {
  bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && packed_.v_weight.has_image());
  TORCH_CHECK(valid, "Not Implemented!")

  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

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
      safe_downcast<int32_t>(packed_.filter[Layout::Filter::input]),
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
    if (method_ == Conv2dPointwise) {
      global_size = {
        safe_downcast<uint32_t>(div_up(v_output.sizes()[Layout::Filter::width], INT64_C(2))),
        safe_downcast<uint32_t>(div_up(v_output.sizes()[Layout::Filter::height], INT64_C(2))),
        v_output.extents().data[2u]
      };
    }

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

void Conv2dOpContext::conv2d_winograd_2_3(
    vTensor& v_output,
    const vTensor& v_input) const {
  // Winograd(2x2, 3x3) calculates 2x2 tile of output for every subprogram
  const int64_t out_h_units = div_up(v_output.sizes()[Layout::Activation4D::height], INT64_C(2));
  const int64_t out_w_units = div_up(v_output.sizes()[Layout::Activation4D::width], INT64_C(2));

  bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && packed_.v_weight.has_image());
  TORCH_CHECK(valid, "Not Implemented!")

  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), "prepacked::conv2d_clamp_run (conv2d_winograd_2_3)");

    vTensor v_input_winograd{
      context,
      {
        v_input.sizes()[Layout::Activation4D::batch],
        v_input.sizes()[Layout::Activation4D::channels],
        out_h_units*4,
        out_w_units*4,
      },
      v_output.options(),
    };

    {
      const struct TransformBlock final {
        uvec3 extents;
        uint32_t fill;
        ivec2 limits;
        ivec2 padding;
      } transform_block {
        v_input_winograd.extents(),
        0u,
        {
          safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::width]),
          safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::height]),
        },
        {
          safe_downcast<int32_t>(packed_.padding[Layout::Parameter::width]),
          safe_downcast<int32_t>(packed_.padding[Layout::Parameter::height]),
        },
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(transform_winograd_2_3_sh),
          v_input_winograd.extents(),
          adaptive_work_group_size(v_input_winograd.extents()),
          v_input_winograd.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          v_input.image(
              command_buffer,
              vTensor::Stage::Compute),
          context->resource().pool.uniform(transform_block).object);

    }
    {
      const struct Block final {
        uvec3 extents;
        int32_t ic4;
        vec2 clamp;
      } block {
        v_output.extents(),
        safe_downcast<int32_t>(packed_.filter[Layout::Filter::input] / 4),
        {
          packed_.output_min,
          packed_.output_max,
        },
      };

      uvec3 global_size = {
        safe_downcast<uint32_t>(out_w_units),
        safe_downcast<uint32_t>(out_h_units),
        v_output.extents().data[2u],
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(conv2d_winograd_2_3),
          global_size,
          adaptive_work_group_size(global_size),
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          v_input_winograd.image(
              command_buffer,
              vTensor::Stage::Compute),
          packed_.v_weight.image(
              command_buffer,
              vTensor::Stage::Compute),
          packed_.v_bias.buffer(
              command_buffer,
              vTensor::Stage::Compute),
          context->resource().pool.uniform(block).object);
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);
}

Tensor Conv2dOpContext::run(const Tensor& input_arg) const {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  TORCH_CHECK(
      usable(input),
      "Vulkan Convolution not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by Vulkan impl.");

  vTensor v_output{
    context,
    conv_output_size(
        v_input.sizes(),
        unpacked_.filter,
        packed_.padding,
        packed_.stride,
        packed_.dilation),
    input.options(),
  };

  switch(method_) {
    case Conv2dWinograd_2_3:
      conv2d_winograd_2_3(v_output, v_input);
    case Conv2dDepthwise:
      conv2d_sliding_window(
        VK_KERNEL(conv2d_dw),
        v_output,
        v_input,
        "prepacked::conv2d_clamp_run (conv2d_sliding_window::conv2d_dw)");
      break;
    case Conv2dPointwise:
      conv2d_sliding_window(
        VK_KERNEL(conv2d_pw_2x2),
        v_output,
        v_input,
        "prepacked::conv2d_clamp_run (conv2d_sliding_window::conv2d_pw_2x2)");
      break;
    default:
      conv2d_sliding_window(
        VK_KERNEL(conv2d),
        v_output,
        v_input,
        "prepacked::conv2d_clamp_run (conv2d_sliding_window::conv2d)");
      break;
  }

  return convert(v_output);
}

Conv2dOpContext::State Conv2dOpContext::unpack() const {
  return Conv2dOpContext::State{
    unpacked_.weight,
    unpacked_.bias,
    unpacked_.stride,
    unpacked_.padding,
    unpacked_.dilation,
    unpacked_.groups,
    unpacked_.output_min,
    unpacked_.output_max,
  };
}

c10::intrusive_ptr<Conv2dOpContext> conv2d_clamp_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dOpContext>(
      Conv2dOpContext::create(
          std::move(weight),
          std::move(bias),
          std::move(stride),
          std::move(padding),
          std::move(dilation),
          /* transposed = */ false,
          /* output_padding = */ {},
          groups,
          output_min,
          output_max));
}

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& context) {
  return context->run(input);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
