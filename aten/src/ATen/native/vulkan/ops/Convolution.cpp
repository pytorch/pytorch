#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>

#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/VulkanOpContext.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;
using namespace at::native::vulkan::ops;

inline bool is_depthwise(const IntArrayRef filter, const int64_t groups) {
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
  return Conv2dSlidingWindow;
}

vTensor pack_weights_dw(api::Context* const context, const Tensor& weight) {
  /* Source */
  const IntArrayRef src_filter = weight.sizes();
  const float* const src_weight_ptr = weight.data_ptr<float>();

  const int64_t src_kw_sz = src_filter[Layout::Filter::width];
  const int64_t src_kh_sz = src_filter[Layout::Filter::height];
  const int64_t src_kernel_sz = src_kw_sz * src_kh_sz;
  const int64_t src_block_sz =
      src_kernel_sz * src_filter[Layout::Filter::input];
  const int64_t num_stacks =
      div_up(src_filter[Layout::Filter::output], INT64_C(4));

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

  api::StorageBuffer staging(context, at::kFloat, v_weight.numcells());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    float* dst_weight_ptr = mapping.template data<float>();

    memset(dst_weight_ptr, 0, v_weight.nbytes());

    for (const auto src_oc : c10::irange(src_filter[Layout::Filter::output])) {
      /* Source */
      const float* const src_weight_oc_ptr =
          src_weight_ptr + src_oc * src_block_sz;

      /* Destination */
      const int64_t dst_oh = src_oc / 4;
      const int64_t dst_c = src_oc % 4;

      float* const dst_weight_c_ptr =
          dst_weight_ptr + dst_c * dst_kernel_sz + dst_oh * dst_kw_sz;

      for (const auto src_ih :
           c10::irange(src_filter[Layout::Filter::height])) {
        memcpy(
            dst_weight_c_ptr + src_ih * src_kw_sz,
            src_weight_oc_ptr + src_ih * src_kw_sz,
            sizeof(float) * src_kw_sz);
      }
    }
  }
  ops::utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  return v_weight;
}

vTensor pack_weights_2d(api::Context* const context, const Tensor& weight) {
  /* Source */
  const IntArrayRef src_filter = weight.sizes();
  const float* const src_weight_ptr = weight.data_ptr<float>();

  const int64_t src_kw_sz = src_filter[Layout::Filter::width];
  const int64_t src_kh_sz = src_filter[Layout::Filter::height];
  const int64_t src_kernel_sz = src_kw_sz * src_kh_sz;
  const int64_t src_block_sz =
      src_kernel_sz * src_filter[Layout::Filter::input];

  const int64_t num_stacks =
      div_up(src_filter[Layout::Filter::output], INT64_C(4));
  const int64_t stack_depth =
      api::utils::align_up(src_filter[Layout::Filter::input], INT64_C(4));

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

  api::StorageBuffer staging(context, at::kFloat, v_weight.numcells());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    float* dst_weight_ptr = mapping.template data<float>();

    memset(dst_weight_ptr, 0, v_weight.nbytes());

    for (const auto src_oc : c10::irange(src_filter[Layout::Filter::output])) {
      /* Source */
      const float* const src_weight_oc_ptr =
          src_weight_ptr + src_oc * src_block_sz;

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
                src_weight_oc_ptr + src_ic * src_kernel_sz +
                    src_ih * src_kw_sz + src_iw,
                sizeof(float));
          }
        }
      }
    }
  }
  ops::utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  return v_weight;
}

vTensor pack_weights(const Tensor& weight_arg, const Conv2dMethod conv_method) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  api::Context* const context = api::context();

  const Tensor weight = weight_arg.contiguous();

  if (conv_method == Conv2dDepthwise) {
    return pack_weights_dw(context, weight);
  }

  return pack_weights_2d(context, weight);
}

vTensor pack_biases(const c10::optional<Tensor>& bias, const Tensor& weight) {
  if (bias && bias->is_vulkan()) {
    return convert(*bias);
  }

  api::Context* const context = api::context();

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

  api::StorageBuffer staging(context, at::kFloat, v_bias.numcells());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    float* dst_bias_ptr = mapping.template data<float>();

    if (bias) {
      const float* const src_bias_ptr = bias->contiguous().data_ptr<float>();

      memset(dst_bias_ptr, 0, v_bias.nbytes());
      for (const auto i : c10::irange(src_w)) {
        const int64_t c = i % 4;
        const int64_t x = i / 4;
        dst_bias_ptr[c * packed_w + x] = src_bias_ptr[i];
      }
    } else {
      memset(
          dst_bias_ptr,
          // 2's complement integers and IEEE-754 floating point numbers both
          // have identical bit representations for 0, so can use memset which
          // only accepts uint8_t parameter.
          0,
          v_bias.nbytes());
    }
  }
  ops::utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

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
          filter[Layout::Filter::height], dilation[Layout::Parameter::height]),
      effective(
          filter[Layout::Filter::width], dilation[Layout::Parameter::width]),
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
      (4 == weight.ndimension()) && (weight.size(Layout::Filter::height) > 0) &&
      (weight.size(Layout::Filter::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type()) &&
      // Bias
      ((bias && bias->defined())
           ? ((1 == bias->ndimension()) &&
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
      (!output_max || output_max->isFloatingPoint()) && true;
}

bool usable(const Tensor& input) {
  // Input
  return (4 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      (kFloat == input.scalar_type()) &&
      (input.size(Layout::Activation4D::batch) >= 0) &&
      (input.size(Layout::Activation4D::channels) > 0) &&
      (input.size(Layout::Activation4D::height) > 0) &&
      (input.size(Layout::Activation4D::width) > 0) && !input.requires_grad() &&
      true;
}

} // namespace

VulkanOpContext conv2d_context_create(
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

  const auto method =
      determine_method(weight.sizes(), stride, padding, dilation, groups);

  c10::impl::GenericList packed_context{c10::AnyType::get()};
  packed_context.reserve(10);
  packed_context.emplace_back(convert(pack_weights(weight, method)));
  packed_context.emplace_back(convert(pack_biases(bias, weight)));
  packed_context.emplace_back(pack_filter(weight, dilation));
  packed_context.emplace_back(pack_params(stride));
  packed_context.emplace_back(pack_params(padding));
  packed_context.emplace_back(output_padding);
  packed_context.emplace_back(pack_params(dilation));
  packed_context.emplace_back(safe_downcast<int32_t>(groups));
  packed_context.emplace_back(
      output_min ? output_min->template to<float>()
                 : -std::numeric_limits<float>::infinity());
  packed_context.emplace_back(
      output_max ? output_max->template to<float>()
                 : +std::numeric_limits<float>::infinity());
  packed_context.emplace_back(method);

  c10::impl::GenericList unpacked_context{c10::AnyType::get()};
  unpacked_context.reserve(10);
  unpacked_context.emplace_back(weight);
  unpacked_context.emplace_back(bias);
  unpacked_context.emplace_back(weight.sizes().vec());
  unpacked_context.emplace_back(stride_arg.vec());
  unpacked_context.emplace_back(padding_arg.vec());
  unpacked_context.emplace_back(output_padding_arg.vec());
  unpacked_context.emplace_back(dilation_arg.vec());
  unpacked_context.emplace_back(groups);
  unpacked_context.emplace_back(output_min);
  unpacked_context.emplace_back(output_max);
  unpacked_context.emplace_back(method);

  return VulkanOpContext::create(packed_context, unpacked_context);
}

void conv2d_sliding_window(
    const api::ShaderSource& shader,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& packed_v_weight,
    const vTensor& packed_v_bias,
    const IntArrayRef packed_filter,
    const IntArrayRef packed_stride,
    const IntArrayRef packed_padding,
    const IntArrayRef packed_dilation,
    const float packed_output_min,
    const float packed_output_max,
    const IntArrayRef unpacked_filter,
    const Conv2dMethod method_) {
  api::Context* const context = api::context();

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
  } block{
      v_output.extents(),
      safe_downcast<int32_t>(packed_filter[Layout::Filter::input]),
      {
          safe_downcast<int32_t>(packed_filter[Layout::Filter::width]),
          safe_downcast<int32_t>(packed_filter[Layout::Filter::height]),
          safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::width]),
          safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::height]),
      },
      {
          safe_downcast<int32_t>(unpacked_filter[Layout::Filter::width]),
          safe_downcast<int32_t>(unpacked_filter[Layout::Filter::height]),
      },
      {
          safe_downcast<int32_t>(packed_stride[Layout::Parameter::width]),
          safe_downcast<int32_t>(packed_stride[Layout::Parameter::height]),
      },
      {
          safe_downcast<int32_t>(packed_padding[Layout::Parameter::width]),
          safe_downcast<int32_t>(packed_padding[Layout::Parameter::height]),
      },
      {
          safe_downcast<int32_t>(packed_dilation[Layout::Parameter::width]),
          safe_downcast<int32_t>(packed_dilation[Layout::Parameter::height]),
      },
      {
          packed_output_min,
          packed_output_max,
      },
  };

  uvec3 global_size = v_output.extents();
  if (method_ == Conv2dPointwise) {
    global_size = {
        safe_downcast<uint32_t>(
            div_up(v_output.sizes()[Layout::Filter::width], INT64_C(2))),
        safe_downcast<uint32_t>(
            div_up(v_output.sizes()[Layout::Filter::height], INT64_C(2))),
        v_output.extents().data[2u]};
  }

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      adaptive_work_group_size(global_size),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());
}

Tensor conv2d_context_run(
    const Tensor& input_arg,
    const c10::impl::GenericList& packed_context,
    const c10::impl::GenericList& unpacked_context) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  const vTensor& packed_v_weight = convert(packed_context.get(0).toTensor());
  const vTensor& packed_v_bias = convert(packed_context.get(1).toTensor());

  const auto packed_filter = packed_context.get(2).toIntVector();
  const auto packed_stride = packed_context.get(3).toIntVector();
  const auto packed_padding = packed_context.get(4).toIntVector();
  const auto packed_dilation = packed_context.get(6).toIntVector();
  const float packed_output_min = packed_context.get(8).toDouble();
  const float packed_output_max = packed_context.get(9).toDouble();
  const auto unpacked_filter = unpacked_context.get(2).toIntVector();
  const Conv2dMethod method_ = (Conv2dMethod)unpacked_context.get(10).toInt();

  TORCH_CHECK(
      usable(input),
      "Vulkan Convolution not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by Vulkan impl.");

  vTensor v_output{
      context,
      conv_output_size(
          v_input.sizes(),
          unpacked_filter,
          packed_padding,
          packed_stride,
          packed_dilation),
      input.options(),
  };

  switch (method_) {
    case Conv2dDepthwise:
      conv2d_sliding_window(
          VK_KERNEL(conv2d_dw),
          v_output,
          v_input,
          packed_v_weight,
          packed_v_bias,
          packed_filter,
          packed_stride,
          packed_padding,
          packed_dilation,
          packed_output_min,
          packed_output_max,
          unpacked_filter,
          method_);
      break;
    case Conv2dPointwise:
      conv2d_sliding_window(
          VK_KERNEL(conv2d_pw_2x2),
          v_output,
          v_input,
          packed_v_weight,
          packed_v_bias,
          packed_filter,
          packed_stride,
          packed_padding,
          packed_dilation,
          packed_output_min,
          packed_output_max,
          unpacked_filter,
          method_);
      break;
    default:
      conv2d_sliding_window(
          VK_KERNEL(conv2d),
          v_output,
          v_input,
          packed_v_weight,
          packed_v_bias,
          packed_filter,
          packed_stride,
          packed_padding,
          packed_dilation,
          packed_output_min,
          packed_output_max,
          unpacked_filter,
          method_);
      break;
  }

  return convert(v_output);
}

c10::intrusive_ptr<VulkanOpContext> create_conv2d_clamp_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return c10::make_intrusive<VulkanOpContext>(conv2d_context_create(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ false,
      /* output_padding_arg = */ {},
      groups,
      output_min,
      output_max));
}

Tensor run_conv2d_clamp_context(
    const Tensor& input,
    const c10::intrusive_ptr<VulkanOpContext>& vulkan_context) {
  return conv2d_context_run(
      input, vulkan_context->get_packed(), vulkan_context->get_unpacked());
}

/* Backwards compatibility */
Conv2dOpContext::Conv2dOpContext(VulkanOpContext vulkan_context)
    : vulkan_context_{std::move(vulkan_context)} {}

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
  return Conv2dOpContext{conv2d_context_create(
      weight,
      bias,
      stride_arg,
      padding_arg,
      dilation_arg,
      transposed,
      output_padding_arg,
      groups,
      output_min,
      output_max)};
}

Tensor Conv2dOpContext::run(const Tensor& input_arg) const {
  return conv2d_context_run(
      input_arg, vulkan_context_.get_packed(), vulkan_context_.get_unpacked());
}

Conv2dOpContext::State Conv2dOpContext::unpack() const {
  const c10::impl::GenericList unpacked_ =
      std::get<1>(vulkan_context_.get_state());
  const Tensor unpacked_weight = unpacked_.get(0).toTensor();
  const c10::optional<Tensor> unpacked_bias = unpacked_.get(1).isTensor()
      ? unpacked_.get(1).toTensor()
      : (c10::optional<Tensor>&)c10::nullopt;
  const std::vector<int64_t> unpacked_stride = unpacked_.get(2).toIntVector();
  const std::vector<int64_t> unpacked_padding = unpacked_.get(3).toIntVector();
  const std::vector<int64_t> unpacked_dilation = unpacked_.get(4).toIntVector();
  const int64_t unpacked_groups = unpacked_.get(5).toInt();
  const c10::optional<Scalar> unpacked_output_min = unpacked_.get(6).isScalar()
      ? unpacked_.get(6).toScalar()
      : (c10::optional<Scalar>)c10::nullopt;
  const c10::optional<Scalar> unpacked_output_max = unpacked_.get(6).isScalar()
      ? unpacked_.get(7).toScalar()
      : (c10::optional<Scalar>)c10::nullopt;
  return Conv2dOpContext::State{
      unpacked_weight,
      unpacked_bias,
      unpacked_stride,
      unpacked_padding,
      unpacked_dilation,
      unpacked_groups,
      unpacked_output_min,
      unpacked_output_max};
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
  return c10::make_intrusive<Conv2dOpContext>(Conv2dOpContext::create(
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
