#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>

#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Utils.h>
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
    const int64_t groups,
    const bool transposed,
    const bool quantized) {
  if (transposed) {
    return TConv2dSlidingWindow;
  }
  if (is_depthwise(filter, groups)) {
    return quantized ? QConv2dDepthwise : Conv2dDepthwise;
  }
  if (is_pointwise(filter)) {
    return quantized ? QConv2dPointwise : Conv2dPointwise;
  }
  return quantized ? QConv2dSlidingWindow : Conv2dSlidingWindow;
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

vTensor pack_weights_2d(
    api::Context* const context,
    const Tensor& weight,
    bool reversed) {
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

      if (reversed) {
        for (const auto src_ic :
             c10::irange(src_filter[Layout::Filter::input])) {
          for (const auto src_ih : c10::irange(src_kh_sz)) {
            const int64_t dst_h = src_kh_sz - 1 - src_ih;
            for (const auto src_iw : c10::irange(src_kw_sz)) {
              const int64_t dst_w = src_kw_sz - 1 - src_iw;
              const int64_t dst_w_offset = dst_w * stack_depth;
              memcpy(
                  dst_weight_c_ptr + (dst_oh * src_kh_sz + dst_h) * dst_kw_sz +
                      src_ic + dst_w_offset,
                  src_weight_oc_ptr + src_ic * src_kernel_sz +
                      src_ih * src_kw_sz + src_iw,
                  sizeof(float));
            }
          }
        }
      } else {
        for (const auto src_ic :
             c10::irange(src_filter[Layout::Filter::input])) {
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
  }
  utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  return v_weight;
}

vTensor pack_weights_dw_q(api::Context* const context, const Tensor& weight) {
  /* Source */
  const IntArrayRef src_filter = weight.sizes();
  const c10::quint8* const src_weight_ptr = weight.data_ptr<c10::quint8>();

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
      weight.q_scale(),
      weight.q_zero_point(),
  };

  api::StorageBuffer staging(context, at::kFloat, v_weight.numcells());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    c10::quint8* dst_weight_ptr = mapping.template data<c10::quint8>();

    memset(dst_weight_ptr, 0, v_weight.nbytes());

    for (const auto src_oc : c10::irange(src_filter[Layout::Filter::output])) {
      /* Source */
      const c10::quint8* const src_weight_oc_ptr =
          src_weight_ptr + src_oc * src_block_sz;

      /* Destination */
      const int64_t dst_oh = src_oc / 4;
      const int64_t dst_c = src_oc % 4;

      c10::quint8* const dst_weight_c_ptr =
          dst_weight_ptr + dst_c * dst_kernel_sz + dst_oh * dst_kw_sz;

      for (const auto src_ih :
           c10::irange(src_filter[Layout::Filter::height])) {
        memcpy(
            dst_weight_c_ptr + src_ih * src_kw_sz,
            src_weight_oc_ptr + src_ih * src_kw_sz,
            sizeof(c10::quint8) * src_kw_sz);
      }
    }
  }
  ops::utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  return v_weight;
}

vTensor pack_weights_2d_q(api::Context* const context, const Tensor& weight) {
  /* Source */
  const IntArrayRef src_filter = weight.sizes();
  const c10::quint8* const src_weight_ptr = weight.data_ptr<c10::quint8>();

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
      weight.q_scale(),
      weight.q_zero_point(),
  };

  api::StorageBuffer staging(context, at::kFloat, v_weight.numcells());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    c10::quint8* dst_weight_ptr = mapping.template data<c10::quint8>();

    memset(dst_weight_ptr, 0, v_weight.nbytes());

    for (const auto src_oc : c10::irange(src_filter[Layout::Filter::output])) {
      /* Source */
      const c10::quint8* const src_weight_oc_ptr =
          src_weight_ptr + src_oc * src_block_sz;

      /* Destination */
      const int64_t dst_oh = src_oc / 4;
      const int64_t dst_c = src_oc % 4;

      c10::quint8* const dst_weight_c_ptr =
          dst_weight_ptr + dst_c * dst_kernel_sz;

      for (const auto src_ic : c10::irange(src_filter[Layout::Filter::input])) {
        const int64_t dst_ic4 = src_ic / 4;

        for (const auto src_ih : c10::irange(src_kh_sz)) {
          for (const auto src_iw : c10::irange(src_kw_sz)) {
            memcpy(
                dst_weight_c_ptr + (dst_oh * src_kh_sz + src_ih) * dst_kw_sz +
                    dst_ic4 * src_kw_sz * 4 + src_iw * 4 + src_ic % 4,
                src_weight_oc_ptr + src_ic * src_kernel_sz +
                    src_ih * src_kw_sz + src_iw,
                sizeof(c10::quint8));
          }
        }
      }
    }
  }
  ops::utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  return v_weight;
}

vTensor pack_weights(
    const Tensor& weight_arg,
    const bool transposed,
    const bool quantized,
    const Conv2dMethod conv_method) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  api::Context* const context = api::context();

  const Tensor weight = transposed
      ? at::permute(weight_arg, {1, 0, 2, 3}).contiguous()
      : weight_arg.contiguous();

  if (transposed) {
    return pack_weights_2d(context, weight, true);
  }
  if (quantized) {
    if (conv_method == QConv2dDepthwise) {
      return pack_weights_dw_q(context, weight);
    }
    return pack_weights_2d_q(context, weight);
  }
  if (conv_method == Conv2dDepthwise) {
    return pack_weights_dw(context, weight);
  }
  return pack_weights_2d(context, weight, false);
}

vTensor pack_biases_reg(
    const c10::optional<Tensor>& bias,
    const Tensor& weight,
    const bool transposed) {
  if (bias && bias->is_vulkan()) {
    return convert(*bias);
  }

  api::Context* const context = api::context();

  const int64_t src_w = weight.size(
      transposed ? Layout::TransposedFilter::output : Layout::Filter::output);
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
  utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

  return v_bias;
}

vTensor pack_biases_q(const c10::optional<Tensor>& bias, const Tensor& weight) {
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
      weight.q_scale(),
      weight.q_zero_point(),
  };

  api::StorageBuffer staging(context, at::kFloat, v_bias.numcells());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    c10::quint8* dst_bias_ptr = mapping.template data<c10::quint8>();

    if (bias) {
      const c10::quint8* const src_bias_ptr =
          bias->contiguous().data_ptr<c10::quint8>();

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

vTensor pack_biases(
    const c10::optional<Tensor>& bias,
    const Tensor& weight,
    const bool transposed,
    const bool quantized) {
  if (quantized) {
    return pack_biases_q(bias, weight);
  }
  return pack_biases_reg(bias, weight, transposed);
}

std::array<int64_t, 4> pack_filter(
    const Tensor& weight,
    const IntArrayRef dilation,
    const bool transposed) {
  const IntArrayRef filter = weight.sizes();

  const auto effective = [](const int64_t k, const int64_t d) {
    return k + (k - 1) * (d - 1);
  };

  return {
      align_up(
          transposed ? filter[Layout::TransposedFilter::output]
                     : filter[Layout::Filter::output],
          INT64_C(4)),
      align_up(
          transposed ? filter[Layout::TransposedFilter::input]
                     : filter[Layout::Filter::input],
          INT64_C(4)),
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

bool weight_valid(const Tensor& weight, const bool quantized) {
  return (4 == weight.ndimension()) &&
      (weight.size(Layout::Filter::height) > 0) &&
      (weight.size(Layout::Filter::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type() ||
       (quantized && c10::kQUInt8 == weight.scalar_type()));
}

bool bias_valid(
    const c10::optional<Tensor>& bias,
    const Tensor& weight,
    const bool transposed,
    const bool quantized) {
  if (bias && bias->defined()) {
    return (1 == bias->ndimension()) &&
        ((bias->device().is_cpu()) ||
         (c10::DeviceType::Vulkan == bias->device().type())) &&
        (kFloat == bias->scalar_type() ||
         (quantized && c10::kQUInt8 == bias->scalar_type())) &&
        (transposed ? (weight.size(Layout::TransposedFilter::output) ==
                       bias->size(Layout::Filter::output))
                    : (weight.size(Layout::Filter::output) ==
                       bias->size(Layout::Filter::output)));
  }
  return true;
}

bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const bool quantized,
    const IntArrayRef /* output_padding */,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return api::available() &&
      // Weight
      weight_valid(weight, quantized) &&
      // Bias
      bias_valid(bias, weight, transposed, quantized) &&
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
      (!output_max || output_max->isFloatingPoint()) && true;
}

bool usable(const Tensor& input, const bool quantized) {
  // Input
  return (4 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      (kFloat == input.scalar_type() ||
       (quantized && c10::kQUInt8 == input.scalar_type())) &&
      (input.size(Layout::Activation4D::batch) >= 0) &&
      (input.size(Layout::Activation4D::channels) > 0) &&
      (input.size(Layout::Activation4D::height) > 0) &&
      (input.size(Layout::Activation4D::width) > 0) && !input.requires_grad() &&
      true;
}

static inline std::vector<int64_t> get_conv_transpose_output_size(
    IntArrayRef input_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation = IntArrayRef()) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_input_channels_dim];
  for (const auto d : c10::irange(2, dim)) {
    output_size[d] = stride[d - 2] * (input_size[d] - 1) + weight_size[d] -
        2 * padding[d - 2] + output_padding[d - 2];
  }
  return output_size;
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
      safe_downcast<int32_t>(
          packed_filter[Layout::Filter::input]), /* this is aligned up */
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

void conv2d_sliding_window_q(
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
    const Conv2dMethod method_,
    const double scale,
    const int64_t zero_point) {
  api::Context* const context = api::context();

  const double scale_out = v_output.get_scale();
  const int64_t zero_point_out = v_output.get_zero_point();

  const double weight_scale = packed_v_weight.get_scale();
  const int64_t weight_zero_point = packed_v_weight.get_zero_point();

  const double bias_scale = packed_v_bias.get_scale();
  const int64_t bias_zero_point = packed_v_bias.get_zero_point();

  const struct Block final {
    uvec3 extents;
    int32_t ic4;
    ivec4 kernel;
    float scale_out;
    float scale;
    int32_t zero_point_out;
    int32_t zero_point;
    float weight_scale;
    float bias_scale;
    int32_t weight_zero_point;
    int32_t bias_zero_point;
    ivec2 ikernel;
    ivec2 stride;
    ivec2 padding;
    ivec2 dilate;
    vec2 clamp;
  } block{
      v_output.extents(),
      safe_downcast<int32_t>(packed_filter[Layout::Filter::input]),
      {
          safe_downcast<int32_t>(packed_filter[Layout::Filter::width]),
          safe_downcast<int32_t>(packed_filter[Layout::Filter::height]),
          safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::width]),
          safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::height]),
      },
      safe_downcast<float>(scale_out),
      safe_downcast<float>(scale),
      safe_downcast<int32_t>(zero_point_out),
      safe_downcast<int32_t>(zero_point),
      safe_downcast<float>(weight_scale),
      safe_downcast<float>(bias_scale),
      safe_downcast<int32_t>(weight_zero_point),
      safe_downcast<int32_t>(bias_zero_point),
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
  if (method_ == QConv2dPointwise) {
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

Tensor convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef output_padding,
    const int64_t groups) {
  Conv2dPackedContext conv_context = Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      false,
      output_padding,
      groups);

  return run_conv2d_context(
      input, c10::make_intrusive<Conv2dPackedContext>(conv_context));
}

Tensor quantized_convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef output_padding,
    const int64_t groups,
    const double out_scale,
    const int64_t out_zero_point) {
  if (transposed) {
    return run_tconv2d_context(
        input,
        c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            false,
            output_padding,
            groups)));
  }

  Conv2dPackedContext conv_context = Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      true,
      output_padding,
      groups);

  return run_qconv2d_context(
      input,
      out_scale,
      out_zero_point,
      c10::make_intrusive<Conv2dPackedContext>(conv_context));
}

} // namespace

Conv2dPackedContext::Conv2dPackedContext(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding_arg,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max)
    : unpacked_{c10::AnyType::get()} {
  const auto stride = expand_param_if_needed(stride_arg, "stride", 2);
  const auto padding = expand_param_if_needed(padding_arg, "padding", 2);
  const auto dilation = expand_param_if_needed(dilation_arg, "dilation", 2);
  const auto output_padding =
      expand_param_if_needed(output_padding_arg, "output_padding", 2);

  TORCH_CHECK(
      available(
          weight,
          bias,
          stride,
          padding,
          dilation,
          transposed,
          quantized,
          output_padding,
          groups,
          output_min,
          output_max),
      "Vulkan::convolution not available! "
      "Reason: The provided (weight, bias, stride, padding, dilation, groups, "
      "transposed, output_padding, output_min, output_max) parameters are either "
      "invalid individually or their combination is not supported by Vulkan impl.");

  const auto method = determine_method(
      weight.sizes(), stride, padding, dilation, groups, transposed, quantized);

  packed_.reserve(Packed::NumArgs);
  packed_.emplace_back(
      convert(pack_weights(weight, transposed, quantized, method)));
  packed_.emplace_back(
      convert(pack_biases(bias, weight, transposed, quantized)));
  packed_.emplace_back(pack_filter(weight, dilation, transposed));
  packed_.emplace_back(pack_params(stride));
  packed_.emplace_back(pack_params(padding));
  packed_.emplace_back(output_padding);
  packed_.emplace_back(pack_params(dilation));
  packed_.emplace_back(transposed);
  packed_.emplace_back(quantized);
  packed_.emplace_back(safe_downcast<int32_t>(groups));
  packed_.emplace_back(
      output_min ? output_min->template to<float>()
                 : -std::numeric_limits<float>::infinity());
  packed_.emplace_back(
      output_max ? output_max->template to<float>()
                 : +std::numeric_limits<float>::infinity());
  packed_.emplace_back(method);
  packed_.emplace_back(weight.sizes().vec());

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(bias);
    unpacked_.emplace_back(stride_arg.vec());
    unpacked_.emplace_back(padding_arg.vec());
    unpacked_.emplace_back(dilation_arg.vec());
    unpacked_.emplace_back(transposed);
    unpacked_.emplace_back(quantized);
    unpacked_.emplace_back(output_padding_arg.vec());
    unpacked_.emplace_back(groups);
    unpacked_.emplace_back(output_min);
    unpacked_.emplace_back(output_max);
  }
}

Conv2dPackedContext Conv2dPackedContext::pack(c10::impl::GenericList unpacked) {
  return Conv2dPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Bias),
      unpacked.get(Unpacked::Stride).toIntVector(),
      unpacked.get(Unpacked::Padding).toIntVector(),
      unpacked.get(Unpacked::Dilation).toIntVector(),
      unpacked.get(Unpacked::isTransposed).toBool(),
      unpacked.get(Unpacked::isQuantized).toBool(),
      unpacked.get(Unpacked::OutputPadding).toIntVector(),
      unpacked.get(Unpacked::Groups).toInt(),
      get_optional_scalar(unpacked, Unpacked::OutputMin),
      get_optional_scalar(unpacked, Unpacked::OutputMax));
}

c10::intrusive_ptr<Conv2dPackedContext> create_conv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ false,
      /* quantized = */ false,
      /* output_padding_arg = */ {0},
      groups,
      output_min,
      output_max));
}

c10::intrusive_ptr<Conv2dPackedContext> create_tconv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ true,
      /* quantized = */ false,
      output_padding,
      groups,
      output_min,
      output_max));
}

c10::intrusive_ptr<Conv2dPackedContext> create_qconv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ false,
      /* quantized = */ true,
      /* output_padding_arg = */ {},
      groups,
      output_min,
      output_max));
}

Tensor run_conv2d_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  const vTensor& packed_v_weight = convert(
      conv_context->get_val(Conv2dPackedContext::Packed::Weight).toTensor());
  const vTensor& packed_v_bias = convert(
      conv_context->get_val(Conv2dPackedContext::Packed::Bias).toTensor());
  const auto packed_filter =
      conv_context->get_val(Conv2dPackedContext::Packed::FilterSizes)
          .toIntVector();
  const auto packed_stride =
      conv_context->get_val(Conv2dPackedContext::Packed::Stride).toIntVector();
  const auto packed_padding =
      conv_context->get_val(Conv2dPackedContext::Packed::Padding).toIntVector();
  const auto packed_output_padding =
      conv_context->get_val(Conv2dPackedContext::Packed::OutputPadding)
          .toIntVector();
  const auto packed_dilation =
      conv_context->get_val(Conv2dPackedContext::Packed::Dilation)
          .toIntVector();
  const auto transposed =
      conv_context->get_val(Conv2dPackedContext::Packed::isTransposed).toBool();
  const auto quantized =
      conv_context->get_val(Conv2dPackedContext::Packed::isQuantized).toBool();
  const float packed_output_min = safe_downcast<float>(
      conv_context->get_val(Conv2dPackedContext::Packed::OutputMin).toDouble());
  const float packed_output_max = safe_downcast<float>(
      conv_context->get_val(Conv2dPackedContext::Packed::OutputMax).toDouble());
  const Conv2dMethod method_ =
      (Conv2dMethod)conv_context
          ->get_val(Conv2dPackedContext::Packed::ConvMethod)
          .toInt();
  const auto unpacked_filter =
      conv_context->get_val(Conv2dPackedContext::Packed::WeightSizes)
          .toIntVector();

  TORCH_CHECK(
      usable(input, quantized),
      "Vulkan Convolution not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by Vulkan impl.");

  vTensor v_output{
      context,
      transposed ? get_conv_transpose_output_size(
                       v_input.sizes(),
                       unpacked_filter,
                       packed_padding,
                       packed_output_padding,
                       packed_stride,
                       packed_dilation)
                 : conv_output_size(
                       v_input.sizes(),
                       unpacked_filter,
                       packed_padding,
                       packed_stride,
                       packed_dilation),
      input.options(),
  };

  switch (method_) {
    case TConv2dSlidingWindow:
      conv2d_sliding_window(
          VK_KERNEL(conv_transpose2d),
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

Tensor run_tconv2d_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context(input_arg, conv_context);
}

// TODO: this can probably be consolidated with the other run method
Tensor run_qconv2d_context(
    const Tensor& input_arg,
    double scale,
    int64_t zero_point,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  const vTensor& packed_v_weight = convert(
      conv_context->get_val(Conv2dPackedContext::Packed::Weight).toTensor());
  const vTensor& packed_v_bias = convert(
      conv_context->get_val(Conv2dPackedContext::Packed::Bias).toTensor());
  const auto packed_filter =
      conv_context->get_val(Conv2dPackedContext::Packed::FilterSizes)
          .toIntVector();
  const auto packed_stride =
      conv_context->get_val(Conv2dPackedContext::Packed::Stride).toIntVector();
  const auto packed_padding =
      conv_context->get_val(Conv2dPackedContext::Packed::Padding).toIntVector();
  const auto packed_output_padding =
      conv_context->get_val(Conv2dPackedContext::Packed::OutputPadding)
          .toIntVector();
  const auto packed_dilation =
      conv_context->get_val(Conv2dPackedContext::Packed::Dilation)
          .toIntVector();
  const auto quantized =
      conv_context->get_val(Conv2dPackedContext::Packed::isQuantized).toBool();
  const float packed_output_min = safe_downcast<float>(
      conv_context->get_val(Conv2dPackedContext::Packed::OutputMin).toDouble());
  const float packed_output_max = safe_downcast<float>(
      conv_context->get_val(Conv2dPackedContext::Packed::OutputMax).toDouble());
  const Conv2dMethod method_ =
      (Conv2dMethod)conv_context
          ->get_val(Conv2dPackedContext::Packed::ConvMethod)
          .toInt();
  const auto unpacked_filter =
      conv_context->get_val(Conv2dPackedContext::Packed::WeightSizes)
          .toIntVector();

  TORCH_CHECK(
      usable(input, quantized),
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
      scale,
      zero_point,
  };

  switch (method_) {
    case QConv2dSlidingWindow:
      conv2d_sliding_window_q(
          VK_KERNEL(quantized_conv2d),
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
          method_,
          v_input.get_scale(),
          v_input.get_zero_point());
      break;
    case QConv2dPointwise:
      conv2d_sliding_window_q(
          VK_KERNEL(quantized_conv2d_pw_2x2),
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
          method_,
          v_input.get_scale(),
          v_input.get_zero_point());
      break;
    case QConv2dDepthwise:
      conv2d_sliding_window_q(
          VK_KERNEL(quantized_conv2d_dw),
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
          method_,
          v_input.get_scale(),
          v_input.get_zero_point());
      break;
    default:
      TORCH_CHECK(false, "Invalid Method");
  }

  return convert_quantized(v_output);
}

Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    double out_scale,
    int64_t out_zero_point) {
  return quantized_convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups,
      out_scale,
      out_zero_point);
}

/* Backwards compatibility */
Conv2dOpContext::Conv2dOpContext(Conv2dPackedContext conv_context)
    : conv_context_{std::move(conv_context)} {}

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
  return Conv2dOpContext{Conv2dPackedContext(
      weight,
      bias,
      stride_arg,
      padding_arg,
      dilation_arg,
      transposed,
      /* quantized = */ false,
      output_padding_arg,
      groups,
      output_min,
      output_max)};
}

Tensor Conv2dOpContext::run(const Tensor& input_arg) const {
  return run_conv2d_context(
      input_arg, c10::make_intrusive<Conv2dPackedContext>(conv_context_));
}

Conv2dOpContext::State Conv2dOpContext::unpack() const {
  const c10::impl::GenericList unpacked_ = conv_context_.unpack();

  TORCH_CHECK(unpacked_.size() > 0u, "unpacked_ does not have any elements!");

  return Conv2dOpContext::State(
      unpacked_.get(Conv2dPackedContext::Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked_, Conv2dPackedContext::Unpacked::Bias),
      unpacked_.get(Conv2dPackedContext::Unpacked::Stride).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Padding).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Dilation).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Groups).toInt(),
      get_optional_scalar(unpacked_, Conv2dPackedContext::Unpacked::OutputMin),
      get_optional_scalar(unpacked_, Conv2dPackedContext::Unpacked::OutputMax));
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
      /* output_padding = */ {0},
      groups,
      output_min,
      output_max));
}

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& context) {
  return context->run(input);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("convolution_overrideable", convolution);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
