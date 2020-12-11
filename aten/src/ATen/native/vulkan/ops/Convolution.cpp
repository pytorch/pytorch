#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

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

vTensor pack_weights(
    api::Resource::Pool& pool,
    const Tensor& weight_arg,
    const int64_t groups) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  /* Source */

  const Tensor weight = weight_arg.contiguous();
  const IntArrayRef src_filter = weight.sizes();
  const float* const src_weight_ptr = weight.data_ptr<float>();

  //
  // Depthwise
  //

  if (is_depthwise(src_filter, groups)) {
    vTensor v_weight{
        api::context(),
        &pool,
        src_filter,
        weight.options(),
    };

    using Future = vTensor::Future<void, vTensor::Access::Write>;
    Future v_weight_future = v_weight.host<void, vTensor::Access::Write>();
    Future::Payload v_weight_payload = v_weight_future.wait();

    memcpy(
        v_weight_payload.get(),
        src_weight_ptr,
        std::min(weight.nbytes(), v_weight.nbytes()));

    return v_weight;
  }

  //
  // General
  //

  if (Experimentation::kUseConv2dOldApi) {
    const uint32_t OC = src_filter[Layout::Filter::output];
    const uint32_t OC_4 = at::native::vulkan::api::utils::div_up(OC, 4u);
    const uint32_t C = src_filter[Layout::Filter::input];
    const uint32_t C_4 = at::native::vulkan::api::utils::div_up(C, 4u);
    const uint32_t KH = src_filter[Layout::Filter::height];
    const uint32_t KW = src_filter[Layout::Filter::width];

    vTensor v_weight{
      api::context(),
      &pool,
      {
        1,
        4 * KH * KW,
        OC_4,
        4 * C_4
      },
      weight.options(),
    };

    using Future = vTensor::Future<float, vTensor::Access::Write>;
    Future v_weight_future = v_weight.host<float, vTensor::Access::Write>();
    Future::Payload v_weight_payload = v_weight_future.wait();

    float* const dst_weight_ptr = v_weight_payload.get();
    memset(dst_weight_ptr, 0, v_weight.nbytes());

    const float* src = src_weight_ptr;
    float* const dst = dst_weight_ptr;

    {
      uint32_t ridx = 0;
      const uint32_t oc_4SizeNumel = KW * KH * C_4 * 16;
      for (uint32_t oc = 0; oc < OC; ++oc) {
        int oc_4 = oc / 4;
        int oc_4_i = oc % 4;
        float* dst_oc = dst + oc_4 * oc_4SizeNumel;
        for (uint32_t ic = 0; ic < C; ++ic) {
          int ic_4 = ic / 4;
          int ic_4_i = ic % 4;
          float* dst_ic = dst_oc + ic_4 * KW * KH * 16;
          for (uint32_t ky = 0; ky < KH; ++ky) {
            float* dst_ky = dst_ic + ky * KW * 16;
            for (uint32_t kx = 0; kx < KW; ++kx) {
              float* dst_kx = dst_ky + kx * 16;
              dst_kx[4 * ic_4_i + oc_4_i] = src[ridx++];
            }
          }
        }
      }

      // shader KO4C4HW_to_image
      struct Image3D {
        float* data_;
        uint32_t dim0_, dim1_, dim2_;

        Image3D(uint32_t dim0, uint32_t dim1, uint32_t dim2) {
          dim0_ = dim0;
          dim1_ = dim1;
          dim2_ = dim2;
          data_ = new float[dim0 * dim1 * dim2 * 4];
          memset(data_, 0.f, dim0 * dim1 * dim2 * 4 * sizeof(float));
        }

        inline uint32_t idx(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
          return i3 + i2 * 4 + i1 * 4 * dim2_ + i0 * 4 * dim2_ * dim1_;
        }

        void set(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, float value) {
          data_[idx(i0, i1, i2, i3)] = value;
        }

        float get(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
          return data_[idx(i0, i1, i2, i3)];
        }
      } image{4 * C_4, OC_4, KH * KW};

      for (uint32_t sx = 0; sx < C_4; ++sx) {
        for (uint32_t sy = 0; sy < OC_4; ++sy) {
          for (uint32_t sz = 0; sz < (KH * KW); ++sz) {
            for (uint32_t vi = 0; vi < 4; ++vi) {
              int bufferVIdx = 4 * sx * KH * KW + 4 * sy * C_4 * KH * KW + 4 * sz;
              image.set(4 * sx + 0, sy, sz, vi, dst[4 * (bufferVIdx + 0) + vi]);
              image.set(4 * sx + 1, sy, sz, vi, dst[4 * (bufferVIdx + 1) + vi]);
              image.set(4 * sx + 2, sy, sz, vi, dst[4 * (bufferVIdx + 2) + vi]);
              image.set(4 * sx + 3, sy, sz, vi, dst[4 * (bufferVIdx + 3) + vi]);
            }
          }
        }
      }

      // inverse function of nchw_to_image
      const uint32_t W = 4 * C_4;
      const uint32_t H = OC_4;
      const uint32_t D = KH * KW;
      for (uint32_t sx = 0; sx < W; ++sx) {
        for (uint32_t sy = 0; sy < H; ++sy) {
          for (uint32_t sz = 0; sz < D; ++sz) {
            for (uint32_t szvi = 0; szvi < 4; ++szvi) {
              dst_weight_ptr[W * sy + sx + (4 * sz + szvi) * W * H] = image.get(sx, sy, sz, szvi);
            }
          }
        }
      }
    }

    return v_weight;
  }

  const int64_t num_stacks = div_up(src_filter[Layout::Filter::output], INT64_C(4));
  const int64_t stack_depth =
      4 * api::utils::align_up(src_filter[Layout::Filter::input], INT64_C(4));
  const int64_t max_stacks_per_tower =
      ConvPrepackLimits::maxStackDepth / stack_depth;
  const int64_t num_towers = div_up(num_stacks, max_stacks_per_tower);
  int64_t stacks_per_tower = num_stacks;
  if (num_towers > 1) {
    stacks_per_tower = div_up(num_stacks, num_towers);
  }
  vTensor v_weight{
      api::context(),
      &pool,
      {
          stacks_per_tower,
          stack_depth,
          src_filter[Layout::Filter::height] * num_towers,
          src_filter[Layout::Filter::width],
      },
      weight.options(),
  };

  using Future = vTensor::Future<float, vTensor::Access::Write>;
  Future v_weight_future = v_weight.host<float, vTensor::Access::Write>();
  Future::Payload v_weight_payload = v_weight_future.wait();

  /* Source */
  const int64_t src_kw_sz = src_filter[Layout::Filter::width];
  const int64_t src_kh_sz = src_filter[Layout::Filter::height];
  const int64_t src_kernel_sz = src_kw_sz * src_kh_sz;
  const int64_t src_block_sz =
      src_kernel_sz * src_filter[Layout::Filter::input];

  /* Destination */
  const IntArrayRef dst_filter = v_weight.sizes();
  const int64_t dst_kw_sz = src_filter[Layout::Filter::width];
  const int64_t dst_kh_sz = src_filter[Layout::Filter::height] * num_towers;
  const int64_t dst_kernel_sz = dst_kw_sz * dst_kh_sz;
  const int64_t dst_block_sz =
      dst_kernel_sz * dst_filter[Layout::Filter::input];

  TORCH_INTERNAL_ASSERT(src_kernel_sz*num_towers == dst_kernel_sz, "Internal error!");

  float* const dst_weight_ptr = v_weight_payload.get();
  memset(dst_weight_ptr, 0, v_weight.nbytes());

  for (int64_t src_oc = 0; src_oc < src_filter[Layout::Filter::output]; ++src_oc) {
    const int64_t i_tower = src_oc / (stacks_per_tower * 4);
    /* Source */
    const float* const src_weight_oc_ptr =
        src_weight_ptr + src_oc * src_block_sz;

    /* Destination */
    const int64_t local_oc = src_oc % (stacks_per_tower * 4);
    const int64_t dst_oc = local_oc / 4;
    const int64_t dst_oc_offset = local_oc % 4;

    float* const dst_weight_oc_ptr = dst_weight_ptr + dst_oc * dst_block_sz +
        dst_oc_offset * dst_kernel_sz;

    for (int64_t src_ic = 0; src_ic < src_filter[Layout::Filter::input]; ++src_ic) {
      const int64_t dst_ic = 4 * src_ic;

      memcpy(
          dst_weight_oc_ptr + dst_ic * dst_kernel_sz +
              (i_tower * src_kernel_sz),
          src_weight_oc_ptr + src_ic * src_kernel_sz,
          sizeof(float) * src_kernel_sz);
    }
  }

  return v_weight;
}

vTensor pack_biases(
    api::Resource::Pool& pool,
    const c10::optional<Tensor>& bias,
    const Tensor& weight) {
  if (bias && bias->is_vulkan()) {
    return convert(*bias);
  }

  vTensor v_bias{
    api::context(),
    &pool,
    {
      // 1D
      weight.size(Layout::Filter::output),
    },
    weight.options(),
  };

  {
      using Future = vTensor::Future<void, vTensor::Access::Write>;
      Future v_bias_future = v_bias.host<void, vTensor::Access::Write>();
      Future::Payload v_bias_payload = v_bias_future.wait();

      if (bias) {
        memcpy(
            v_bias_payload.get(),
            bias->contiguous().data_ptr<float>(),
            std::min(bias->nbytes(), v_bias.nbytes()));
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
    const c10::optional<Scalar> output_min,
    const c10::optional<Scalar> output_max) {
  return api::available() &&
         // Weight
         (4 == weight.ndimension()) &&
         (weight.size(Layout::Filter::height) > 0) &&
         (weight.size(Layout::Filter::width) > 0) &&
         ((c10::DeviceType::CPU == weight.device().type()) ||
          (c10::DeviceType::Vulkan == weight.device().type())) &&
         (kFloat == weight.scalar_type()) &&
         // Bias
         ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                       ((c10::DeviceType::CPU == bias->device().type()) ||
                                        (c10::DeviceType::Vulkan == bias->device().type())) &&
                                       (kFloat == bias->scalar_type()) &&
                                       (transposed ? false /* to be addded in the future */
                                                   : (weight.size(Layout::Filter::output) == bias->size(Layout::Filter::output))))
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

void conv2d_depthwise(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max) {
  if (v_output.has_image() && v_input.has_image() && v_weight.has_image()) {
    const struct {
      int32_t kernel_x, kernel_y;
      int32_t stride_x, stride_y;
      int32_t padding_x, padding_y;
      int32_t dilate_x, dilate_y;
      float clamp_x, clamp_y;
    } block {
      safe_downcast<int32_t>(filter[Layout::Filter::width]),
      safe_downcast<int32_t>(filter[Layout::Filter::height]),
      safe_downcast<int32_t>(stride[Layout::Parameter::width]),
      safe_downcast<int32_t>(stride[Layout::Parameter::height]),
      safe_downcast<int32_t>(padding[Layout::Parameter::width]),
      safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      safe_downcast<int32_t>(dilation[Layout::Parameter::width]),
      safe_downcast<int32_t>(dilation[Layout::Parameter::height]),
      output_min,
      output_max,
    };

    context->dispatch(
        command_buffer,
        {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        VK_KERNEL(conv2d_dw),
        v_output.extents(),
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
        v_bias.buffer(
            command_buffer,
            vTensor::Stage::Compute),
        // Object lifetime is managed by the resource pool.
        // It is OK not to keep track of the handle.
        context->resource().pool.uniform(block).object);
  }
  else {
    TORCH_CHECK(false, "Not implemented!");
  }
}

void conv2d_pointwise(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const float output_min,
    const float output_max) {
  if (v_output.has_image() && v_input.has_image() && v_weight.has_image()) {
    const int64_t stacks_per_tower = v_weight.sizes()[0];

    const struct {
      int32_t kernel_ic, kernel_oc;
      int32_t stride_x, stride_y;
      int32_t padding_x, padding_y;
      float clamp_x, clamp_y;
      int32_t stacks_per_tower;
    } block {
      safe_downcast<int32_t>(filter[Layout::Filter::input]),
      safe_downcast<int32_t>(filter[Layout::Filter::output]),
      safe_downcast<int32_t>(stride[Layout::Parameter::width]),
      safe_downcast<int32_t>(stride[Layout::Parameter::height]),
      safe_downcast<int32_t>(padding[Layout::Parameter::width]),
      safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      output_min,
      output_max,
      safe_downcast<int32_t>(stacks_per_tower),
    };

    context->dispatch(
        command_buffer,
        {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        VK_KERNEL(conv2d_pw),
        v_output.extents(),
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
        v_bias.buffer(
            command_buffer,
            vTensor::Stage::Compute),
        // Object lifetime is managed by the resource pool.
        // It is OK not to keep track of the handle.
        context->resource().pool.uniform(block).object);
  }
  else {
    TORCH_CHECK(false, "Not implemented!");
  }
}

void conv2d(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max) {
  if (v_output.has_image() && v_input.has_image() && v_weight.has_image()) {
    const int64_t stacks_per_tower = v_weight.sizes()[0];
    const struct {
      int32_t kernel_x, kernel_y, kernel_ic, kernel_oc;
      int32_t stride_x, stride_y;
      int32_t padding_x, padding_y;
      int32_t dilate_x, dilate_y;
      float clamp_x, clamp_y;
      int32_t stacks_per_tower;
    } block {
      safe_downcast<int32_t>(filter[Layout::Filter::width]),
      safe_downcast<int32_t>(filter[Layout::Filter::height]),
      safe_downcast<int32_t>(filter[Layout::Filter::input]),
      safe_downcast<int32_t>(filter[Layout::Filter::output]),
      safe_downcast<int32_t>(stride[Layout::Parameter::width]),
      safe_downcast<int32_t>(stride[Layout::Parameter::height]),
      safe_downcast<int32_t>(padding[Layout::Parameter::width]),
      safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      safe_downcast<int32_t>(dilation[Layout::Parameter::width]),
      safe_downcast<int32_t>(dilation[Layout::Parameter::height]),
      output_min,
      output_max,
      safe_downcast<int32_t>(stacks_per_tower),
    };

    context->dispatch(
        command_buffer,
        {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        VK_KERNEL(conv2d),
        v_output.extents(),
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
        v_bias.buffer(
            command_buffer,
            vTensor::Stage::Compute),
        // Object lifetime is managed by the resource pool.
        // It is OK not to keep track of the handle.
        context->resource().pool.uniform(block).object);
  }
  else {
    TORCH_CHECK(false, "Not implemented!");
  }
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
  return Conv2dOpContext::create(
      api::context()->resource().pool,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups
  ).run(input);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("convolution_overrideable", convolution);
}

#endif /* USE_VULKAN_API */

} // namespace

Conv2dOpContext::Conv2dOpContext(
    api::Resource::Pool& pool,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool /* transposed */,
    const IntArrayRef /* output_padding */,
    const int64_t groups,
    const c10::optional<Scalar> output_min,
    const c10::optional<Scalar> output_max)
  : packed_{
      pack_weights(pool, weight, groups),
      pack_biases(pool, bias, weight),
      pack_filter(weight, expand_param_if_needed(dilation, "dilation", 2)),
      pack_params(expand_param_if_needed(stride, "stride", 2)),
      pack_params(expand_param_if_needed(padding, "padding", 2)),
      pack_params(expand_param_if_needed(dilation, "dilation", 2)),
      groups,
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
    } {
}

Conv2dOpContext Conv2dOpContext::create(
    api::Resource::Pool& pool,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool transposed,
    const IntArrayRef output_padding_arg,
    const int64_t groups,
    const c10::optional<Scalar> output_min,
    const c10::optional<Scalar> output_max) {
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

  // Pass in the originals
  return Conv2dOpContext{
    pool,
    weight,
    bias,
    stride_arg,
    padding_arg,
    dilation_arg,
    transposed,
    output_padding_arg,
    groups,
    output_min,
    output_max,
  };
}

void conv2d_old(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max) {

  using namespace api::utils;

  if (v_output.has_image() && v_input.has_image() && v_weight.has_image()) {
    const int32_t W = v_input.extents().data[0];
    const int32_t H = v_input.extents().data[1];
    const int32_t C_4 = v_input.extents().data[2];
    const int32_t C = 4 * C_4;

    const int32_t OW = v_output.extents().data[0];
    const int32_t OH = v_output.extents().data[1];
    const int32_t OC_4 = v_output.extents().data[2];
    const int32_t OC = 4 * OC_4;

    const struct {
      int32_t padding_x, padding_y;
      int32_t kernel_x, kernel_y;
      int32_t stride_x, stride_y;
      int32_t dilate_x, dilate_y;
      int32_t outputSize[4];
      int32_t inputSize[4];
      float outputMin;
      float outputMax;
    } block {
      safe_downcast<int32_t>(padding[Layout::Parameter::width]),
      safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      safe_downcast<int32_t>(filter[Layout::Filter::width]),
      safe_downcast<int32_t>(filter[Layout::Filter::height]),
      safe_downcast<int32_t>(stride[Layout::Parameter::width]),
      safe_downcast<int32_t>(stride[Layout::Parameter::height]),
      safe_downcast<int32_t>(dilation[Layout::Parameter::width]),
      safe_downcast<int32_t>(dilation[Layout::Parameter::height]),
      { OW, OH, OC_4, OC },
      { W, H, C_4, C },
      output_min,
      output_max,
    };

    context->dispatch(
        command_buffer,
        {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        VK_KERNEL(conv2d_nogroup_clamp),
        //VK_KERNEL(conv2d_nogroup_clamp_1x),
        v_output.extents(),
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
        v_bias.buffer(
          command_buffer,
          vTensor::Stage::Compute),
        // Object lifetime is managed by the resource pool.
        // It is OK not to keep track of the handle.
        context->resource().pool.uniform(block).object);
  }
  else {
    TORCH_CHECK(false, "Not implemented!");
  }
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

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (is_depthwise(unpacked_.filter, unpacked_.groups)) {
      conv2d_depthwise(
          context,
          command_buffer,
          v_output,
          v_input,
          packed_.v_weight,
          packed_.v_bias,
          packed_.filter,
          packed_.stride,
          packed_.padding,
          packed_.dilation,
          packed_.output_min,
          packed_.output_max);
    }
    else {
      if (Experimentation::kUseConv2dOldApi) {
        conv2d_old(
            context,
            command_buffer,
            v_output,
            v_input,
            packed_.v_weight,
            packed_.v_bias,
            packed_.filter,
            packed_.stride,
            packed_.padding,
            packed_.dilation,
            packed_.output_min,
            packed_.output_max);
      } else {
        if (is_pointwise(unpacked_.filter)) {
          conv2d_pointwise(
              context,
              command_buffer,
              v_output,
              v_input,
              packed_.v_weight,
              packed_.v_bias,
              packed_.filter,
              packed_.stride,
              packed_.padding,
              packed_.output_min,
              packed_.output_max);
        }
        else {
          conv2d(
              context,
              command_buffer,
              v_output,
              v_input,
              packed_.v_weight,
              packed_.v_bias,
              packed_.filter,
              packed_.stride,
              packed_.padding,
              packed_.dilation,
              packed_.output_min,
              packed_.output_max);
        }
      }
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

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
    const c10::optional<Scalar> output_min,
    const c10::optional<Scalar> output_max) {
  return c10::make_intrusive<Conv2dOpContext>(
      Conv2dOpContext::create(
          persistent()->pool,
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
