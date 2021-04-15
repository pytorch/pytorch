#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/ops/Persistent.h>
#include <ATen/native/vulkan/api/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

struct Experimentation final {
  static constexpr bool kUseConv2dOldApi = false;
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
  for (size_t i = 0; i < arr.size(); i++) {
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
  if (Experimentation::kUseConv2dOldApi)
    return Conv2dOld;
  if (is_pointwise(filter))
    return Conv2dPointwise;
  if (Experimentation::kUseWinogradConvs && is_winograd_n_3(filter, stride, dilation))
    return Conv2dWinograd_2_3;
  return Conv2dSlidingWindow;
}

vTensor pack_weights_dw(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    api::Resource::Pool& pool,
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
      &pool,
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

  for (int64_t src_oc = 0; src_oc < src_filter[Layout::Filter::output]; ++src_oc) {
    /* Source */
    const float* const src_weight_oc_ptr = src_weight_ptr + src_oc * src_block_sz;

    /* Destination */
    const int64_t dst_oh = src_oc / 4;
    const int64_t dst_c = src_oc % 4;

    float* const dst_weight_c_ptr = dst_weight_ptr +
                                    dst_c * dst_kernel_sz +
                                    dst_oh * dst_kw_sz;

    for (int64_t src_ih = 0; src_ih < src_filter[Layout::Filter::height]; ++src_ih) {
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
    api::Resource::Pool& pool,
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
      &pool,
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

  for (int64_t src_oc = 0; src_oc < src_filter[Layout::Filter::output]; ++src_oc) {
    /* Source */
    const float* const src_weight_oc_ptr = src_weight_ptr + src_oc * src_block_sz;

    /* Destination */
    const int64_t dst_oh = src_oc / 4;
    const int64_t dst_c = src_oc % 4;

    float* const dst_weight_c_ptr = dst_weight_ptr + dst_c * dst_kernel_sz;

    for (int64_t src_ic = 0; src_ic < src_filter[Layout::Filter::input]; ++src_ic) {
      const int64_t dst_ic4 = src_ic / 4;

      for (int64_t src_ih = 0; src_ih < src_kh_sz; ++src_ih) {
        for (int64_t src_iw = 0; src_iw < src_kw_sz; ++src_iw) {
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

vTensor pack_weights_2d_old(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    api::Resource::Pool& pool,
    const Tensor& weight) {
  const IntArrayRef src_filter = weight.sizes();
  const float* const src_weight_ptr = weight.data_ptr<float>();

  const uint32_t OC = src_filter[Layout::Filter::output];
  const uint32_t OC_4 = at::native::vulkan::api::utils::div_up(OC, 4u);
  const uint32_t C = src_filter[Layout::Filter::input];
  const uint32_t C_4 = at::native::vulkan::api::utils::div_up(C, 4u);
  const uint32_t KH = src_filter[Layout::Filter::height];
  const uint32_t KW = src_filter[Layout::Filter::width];

  vTensor v_weight{
    context,
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
  Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
  Future::Payload v_weight_payload = v_weight_future.wait();

  float* const dst_weight_ptr = v_weight_payload.get();
  memset(dst_weight_ptr, 0, v_weight.nbytes());

  const float* const src = src_weight_ptr;
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
        data_ = new float[dim0 * dim1 * dim2 * 4];  // TODO: memory leak
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

vTensor pack_weights_2d_winograd_2_3(
    api::Context* const context,
    api::Command::Buffer& command_buffer,
    api::Resource::Pool& pool,
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
      &pool,
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

  for (int64_t src_oc = 0; src_oc < src_oc_sz; ++src_oc) {
    const int64_t dst_oh = src_oc / 4;
    const int64_t dst_iw = src_oc % 4;

    for (int64_t src_ic = 0; src_ic < src_ic_sz; ++src_ic) {
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
    api::Resource::Pool& pool,
    const Tensor& weight_arg,
    const Conv2dMethod conv_method) {
  if (weight_arg.is_vulkan()) {
    return convert(weight_arg);
  }

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();

  const Tensor weight = weight_arg.contiguous();

  if (conv_method == Conv2dDepthwise) {
    return pack_weights_dw(
        context,
        command_buffer,
        pool,
        weight);
  }

  if (conv_method == Conv2dOld) {
    return pack_weights_2d_old(
        context,
        command_buffer,
        pool,
        weight);
  }

  if (conv_method == Conv2dWinograd_2_3) {
    return pack_weights_2d_winograd_2_3(
        context,
        command_buffer,
        pool,
        weight);
  }

  return pack_weights_2d(
      context,
      command_buffer,
      pool,
      weight);
}

vTensor pack_biases(
    api::Resource::Pool& pool,
    const c10::optional<Tensor>& bias,
    const Tensor& weight) {
  if (bias && bias->is_vulkan()) {
    return convert(*bias);
  }

  api::Context* const context = api::context();
  api::Command::Buffer& command_buffer = context->command().pool.stream();

  vTensor v_bias{
    context,
    &pool,
    {
      // 1D
      weight.size(Layout::Filter::output),
    },
    weight.options(),
  };

  {
      using Future = vTensor::Future<void, vTensor::Access::Write>;
      Future v_bias_future = v_bias.host<void, vTensor::Access::Write>(command_buffer);
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

void conv2d_dw(
    api::Context* const context,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef src_filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max) {
  bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
  TORCH_CHECK(valid, "Not Implemented!")

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    const struct Block final {
      uvec3 extents;
      int32_t src_filter_width;
      ivec4 kernel;
      ivec2 stride;
      ivec2 padding;
      ivec2 dilate;
      vec2 clamp;
    } block {
      v_output.extents(),
      safe_downcast<int32_t>(src_filter[Layout::Filter::width]),
      {
        safe_downcast<int32_t>(filter[Layout::Filter::width]),
        safe_downcast<int32_t>(filter[Layout::Filter::height]),
        safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::width]),
        safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::height]),
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
        output_min,
        output_max,
      },
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
        context->gpu().adapter->local_work_group_size(),
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
  command_pool.submit(context->gpu().queue, command_buffer);
}

void conv2d_pw(
    api::Context* const context,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef src_filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max) {
  bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
  TORCH_CHECK(valid, "Not Implemented!")

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    const struct Block final {
      uvec3 extents;
      int32_t ic;
      ivec2 stride;
      ivec2 padding;
      vec2 clamp;
    } block {
      v_output.extents(),
      safe_downcast<int32_t>(filter[Layout::Filter::input]),
      {
        safe_downcast<int32_t>(stride[Layout::Parameter::width]),
        safe_downcast<int32_t>(stride[Layout::Parameter::height]),
      },
      {
        safe_downcast<int32_t>(padding[Layout::Parameter::width]),
        safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      },
      {
        output_min,
        output_max,
      },
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
        context->gpu().adapter->local_work_group_size(),
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
  command_pool.submit(context->gpu().queue, command_buffer);
}

void conv2d(
    api::Context* const context,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef src_filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max) {
  bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
  TORCH_CHECK(valid, "Not Implemented!")

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
      safe_downcast<int32_t>(filter[Layout::Filter::input] / 4),
      {
        safe_downcast<int32_t>(filter[Layout::Filter::width]),
        safe_downcast<int32_t>(filter[Layout::Filter::height]),
        safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::width]),
        safe_downcast<int32_t>(v_input.sizes()[Layout::Activation4D::height]),
      },
      {
        safe_downcast<int32_t>(src_filter[Layout::Filter::width] * 4),
        safe_downcast<int32_t>(src_filter[Layout::Filter::height]),
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
        output_min,
        output_max,
      },
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
        context->gpu().adapter->local_work_group_size(),
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
  command_pool.submit(context->gpu().queue, command_buffer);
}

void conv2d_winograd_2_3(
    api::Context* const context,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef src_filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max) {
  // Winograd(2x2, 3x3) calculates 2x2 tile of output for every subprogram
  const int64_t out_h_units = div_up(v_output.sizes()[Layout::Activation4D::height], INT64_C(2));
  const int64_t out_w_units = div_up(v_output.sizes()[Layout::Activation4D::width], INT64_C(2));

  bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
  TORCH_CHECK(valid, "Not Implemented!")

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();

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
        v_input.sizes()[Layout::Activation4D::width],
        v_input.sizes()[Layout::Activation4D::height],
      },
      {
        safe_downcast<int32_t>(padding[Layout::Parameter::width]),
        safe_downcast<int32_t>(padding[Layout::Parameter::height]),
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
        context->gpu().adapter->local_work_group_size(),
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
      safe_downcast<int32_t>(filter[Layout::Filter::input] / 4),
      {
        output_min,
        output_max,
      },
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
        VK_KERNEL(conv2d_winograd_2_3),
        {
          out_w_units,
          out_h_units,
          v_output.extents().data[2u],
        },
        context->gpu().adapter->local_work_group_size(),
        v_output.image(
            command_buffer,
            vTensor::Stage::Compute,
            vTensor::Access::Write),
        v_input_winograd.image(
            command_buffer,
            vTensor::Stage::Compute),
        v_weight.image(
            command_buffer,
            vTensor::Stage::Compute),
        v_bias.buffer(
            command_buffer,
            vTensor::Stage::Compute),
        context->resource().pool.uniform(block).object);
  }
  command_pool.submit(context->gpu().queue, command_buffer);
}

void conv2d_old(
    api::Context* const context,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef filter,
    const IntArrayRef src_filter,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max) {
  using namespace api::utils;
  bool valid = C10_LIKELY(v_output.has_image() && v_input.has_image() && v_weight.has_image());
  TORCH_CHECK(valid, "Not Implemented!")

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    const int32_t W = v_input.extents().data[0];
    const int32_t H = v_input.extents().data[1];
    const int32_t C_4 = v_input.extents().data[2];
    const int32_t C = 4 * C_4;

    const int32_t OW = v_output.extents().data[0];
    const int32_t OH = v_output.extents().data[1];
    const int32_t OC_4 = v_output.extents().data[2];
    const int32_t OC = 4 * OC_4;

    const struct Block final {
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
        v_output.extents(),
        context->gpu().adapter->local_work_group_size(),
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
  command_pool.submit(context->gpu().queue, command_buffer);
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
    const Conv2dMethod method,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max)
  : packed_{
      pack_weights(pool, weight, method),
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
    },
    method_(method) {
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
    pool,
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
    output_max,
  };
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

  {
    void (*conv_func) (
      api::Context* const,
      vTensor&,
      const vTensor&,
      const vTensor&,
      const vTensor&,
      const IntArrayRef,
      const IntArrayRef,
      const IntArrayRef,
      const IntArrayRef,
      const IntArrayRef,
      const float,
      const float
    );
    switch(method_) {
      case Conv2dDepthwise:
        conv_func = &conv2d_dw;
        break;
      case Conv2dPointwise:
        conv_func = &conv2d_pw;
        break;
      case Conv2dOld:
        conv_func = &conv2d_old;
        break;
      case Conv2dWinograd_2_3:
        conv_func = &conv2d_winograd_2_3;
        break;
      default:
        conv_func = &conv2d;
        break;
    }
    conv_func(
      context,
      v_output,
      v_input,
      packed_.v_weight,
      packed_.v_bias,
      packed_.filter,
      unpacked_.filter,
      packed_.stride,
      packed_.padding,
      packed_.dilation,
      packed_.output_min,
      packed_.output_max);
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
