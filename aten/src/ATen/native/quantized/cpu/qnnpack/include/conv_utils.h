#pragma once
#include <array>
#include <cinttypes>
#include <cmath>
#include <string>

#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>

// TODO: Might combine the conv and deconv parameters/

namespace qnnpack {

namespace {
static inline size_t compute_output_dimension(
    size_t input_dim,       // Input dimension
    size_t pad_dim,         // Input padding
    size_t adjustment_dim,  // Adjustment to the output dimension
    size_t kernel_dim,      // Kernel dimension
    size_t dilation_dim,    // Dilation dimension
    size_t stride_dim,      // Stride or subsampling dimension
    bool transpose) {       // Transposed convolution
  kernel_dim = (kernel_dim - 1) * dilation_dim + 1;  // Effective kernel dim
  if (transpose) {
    return stride_dim * (input_dim - 1) + adjustment_dim + kernel_dim - pad_dim;
  } else {
    return (input_dim + pad_dim - kernel_dim) / stride_dim + 1;
  }
}
}  // namespace

struct conv_param_t {
  const std::array<uint32_t, 2> kernel_dims; // kernel width, height
  const std::array<uint32_t, 2> subsampling_dims; // subsampling width, height
  const std::array<uint32_t, 2> dilation; // dilation width, height
  const std::array<uint32_t, 4> pad; // input padding top, left, bottom, right
  const uint32_t groups;
  const size_t input_channels;
  const size_t output_channels;
  const uint8_t kernel_zero_point;
  const float kernel_scale;
  const uint8_t output_min;
  const uint8_t output_max;

  // The following are derived parameters
  enum pytorch_qnnp_ukernel_type ukernel_type; // kernel type based on input params
  size_t group_input_channels;
  size_t group_output_channels;

  /**
   * @brief Constructor for initializing the convolution parameters.
   */
  conv_param_t(
      const std::array<uint32_t, 2> kernel,
      const std::array<uint32_t, 2> subsampling,
      const std::array<uint32_t, 2> dil,
      const std::array<uint32_t, 4> pd,
      const uint32_t grp,
      const size_t in_ch,
      const size_t out_ch,
      const uint8_t kernel_zp,
      const float kernel_s,
      const uint8_t out_min,
      const uint8_t out_max)
      : kernel_dims(kernel),
        subsampling_dims(subsampling),
        dilation(dil),
        pad(pd),
        groups(grp),
        input_channels(in_ch),
        output_channels(out_ch),
        kernel_zero_point(kernel_zp),
        kernel_scale(kernel_s),
        output_min(out_min),
        output_max(out_max) {
    const uint32_t kernel_width = kernel_dims[0];
    const uint32_t kernel_height = kernel_dims[1];

    const uint32_t input_padding_top = pad[0];
    const uint32_t input_padding_left = pad[1];
    const uint32_t input_padding_bottom = pad[2];
    const uint32_t input_padding_right = pad[3];

    group_input_channels = input_channels / groups;
    group_output_channels = output_channels / groups;

    if (kernel_width == 0 || kernel_height == 0) {
      pytorch_qnnp_log_error(
          "failed to create convolution with %" PRIu32 "x%" PRIu32
          " kernel: kernel dimensions must be non-zero",
          kernel_width,
          kernel_height);
      assert("Failed to initialize QNNPACK conv_param_t struct.");
    }

    if (subsampling_dims[0] == 0 || subsampling_dims[1] == 0) {
      pytorch_qnnp_log_error(
          "failed to create convolution with %" PRIu32 "x%" PRIu32
          " subsampling: "
          "subsampling dimensions must be non-zero",
          subsampling_dims[0],
          subsampling_dims[1]);
      assert("Failed to initialize QNNPACK conv_param_t struct.");
    }

    if (dilation[0] == 0 || dilation[1] == 0) {
      pytorch_qnnp_log_error(
          "failed to create convolution with %" PRIu32 "x%" PRIu32
          " dilation: "
          "dilation dimensions must be non-zero",
          dilation[0],
          dilation[1]);
      assert("Failed to initialize QNNPACK conv_param_t struct.");
    }

    if (kernel_scale <= 0.0f || !std::isnormal(kernel_scale)) {
      pytorch_qnnp_log_error(
          "failed to create convolution with %.7g kernel scale: scale must be"
          "finite and positive",
          kernel_scale);
      assert("Failed to initialize QNNPACK conv_param_t struct.");
    }

    if (subsampling_dims[1] > kernel_height) {
      pytorch_qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "x%" PRIu32
          " subsampling: "
          "height subsampling is greater than kernel height; subsampling should"
          " be performed before the convolution",
          kernel_width,
          kernel_height,
          subsampling_dims[0],
          subsampling_dims[1]);
    }

    if (subsampling_dims[0] > kernel_width) {
      pytorch_qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "x%" PRIu32
          " subsampling: "
          "width subsampling is greater than kernel width; subsampling should"
          " be performed before the convolution",
          kernel_width,
          kernel_height,
          subsampling_dims[0],
          subsampling_dims[1]);
    }

    if (input_padding_top >= kernel_height) {
      pytorch_qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "+%" PRIu32
          " height padding: "
          "input top padding is greater or equal to kernel height",
          kernel_width,
          kernel_height,
          input_padding_top,
          input_padding_bottom);
    }

    if (input_padding_bottom >= kernel_height) {
      pytorch_qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "+%" PRIu32
          " height padding: "
          "input bottom padding is greater or equal to kernel height",
          kernel_width,
          kernel_height,
          input_padding_top,
          input_padding_bottom);
    }

    if (input_padding_right >= kernel_width) {
      pytorch_qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "+%" PRIu32
          " width padding: "
          "input right padding is greater or equal to kernel width",
          kernel_width,
          kernel_height,
          input_padding_left,
          input_padding_right);
    }

    if (input_padding_left >= kernel_width) {
      pytorch_qnnp_log_info(
          "inefficiency in convolution with %" PRIu32 "x%" PRIu32
          " kernel and %" PRIu32 "+%" PRIu32
          " width padding: "
          "input left padding is greater or equal to kernel width",
          kernel_width,
          kernel_height,
          input_padding_left,
          input_padding_right);
    }

    const size_t kernel_size = kernel_height * kernel_width;

    ukernel_type = pytorch_qnnp_ukernel_type_none;
    const bool any_padding = (input_padding_left | input_padding_top
        | input_padding_right | input_padding_bottom) != 0;

    if ((kernel_size == 9 || kernel_size == 25) &&
        group_input_channels == 1 && group_output_channels == 1 && groups > 1) {
      ukernel_type = pytorch_qnnp_ukernel_type_dwconv;
    } else if (kernel_size == 1 && subsampling_dims[1] == 1 && subsampling_dims[0] == 1 && !any_padding) {
      ukernel_type = group_input_channels >= SIZE_MAX ? pytorch_qnnp_ukernel_type_xzp_gemm : pytorch_qnnp_ukernel_type_gemm;
    } else {
      ukernel_type = pytorch_qnnp_ukernel_type_conv;
    }
  }

  /**
   * @brief Computes the output dimensions given a 2D input.
   */
  std::array<size_t, 2> compute_output_dims(
      std::array<size_t, 2> input_dims) const {
    std::array<size_t, 2> output_dims;
    output_dims[0] = compute_output_dimension(input_dims[0],  // width
                                              pad[1] + pad[3],
                                              /*adjustment=*/0,
                                              kernel_dims[0],
                                              dilation[0],
                                              subsampling_dims[0],
                                              /*transpose=*/false);
    output_dims[1] = compute_output_dimension(input_dims[1],  // height
                                              pad[0] + pad[2],
                                              /*adjustment=*/0,
                                              kernel_dims[1],
                                              dilation[1],
                                              subsampling_dims[1],
                                              /*transpose=*/false);
    return output_dims;
  }
};

struct deconv_param_t {
  const std::array<uint32_t, 2> kernel_dims; // kernel width, kernel height
  const std::array<uint32_t, 2> stride_dims; // stride width, height
  const std::array<uint32_t, 2> dilation_dims; // dilation width, height
  const std::array<uint32_t, 4> padding_dims; // input padding top, left, bottom, right
  const std::array<uint32_t, 2> adjustment_dims; // output adjustment

  const uint32_t groups;
  const size_t input_channels;
  const size_t output_channels;
  const uint8_t kernel_zero_point;
  const float kernel_scale;
  const uint8_t output_min;
  const uint8_t output_max;

  // The following are derived parameters
  enum pytorch_qnnp_ukernel_type ukernel_type; // kernel type based on input params
  size_t group_input_channels;
  size_t group_output_channels;

  /**
   * @brief Constructor for initializing the deconvolution parameters.
   */
  deconv_param_t(
      const std::array<uint32_t, 2> kernel,
      const std::array<uint32_t, 2> stride,
      const std::array<uint32_t, 2> dilation,
      const std::array<uint32_t, 4> padding,
      const std::array<uint32_t, 2> adjustment,

      const uint32_t grps,
      const size_t in_ch,
      const size_t out_ch,
      const uint8_t kernel_zp,
      const float kernel_s,
      const uint8_t out_min,
      const uint8_t out_max)
      : kernel_dims(kernel),
        stride_dims(stride),
        dilation_dims(dilation),
        padding_dims(padding),
        adjustment_dims(adjustment),
        groups(grps),
        input_channels(in_ch),
        output_channels(out_ch),
        kernel_zero_point(kernel_zp),
        kernel_scale(kernel_s),
        output_min(out_min),
        output_max(out_max) {

    const uint32_t kernel_width = kernel_dims[0];
    const uint32_t kernel_height = kernel_dims[1];
    const uint32_t dilation_width = dilation_dims[0];
    const uint32_t dilation_height = dilation_dims[1];
    const uint32_t stride_width = stride_dims[0];
    const uint32_t stride_height = stride_dims[1];
    const uint32_t adjustment_width = adjustment_dims[0];
    const uint32_t adjustment_height = adjustment_dims[1];

    const uint32_t input_padding_top = padding_dims[0];
    const uint32_t input_padding_left = padding_dims[1];
    const uint32_t input_padding_bottom = padding_dims[2];
    const uint32_t input_padding_right = padding_dims[3];

    group_input_channels = input_channels / groups;
    group_output_channels = output_channels / groups;

    if (kernel_width == 0 || kernel_height == 0) {
      pytorch_qnnp_log_error(
          "failed to create deconvolution with %" PRIu32 "x%" PRIu32
          " kernel: kernel dimensions must be non-zero",
          kernel_width,
          kernel_height);
      assert("Failed to initialize QNNPACK deconv_param_t struct.");
    }

    if (stride_width == 0 || stride_height== 0) {
      pytorch_qnnp_log_error(
          "failed to create deconvolution with %" PRIu32 "x%" PRIu32
          " subsampling: "
          "subsampling dimensions must be non-zero",
          stride_width,
          stride_height);
      assert("Failed to initialize QNNPACK deconv_param_t struct.");
    }

    if (dilation_width == 0 || dilation_height == 0) {
      pytorch_qnnp_log_error(
          "failed to create deconvolution with %" PRIu32 "x%" PRIu32
          " dilation: "
          "dilation dimensions must be non-zero",
          dilation_width,
          dilation_height);
      assert("Failed to initialize QNNPACK deconv_param_t struct.");
    }

    if (kernel_scale <= 0.0f || !std::isnormal(kernel_scale)) {
      pytorch_qnnp_log_error(
          "failed to create deconvolution with %.7g kernel scale: scale must be"
          "finite and positive",
          kernel_scale);
      assert("Failed to initialize QNNPACK deconv_param_t struct.");
    }

    const size_t kernel_size = kernel_height * kernel_width;
    ukernel_type = pytorch_qnnp_ukernel_type_conv;
  }

  /**
   * @brief Computes the output dimensions given a 2D input.
   */
  std::array<size_t, 2> compute_output_dims(
      std::array<size_t, 2> input_dims) const {
    std::array<size_t, 2> output_dims;
    output_dims[0] = compute_output_dimension(input_dims[0],  // width
                                              padding_dims[1] + padding_dims[3],
                                              adjustment_dims[0],
                                              kernel_dims[0],
                                              dilation_dims[0],
                                              stride_dims[0],
                                              /*transpose=*/true);
    output_dims[1] = compute_output_dimension(input_dims[1],  // height
                                              padding_dims[0] + padding_dims[2],
                                              adjustment_dims[1],
                                              kernel_dims[1],
                                              dilation_dims[1],
                                              stride_dims[1],
                                              /*transpose=*/true);
    return output_dims;
  }
};
} // namespace qnnpack
