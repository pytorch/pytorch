#pragma once

#ifdef USE_XNNPACK
#include <cstdint>

#include <ATen/core/Tensor.h>
#include <ATen/native/xnnpack/Common.h>

using xnnpack_operator = at::native::xnnpack::Operator;

namespace at {
namespace native {
namespace xnnp_utils {

/*
 * Return shape in the same order as the memory format
 * e.g. channels_last will return NHWC instead of NCHW
 */
std::vector<size_t> get_mem_format_aware_shape(const at::Tensor& in);

/*
 * Input is always int8_t, output can be [int8_t, uint8_t].
 * input  + offset = output
 * int8_t + 128    = uint8_t
 * int8_t + 0      = int8_t
 */
template <typename PT>
void q8_copy_int8_weight_and_add_offset(const at::Tensor& in, at::Tensor& out);

template <int kSpatialDim>
Tensor convert_conv_weights_to_channel_last_tensor(
    const at::Tensor& src,
    int groups,
    bool transpose);

/*
 * Series of create wrapper functions to call xnn_create_[de]conv* functions.
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_create_convolution2d_nhwc(
    uint32_t pad_top,
    uint32_t pad_right,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t ip_chan_stride,
    size_t op_chan_stride,
    int8_t izp,
    float ip_scale,
    int8_t kzp,
    const float* k_scales,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t ozp,
    float op_scale,
    int8_t op_min,
    int8_t op_max,
    uint32_t flags,
    xnn_operator_t* op,
    bool per_channel,
    bool transpose) {
  /* Symmetric quantization forces kzp = 0 */
  TORCH_CHECK(!kzp, "XNNPACK Q[SC]8 conv kernels expects kernel zero point to be zero."
                    "But got: ", kzp);

  if (transpose) {
    TORCH_CHECK(!per_channel, "XNNPACK Q[SC]8 does not have a per channel deconvolution!");
    return xnn_create_deconvolution2d_nhwc_qs8(
        pad_top,        /* uint32_t output_padding_top          */
        pad_right,      /* uint32_t output_padding_right        */
        pad_bottom,     /* uint32_t output_padding_bottom       */
        pad_left,       /* uint32_t output_padding_left         */
        kernel_h,       /* uint32_t kernel_height               */
        kernel_w,       /* uint32_t kernel_width                */
        stride_h,       /* uint32_t stride_height               */
        stride_w,       /* uint32_t stride_width                */
        dilation_h,     /* uint32_t dilation_height             */
        dilation_w,     /* uint32_t dilation_width              */
        groups,         /* uint32_t groups                      */
        group_input_channels,  /* size_t group_input_channels   */
        group_output_channels, /* size_t group_output_channels  */
        ip_chan_stride, /* size_t input_pixel_stride            */
        op_chan_stride, /* size_t output_pixel_stride           */
        izp,            /* int8_t input_zero_point              */
        ip_scale,       /* float input_scale                    */
        k_scales[0],    /* float kernel_scale                   */
        kernel,         /* const int8_t* kernel                 */
        bias,           /* const int32_t* bias                  */
        ozp,            /* int8_t output_zero_point             */
        op_scale,       /* float output_scale                   */
        op_min,         /* int8_t output_min                    */
        op_max,         /* int8_t output_max                    */
        flags,          /* uint32_t flags                       */
        nullptr,        /* xnn_caches_t caches                  */
        op);            /* xnn_operator_t* deconvolution_op_out */

  }

  if (!per_channel) {
    return xnn_create_convolution2d_nhwc_qs8(
        pad_top,        /* uint32_t input_padding_top         */
        pad_right,      /* uint32_t input_padding_right       */
        pad_bottom,     /* uint32_t input_padding_bottom      */
        pad_left,       /* uint32_t input_padding_left        */
        kernel_h,       /* uint32_t kernel_height             */
        kernel_w,       /* uint32_t kernel_width              */
        stride_h,       /* uint32_t subsampling_height        */
        stride_w,       /* uint32_t subsampling_width         */
        dilation_h,     /* uint32_t dilation_height           */
        dilation_w,     /* uint32_t dilation_width            */
        groups,         /* uint32_t groups                    */
        group_input_channels,  /* size_t group_input_channels */
        group_output_channels, /* size_t group_output_channels*/
        ip_chan_stride, /* size_t input_channel_stride        */
        op_chan_stride, /* size_t output_channel_stride       */
        izp,            /* int8_t input_zero_point            */
        ip_scale,       /* float input_scale                  */
        k_scales[0],    /* float kernel_scale                 */
        kernel,         /* const int8_t* kernel               */
        bias,           /* const int32_t* bias                */
        ozp,            /* int8_t output_zero_point           */
        op_scale,       /* float output_scale                 */
        op_min,         /* int8_t output_min                  */
        op_max,         /* int8_t output_max                  */
        flags,          /* uint32_t flags                     */
        nullptr,        /* xnn_caches_t caches                */
        op);            /* xnn_operator_t* convolution_op_out */
  } else { /* per_channel */
    return xnn_create_convolution2d_nhwc_qc8(
        pad_top,        /* uint32_t input_padding_top         */
        pad_right,      /* uint32_t input_padding_right       */
        pad_bottom,     /* uint32_t input_padding_bottom      */
        pad_left,       /* uint32_t input_padding_left        */
        kernel_h,       /* uint32_t kernel_height             */
        kernel_w,       /* uint32_t kernel_width              */
        stride_h,       /* uint32_t subsampling_height        */
        stride_w,       /* uint32_t subsampling_width         */
        dilation_h,     /* uint32_t dilation_height           */
        dilation_w,     /* uint32_t dilation_width            */
        groups,         /* uint32_t groups                    */
        group_input_channels,  /* size_t group_input_channels */
        group_output_channels, /* size_t group_output_channels*/
        ip_chan_stride, /* size_t input_channel_stride        */
        op_chan_stride, /* size_t output_channel_stride       */
        izp,            /* int8_t input_zero_point            */
        ip_scale,       /* float input_scale                  */
        k_scales,       /* const float* kernel_scale          */
        kernel,         /* const int8_t* kernel               */
        bias,           /* const int32_t* bias                */
        ozp,            /* int8_t output_zero_point           */
        op_scale,       /* float output_scale                 */
        op_min,         /* int8_t output_min                  */
        op_max,         /* int8_t output_max                  */
        flags,          /* uint32_t flags                     */
        nullptr,        /* xnn_caches_t caches                */
        op);            /* xnn_operator_t* convolution_op_out */
  }
}

/*
 * Series of setup wrapper functions to call xnn_setup_[de]conv* functions.
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_setup_convolution2d_nhwc(
    xnn_operator_t op,
    size_t batch,
    size_t in_h,
    size_t in_w,
    const int8_t* inp,
    int8_t* outp,
    pthreadpool_t pt_pool,
    bool per_channel = false,
    bool transpose = false,
    uint32_t adj_h = 0,
    uint32_t adj_w = 0) {
  if(transpose) {
    TORCH_CHECK(!per_channel, "XNNPACK Q[SC]8 does not have a per channel deconvolution!");
    return xnn_setup_deconvolution2d_nhwc_qs8(
        op,       /* xnn_operator_t deconvolution_op */
        batch,    /* size_t batch_size               */
        in_h,     /* size_t input_height             */
        in_w,     /* size_t input_width              */
        adj_h,    /* uint32_t adjustment_height      */
        adj_w,    /* uint32_t adjustment_width       */
        inp,      /* const int8_t* input             */
        outp,     /* int8_t* output                  */
        pt_pool); /* pthreadpool_t threadpool        */
  }

  if (!per_channel) {
    return xnn_setup_convolution2d_nhwc_qs8(
        op,       /* xnn_operator_t convolution_op */
        batch,    /* size_t batch_size             */
        in_h,     /* size_t input_height           */
        in_w,     /* size_t input_width            */
        inp,      /* const int8_t* input           */
        outp,     /* int8_t* output                */
        pt_pool); /* pthreadpool_t threadpool      */
  } else { /* per_channel */
    return xnn_setup_convolution2d_nhwc_qc8(
        op,       /* xnn_operator_t convolution_op */
        batch,    /* size_t batch_size             */
        in_h,     /* size_t input_height           */
        in_w,     /* size_t input_width            */
        inp,      /* const int8_t* input           */
        outp,     /* int8_t* output                */
        pt_pool); /* pthreadpool_t threadpool      */
  }
}


/*
 * Series of wrapper functions to call xnn_create* and xnn_setup*
 * functions for linear
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_create_fully_connected_nc(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    int8_t input_zero_point,
    float input_scale,
    int8_t kernel_zero_point,
    float kernel_scale,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* fully_connected_op_out) {
  /* Symmetric quantization forces kzp = 0 */
  TORCH_CHECK(!kernel_zero_point, "XNNPACK QS8 linear kernel expects kernel zero point to be zero."
                    "But got: ", kernel_zero_point);
  return xnn_create_fully_connected_nc_qs8(
      input_channels,          /* size_t input_channels                  */
      output_channels,         /* size_t output_channels                 */
      input_stride,            /* size_t input_stride                    */
      output_stride,           /* size_t output_stride                   */
      input_zero_point,        /* int8_t input_zero_point                */
      input_scale,             /* float input_scale                      */
      kernel_scale,            /* float kernel_scale                     */
      kernel,                  /* const int8_t* kernel                   */
      bias,                    /* const int32_t* bias                    */
      output_zero_point,       /* int8_t output_zero_point               */
      output_scale,            /* float output_scale                     */
      output_min,              /* int8_t output_min                      */
      output_max,              /* int8_t output_max                      */
      flags,                   /* uint32_t flags                         */
      nullptr,                 /* xnn_caches_t caches                    */
      fully_connected_op_out); /* xnn_operator_t* fully_connected_op_out */
}

C10_ALWAYS_INLINE
enum xnn_status xnnp_setup_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    const int8_t* input,
    int8_t* output,
    pthreadpool_t threadpool) {
  return xnn_setup_fully_connected_nc_qs8(
      fully_connected_op, /* xnn_operator_t fully_connected_op */
      batch_size,         /* size_t batch_size                 */
      input,              /* const int8_t* input               */
      output,             /* int8_t* output                    */
      threadpool);        /* pthreadpool_t threadpool          */
}

} // namespace xnnp_utils
} // namespace native
} // namespace at

#endif // USE_XNNPACK
