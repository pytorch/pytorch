#include <conv_utils.h>
#include <qnnpack/indirection.h>
#include <qnnpack/pack.h>
#include <qnnpack_func.h>
#include <cstring>
#include <memory>

namespace qnnpack {

static inline size_t compute_output_dimension(
    size_t padded_input_dim,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t subsampling_dimension) {
  const size_t effective_kernel_dim =
      (kernel_dimension - 1) * dilation_dimension + 1;
  return (padded_input_dim - effective_kernel_dim) / subsampling_dimension + 1;
}

struct q8gemm_xzp_context {
  size_t k;
  size_t k_stride;
  size_t n;
  size_t n_stride;
  const uint8_t* a;
  size_t a_stride;
  const void* packed_w;
  uint8_t* c;
  size_t c_stride;
  const int32_t* a_sum;
  size_t groups;
  size_t batch_size;
  size_t a_sum_stride;
  union pytorch_qnnp_q31_requantization_params requantization_params;
  const pytorch_q8gemm_xzp_ukernel_function ukernel;
};
static void compute_q8gemm_xzp(
    const struct q8gemm_xzp_context context[1],
    size_t group_index,
    size_t pixel_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t pixel_range,
    size_t mr_block_size,
    size_t nr_block_size) {
  const size_t k = context->k;
  const size_t k_stride = context->k_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t* a = context->a;
  const size_t a_stride = context->a_stride;
  const void* packed_w = context->packed_w;
  uint8_t* c = context->c;
  const size_t c_stride = context->c_stride;
  const int32_t* a_sum = context->a_sum;
  const size_t groups = context->groups;
  const size_t a_sum_stride = context->a_sum_stride;

  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,
      a_stride,
      a_sum + pixel_index * groups + group_index * a_sum_stride +
          mr_block_start,
      (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(uint8_t) + sizeof(int32_t))),
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start +
          group_index * n,
      c_stride,
      &context->requantization_params);
}

struct q8gemm_context {
  size_t k;
  size_t k_stride;
  size_t n;
  size_t n_stride;
  const uint8_t* a;
  size_t a_stride;
  const uint8_t* packed_w;
  uint8_t* c;
  size_t c_stride;
  union pytorch_qnnp_conv_quantization_params quantization_params;
  const pytorch_q8gemm_ukernel_function ukernel;
};
static void compute_q8gemm(
    const struct q8gemm_context context[1],
    size_t group_index,
    size_t pixel_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t pixel_range,
    size_t mr_block_size,
    size_t nr_block_size) {
  const size_t k = context->k;
  const size_t k_stride = context->k_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t* a = context->a;
  const size_t a_stride = context->a_stride;
  const void* packed_w = context->packed_w;
  uint8_t* c = context->c;
  const size_t c_stride = context->c_stride;

  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,
      a_stride,
      (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(uint8_t) + sizeof(int32_t))),
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start +
          group_index * n,
      c_stride,
      &context->quantization_params);
}

struct q8conv_context {
  size_t bs;
  size_t ks;
  size_t kc;
  size_t kc_stride;
  size_t m;
  size_t m_stride;
  size_t n;
  size_t n_stride;
  const uint8_t** indirect_a;
  const void* packed_w;
  uint8_t* c;
  size_t c_stride;
  union pytorch_qnnp_conv_quantization_params quantization_params;
  const pytorch_q8conv_ukernel_function ukernel;
};
static void compute_q8conv(
    const struct q8conv_context context[1],
    size_t group_index,
    size_t image_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t image_range /* always 1 */,
    size_t mr_block_size,
    size_t nr_block_size) {
  const size_t bs = context->bs;
  const size_t ks = context->ks;
  const size_t kc = context->kc;
  const size_t kc_stride = context->kc_stride;
  const size_t m = context->m;
  const size_t m_stride = context->m_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t** indirect_a = context->indirect_a;
  const void* packed_w = context->packed_w;
  uint8_t* c = context->c;
  const size_t c_stride = context->c_stride;

  context->ukernel(
      mr_block_size,
      nr_block_size,
      kc,
      ks,
      indirect_a +
          (mr_block_start + (image_index + group_index * bs) * m_stride) * ks,
      (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (kc_stride * sizeof(uint8_t) + sizeof(int32_t))),
      c + (mr_block_start + image_index * m) * c_stride + group_index * n +
          nr_block_start,
      c_stride,
      &context->quantization_params);
}

struct q8sum_rows_context {
  const uint8_t* a;
  size_t groups;
  size_t m;
  size_t k;
  size_t a_stride;
  const int32_t multiplier;
  int32_t* a_sum;
  size_t a_sum_stride;
  const pytorch_q8sum_rows_ukernel_function ukernel;
};
static void compute_sum_rows(
    const struct q8sum_rows_context context[1],
    size_t group_index,
    size_t batch_index,
    size_t block_start,
    size_t group_range /* always 1 */,
    size_t batch_range /* always 1 */,
    size_t block_size) {
  const uint8_t* a = context->a;
  const size_t groups = context->groups;
  const size_t m = context->m;
  const size_t k = context->k;
  const size_t a_stride = context->a_stride;
  const int32_t multiplier = context->multiplier;
  int32_t* a_sum = context->a_sum;
  const size_t a_sum_stride = context->a_sum_stride;

  context->ukernel(
      a + batch_index * m * a_stride + group_index * k + block_start * a_stride,
      min(block_size, m - block_start),
      k,
      a_stride,
      multiplier,
      a_sum + batch_index * groups * a_sum_stride + group_index * a_sum_stride +
          block_start);
}

struct q8dwconv_context {
  size_t groups;
  size_t group_stride;
  const uint8_t** indirection_buffer;
  size_t indirection_buffer_row_stride;
  size_t indirection_buffer_col_stride;
  const void* packed_weights;
  uint8_t* output;
  size_t output_height;
  size_t output_width;
  size_t output_row_stride;
  size_t output_col_increment;
  union pytorch_qnnp_conv_quantization_params quantization_params;
  const pytorch_q8dwconv_up_ukernel_function unipass_ukernel;
  const pytorch_q8dwconv_mp_ukernel_function multipass_ukernel;
};
static void compute_dwconv_unipass(
    const struct q8dwconv_context context[1],
    size_t image,
    size_t output_y) {
  const size_t output_height = context->output_height;

  context->unipass_ukernel(
      context->groups,
      context->output_width,
      context->indirection_buffer +
          (image * output_height + output_y) *
              context->indirection_buffer_row_stride,
      context->packed_weights,
      context->output +
          (image * output_height + output_y) * context->output_row_stride,
      context->indirection_buffer_col_stride,
      context->output_col_increment,
      &context->quantization_params);
}
static void compute_dwconv_multiipass(
    const struct q8dwconv_context context[1],
    size_t image,
    size_t output_y) {
  const size_t output_height = context->output_height;
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  int32_t* multipass_acc = _malloca(sizeof(int32_t) * context->group_stride);
#else
  int32_t multipass_acc[context->group_stride];
#endif

  context->multipass_ukernel(
      context->groups,
      context->output_width,
      context->indirection_buffer +
          (image * output_height + output_y) *
              context->indirection_buffer_row_stride,
      context->packed_weights,
      multipass_acc,
      context->output +
          (image * output_height + output_y) * context->output_row_stride,
      context->indirection_buffer_col_stride,
      context->output_col_increment,
      &context->quantization_params);

#ifdef _MSC_VER
  _freea(multipass_acc);
#endif
}

struct QnnpackDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);
  }
};

enum pytorch_qnnp_status qnnpackConv(
    const conv_param_t& conv_p,
    void* packed_weights,
    const size_t batch_size,
    const size_t input_height,
    const size_t input_width,
    const float input_scale,
    const uint8_t input_zero_point,
    const uint8_t* input,
    const float output_scale,
    const uint8_t output_zero_point,
    uint8_t* output,
    pthreadpool_t threadpool) {
  const size_t input_pixel_stride = conv_p.input_channels;
  const size_t output_pixel_stride = conv_p.output_channels;
  const size_t kernel_width = conv_p.kernel_dims[0];
  const size_t kernel_height = conv_p.kernel_dims[1];
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t dilation_width = conv_p.dilation[0];
  const size_t dilation_height = conv_p.dilation[1];
  const size_t groups = conv_p.groups;

  const float convolution_scale =
      input_scale * conv_p.kernel_scale / output_scale;
  if (convolution_scale >= 1.0f) {
    pytorch_qnnp_log_error(
        "failed to create convolution with %.7g input scale, %.7g kernel scale,"
        " and %.7g output scale: "
        "convolution scale %.7g is greater or equal to 1.0",
        input_scale,
        conv_p.kernel_scale,
        output_scale,
        convolution_scale);
  }
  union pytorch_qnnp_q31_requantization_params requantization_params;
  union pytorch_qnnp_conv_quantization_params conv_quantization_params;
  if (conv_p.ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
    requantization_params = pytorch_qnnp_compute_requantization_params(
        convolution_scale,
        output_zero_point,
        conv_p.output_min,
        conv_p.output_max);
  } else {
    conv_quantization_params = pytorch_qnnp_compute_conv_quantization_params(
        input_zero_point,
        conv_p.kernel_zero_point,
        convolution_scale,
        output_zero_point,
        conv_p.output_min,
        conv_p.output_max);
  }
  uint32_t stride_width = conv_p.subsampling_dims[0];
  uint32_t stride_height = conv_p.subsampling_dims[1];

  size_t output_height = compute_output_dimension(
      conv_p.pad[0] + input_height + conv_p.pad[2],
      kernel_height,
      dilation_height,
      stride_height);
  size_t output_width = compute_output_dimension(
      conv_p.pad[1] + input_width + conv_p.pad[3],
      kernel_width,
      dilation_width,
      stride_width);
  const size_t output_size = output_height * output_width;

  // FIXME temporary solution to create a qnnp_op struct for indirection buffer.
  const bool any_padding =
      (conv_p.pad[0] | conv_p.pad[1] | conv_p.pad[2] | conv_p.pad[3]) != 0;
  size_t zero_size = 0, zero_offset = 0;

  pytorch_qnnp_operator_t convolution{nullptr};
  convolution =
      static_cast<pytorch_qnnp_operator_t>(calloc(1, sizeof(struct pytorch_qnnp_operator)));
  if (convolution == nullptr) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
        sizeof(struct pytorch_qnnp_operator));
    return pytorch_qnnp_status_out_of_memory;
  }

  std::unique_ptr<pytorch_qnnp_operator, QnnpackDeleter> qnnpack_uniq_ptr(convolution);

  convolution->input = input;
  convolution->input_pixel_stride = input_pixel_stride;
  convolution->groups = groups;
  convolution->group_input_channels = conv_p.group_input_channels;
  convolution->batch_size = batch_size;
  convolution->input_height = input_height;
  convolution->input_width = input_width;
  convolution->output_height = output_height;
  convolution->output_width = output_width;
  convolution->kernel_height = kernel_height;
  convolution->kernel_width = kernel_width;
  convolution->stride_height = stride_height;
  convolution->stride_width = stride_width;
  convolution->dilation_height = dilation_height;
  convolution->dilation_width = dilation_width;
  convolution->input_padding_top = conv_p.pad[0];
  convolution->input_padding_left = conv_p.pad[1];

  switch (conv_p.ukernel_type) {
    case pytorch_qnnp_ukernel_type_dwconv: {
      const size_t width_step =
          dilation_width == 1 ? stride_width : kernel_width;
      const uint32_t cr = pytorch_qnnp_params.q8dw9.cr;
      const size_t group_stride = (groups + (cr - 1)) & -cr;

      if (any_padding) {
        if (groups >= 8) {
          zero_size = sizeof(uint8_t) * group_stride;
          zero_offset = 0;
        } else {
          zero_size = sizeof(uint8_t) * group_stride + 8;
          zero_offset = sizeof(uint8_t) * 8;
        }
        void* zero_buffer = malloc(zero_size);
        if (zero_buffer == nullptr) {
          pytorch_qnnp_log_error(
              "failed to allocate %zu bytes for zero padding", zero_size);
          return pytorch_qnnp_status_out_of_memory;
        }
        memset(zero_buffer, input_zero_point, zero_size);
        convolution->zero_buffer = zero_buffer;
        convolution->zero_pointer =
            (void*)((uintptr_t)zero_buffer + zero_offset);
      }
      const size_t step_width = convolution->dilation_width == 1
          ? convolution->stride_width
          : kernel_width;
      const size_t step_height =
          kernel_size + (output_width * step_width - 1) * kernel_height;
      const size_t indirection_buffer_size =
          sizeof(void*) * batch_size * output_height * step_height;

      const void** indirection_buffer = (const void**)realloc(
          convolution->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == nullptr) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for indirection buffer",
            indirection_buffer_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      convolution->indirection_buffer = indirection_buffer;

      pytorch_qnnp_indirection_init_dwconv2d(convolution, 0, step_height, step_width);

      switch (kernel_size) {
        case 9: {
          struct q8dwconv_context context = {
              .groups = groups,
              .group_stride = group_stride,
              .indirection_buffer = (const uint8_t**)indirection_buffer,
              .indirection_buffer_row_stride =
                  kernel_size + (output_width * width_step - 1) * kernel_height,
              .indirection_buffer_col_stride =
                  kernel_height * width_step * sizeof(void*),
              .packed_weights = packed_weights,
              .output = output,
              .output_height = output_height,
              .output_width = output_width,
              .output_row_stride = output_width * output_pixel_stride,
              .output_col_increment =
                  (output_pixel_stride - groups) * sizeof(uint8_t),
              .quantization_params = conv_quantization_params,
              .unipass_ukernel = pytorch_qnnp_params.q8dw9.updw,
              .multipass_ukernel = pytorch_qnnp_params.q8dw25.mpdw,
          };
          pthreadpool_compute_2d(
              threadpool,
              (pthreadpool_function_2d_t)compute_dwconv_unipass,
              &context,
              batch_size,
              output_height);
          break;
        }
        case 25: {
          struct q8dwconv_context context = {
              .groups = groups,
              .group_stride = group_stride,
              .indirection_buffer =
                  (const uint8_t**)convolution->indirection_buffer,
              .indirection_buffer_row_stride =
                  kernel_size + (output_width * width_step - 1) * kernel_height,
              .indirection_buffer_col_stride =
                  kernel_height * width_step * sizeof(void*),
              .packed_weights = packed_weights,
              .output = output,
              .output_height = output_height,
              .output_width = output_width,
              .output_row_stride = output_width * output_pixel_stride,
              .output_col_increment =
                  (output_pixel_stride - groups) * sizeof(uint8_t),
              .quantization_params = conv_quantization_params,
              .unipass_ukernel = pytorch_qnnp_params.q8dw9.updw,
              .multipass_ukernel = pytorch_qnnp_params.q8dw25.mpdw,
          };
          pthreadpool_compute_2d(
              threadpool,
              (pthreadpool_function_2d_t)compute_dwconv_multiipass,
              &context,
              batch_size,
              output_height);
          break;
        }
        default:
          PYTORCH_QNNP_UNREACHABLE;
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_xzp_gemm: {
      const size_t group_input_channels = conv_p.group_input_channels;
      const size_t group_output_channels = conv_p.group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv_xzp.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv_xzp.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv_xzp.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

      /* compute input row sum */
      const size_t input_size = input_height * input_width;
      int32_t* a_sum = (int32_t*)realloc(
          convolution->a_sum,
          sizeof(int32_t) * batch_size * groups * input_height * input_width);
      if (a_sum == nullptr) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for row sum data",
            sizeof(int32_t) * batch_size * groups * input_height * input_width);
        return pytorch_qnnp_status_out_of_memory;
      }
      convolution->a_sum = a_sum;
      struct q8sum_rows_context context = {
          .a = input,
          .groups = groups,
          .m = input_size,
          .k = conv_p.group_input_channels,
          .a_stride = input_pixel_stride,
          .multiplier = (int32_t)-conv_p.kernel_zero_point,
          .a_sum = a_sum,
          .a_sum_stride = input_size,
          .ukernel = pytorch_qnnp_params.q8sum_rows.sum_rows,
      };
      pthreadpool_compute_3d_tiled(
          threadpool,
          (pthreadpool_function_3d_tiled_t)compute_sum_rows,
          &context,
          groups,
          batch_size,
          input_size,
          1,
          1,
          pytorch_qnnp_params.q8sum_rows.m);

      struct q8gemm_xzp_context q8gemm_xzp_context = {
          .k = conv_p.group_input_channels,
          .k_stride = k_stride,
          .n = conv_p.group_output_channels,
          .n_stride = n_stride,
          .a = input,
          .a_stride = input_pixel_stride,
          .packed_w = packed_weights,
          .c = output,
          .c_stride = output_pixel_stride,
          .a_sum = a_sum,
          .groups = groups,
          .batch_size = batch_size,
          .a_sum_stride = input_size,
          .requantization_params = requantization_params,
          .ukernel = pytorch_qnnp_params.q8conv_xzp.gemm,
      };
      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t)compute_q8gemm_xzp,
          &q8gemm_xzp_context,
          groups,
          batch_size * input_size,
          input_size,
          group_output_channels,
          1,
          input_size,
          mr,
          nr);
      break;
    }
    case pytorch_qnnp_ukernel_type_gemm: {
      const size_t group_input_channels = conv_p.group_input_channels;
      const size_t group_output_channels = conv_p.group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

      struct q8gemm_context q8gemm_context = {
          .k = conv_p.group_input_channels,
          .k_stride = k_stride,
          .n = conv_p.group_output_channels,
          .n_stride = n_stride,
          .a = input,
          .a_stride = input_pixel_stride,
          .packed_w = (uint8_t*)packed_weights,
          .c = output,
          .c_stride = output_pixel_stride,
          .quantization_params = conv_quantization_params,
          .ukernel = pytorch_qnnp_params.q8conv.gemm,
      };

      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t)compute_q8gemm,
          &q8gemm_context,
          groups,
          batch_size * output_size,
          output_size,
          group_output_channels,
          1,
          output_size,
          mr,
          nr);
      break;
    }
    case pytorch_qnnp_ukernel_type_conv: {
      const size_t group_input_channels = conv_p.group_input_channels;
      const size_t group_output_channels = conv_p.group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;
      const size_t m_stride = round_up(output_size, mr);

      if (any_padding) {
        if (group_input_channels >= 8) {
          zero_size = sizeof(uint8_t) * k_stride;
          zero_offset = 0;
        } else {
          zero_size = sizeof(uint8_t) * k_stride + 8;
          zero_offset = 8;
        }
        void* zero_buffer = malloc(zero_size);
        if (zero_buffer == nullptr) {
          pytorch_qnnp_log_error(
              "failed to allocate %zu bytes for zero padding", zero_size);
          return pytorch_qnnp_status_out_of_memory;
        }
        memset(zero_buffer, input_zero_point, zero_size);
        convolution->zero_buffer = zero_buffer;
        convolution->zero_pointer =
            (void*)((uintptr_t)zero_buffer + zero_offset);
      }

      const size_t output_tile_size = pytorch_qnnp_params.q8conv.mr;
      const size_t tiled_output_size = round_up(output_size, output_tile_size);
      const size_t indirection_buffer_size =
          sizeof(void*) * batch_size * groups * tiled_output_size * kernel_size;
      const void** indirection_buffer = (const void**)realloc(
          convolution->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == nullptr) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for indirection buffer",
            indirection_buffer_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      convolution->indirection_buffer = indirection_buffer;

      pytorch_qnnp_indirection_init_conv2d(
          convolution, output_tile_size, tiled_output_size);

      struct q8conv_context q8conv_context = {
          .bs = batch_size,
          .ks = kernel_size,
          .kc = group_input_channels,
          .kc_stride = k_stride * kernel_size,
          .m = output_size,
          .m_stride = m_stride,
          .n = group_output_channels,
          .n_stride = n_stride,
          .indirect_a = (const uint8_t**)convolution->indirection_buffer,
          .packed_w = packed_weights,
          .c = output,
          .c_stride = output_pixel_stride,
          .quantization_params = conv_quantization_params,
          .ukernel = pytorch_qnnp_params.q8conv.conv,
      };

      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t)compute_q8conv,
          &q8conv_context,
          groups,
          batch_size,
          output_size,
          group_output_channels,
          1,
          1,
          mr,
          nr);
      break;
    }
    default: {
      pytorch_qnnp_log_error("Invalid kernel type. QNNPACK convolution run failed.");
      PYTORCH_QNNP_UNREACHABLE;
    }
  }
  return pytorch_qnnp_status_success;
}
} // namespace qnnpack
