#include <qnnpack/indirection.h>
#include <qnnpack/log.h>
#include <qnnpack/operator.h>
#include <qnnpack/pack.h>
#include <qnnpack_func.h>
#include <cstring>
#include <memory>
#include <numeric>

namespace qnnpack {

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

  const size_t output_channel_index = nr_block_start + group_index * n;
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
      output_channel_index,
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

  const size_t output_channel_index = group_index * n + nr_block_start;
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
      output_channel_index,
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

struct q8dwconv2d_context {
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
  const pytorch_q8dwconv2d_up_ukernel_function unipass_ukernel;
  const pytorch_q8dwconv2d_mp_ukernel_function multipass_ukernel;
};

struct q8dwconv3d_context {
  size_t groups;
  size_t group_stride;
  const uint8_t** indirection_buffer;
  size_t indirection_buffer_slice_stride;
  size_t indirection_buffer_row_stride;
  size_t indirection_buffer_col_stride;
  const void* packed_weights;
  uint8_t* output;
  size_t output_depth;
  size_t output_height;
  size_t output_width;
  size_t output_slice_stride;
  union pytorch_qnnp_conv_quantization_params quantization_params;
  const pytorch_q8dwconv3d_mp_ukernel_function multipass_ukernel;
};

static void compute_dwconv2d_unipass(
    const struct q8dwconv2d_context context[1],
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
static void compute_dwconv2d_multiipass(
    const struct q8dwconv2d_context context[1],
    size_t image,
    size_t output_y) {
  const size_t output_height = context->output_height;
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  int32_t* multipass_acc = (int32_t*)_malloca(sizeof(int32_t) * context->group_stride);
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

static void compute_dwconv3d_multiipass(
    const struct q8dwconv3d_context context[1],
    size_t image,
    size_t output_z) {
  const size_t output_depth = context->output_depth;
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  int32_t* multipass_acc =
      (int32_t*)_malloca(sizeof(int32_t) * context->group_stride);
#else
  int32_t multipass_acc[context->group_stride];
#endif

  context->multipass_ukernel(
      context->groups,
      context->output_height,
      context->output_width,
      context->indirection_buffer +
          (image * output_depth + output_z) *
              context->indirection_buffer_slice_stride,
      context->packed_weights,
      multipass_acc,
      context->output +
          (image * output_depth + output_z) * context->output_slice_stride,
      context->indirection_buffer_row_stride,
      context->indirection_buffer_col_stride,
      0,
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
    const pytorch_qnnp_operator_t convolution,
    void* packed_weights,
    const size_t batch_size,
    const size_t input_depth,
    const size_t input_height,
    const size_t input_width,
    const uint8_t input_zero_point,
    const uint8_t* input,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    const uint8_t output_zero_point,
    const uint8_t output_min,
    const uint8_t output_max,
    uint8_t* output,
    pthreadpool_t threadpool) {
  const size_t groups = convolution->groups;
  const size_t input_pixel_stride = convolution->group_input_channels * groups;
  const size_t output_pixel_stride =
      convolution->group_output_channels * groups;
  const size_t kernel_width = convolution->kernel_width;
  const size_t kernel_height = convolution->kernel_height;
  const size_t kernel_depth = convolution->kernel_depth;
  const size_t kernel_size = kernel_height * kernel_width * kernel_depth;

  if (batch_size == 0) {
    // If no batches, return
    return pytorch_qnnp_status_success;
  }

  union pytorch_qnnp_q31_requantization_params requantization_params {};
  union pytorch_qnnp_conv_quantization_params conv_quantization_params {};
  if (convolution->ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
    requantization_params = pytorch_qnnp_compute_requantization_params(
        // Note. XZP kernels are not changed for per channel quant.
        requantization_scales[0],
        output_zero_point,
        output_min,
        output_max);
  } else {
    conv_quantization_params = pytorch_qnnp_compute_conv_quantization_params(
        input_zero_point,
        kernel_zero_points,
        requantization_scales,
        output_zero_point,
        output_min,
        output_max);
  }

  // Convolution op caches a few things.
  // We need to check if the corresponding values on this
  // invocation is same as cached values.
  // If so we can skip setup step.
  if (convolution->input != input || convolution->batch_size != batch_size ||
      convolution->input_depth != input_depth ||
      convolution->input_height != input_height ||
      convolution->input_width != input_width ||
      convolution->input_pixel_stride != input_pixel_stride) {
    pytorch_qnnp_status status = pytorch_qnnp_setup_convolution_ndhwc_q8(
        convolution,
        batch_size,
        input_depth,
        input_height,
        input_width,
        input,
        input_pixel_stride,
        output,
        output_pixel_stride,
        threadpool);
    if (status != pytorch_qnnp_status_success) {
      pytorch_qnnp_log_error(
          "failed to run convolution op setup to setup indirection buffer.");
      return status;
    }
  }

  const size_t output_size = convolution->output_height *
      convolution->output_width * convolution->output_depth;

  switch (convolution->ukernel_type) {
    case pytorch_qnnp_ukernel_type_dwconv: {
      const uint32_t cr = pytorch_qnnp_params.q8dw9.cr;
      const size_t group_stride = (groups + (cr - 1)) & -cr;

      const size_t step_height = convolution->step_height;
      const size_t step_width = convolution->step_width;

      switch (kernel_size) {
        case 9: {
          struct q8dwconv2d_context context = {
              .groups = groups,
              .group_stride = group_stride,
              .indirection_buffer =
                  (const uint8_t**)convolution->indirection_buffer,
              .indirection_buffer_row_stride = step_height,
              .indirection_buffer_col_stride =
                  kernel_height * step_width * sizeof(void*),
              .packed_weights = packed_weights,
              .output = output,
              .output_height = convolution->output_height,
              .output_width = convolution->output_width,
              .output_row_stride =
                  convolution->output_width * output_pixel_stride,
              .output_col_increment =
                  (output_pixel_stride - groups) * sizeof(uint8_t),
              .quantization_params = conv_quantization_params,
              .unipass_ukernel = convolution->per_channel
                  ? pytorch_qnnp_params.q8dw9.updw_per_channel
                  : pytorch_qnnp_params.q8dw9.updw,
              .multipass_ukernel = convolution->per_channel
                  ? pytorch_qnnp_params.q8dw25.mpdw_per_channel
                  : pytorch_qnnp_params.q8dw25.mpdw,
          };
          pthreadpool_compute_2d(
              threadpool,
              (pthreadpool_function_2d_t)compute_dwconv2d_unipass,
              &context,
              batch_size,
              convolution->output_height);
          break;
        }
        case 25: {
          struct q8dwconv2d_context context = {
              .groups = groups,
              .group_stride = group_stride,
              .indirection_buffer =
                  (const uint8_t**)convolution->indirection_buffer,
              .indirection_buffer_row_stride = step_height,
              .indirection_buffer_col_stride =
                  kernel_height * step_width * sizeof(void*),
              .packed_weights = packed_weights,
              .output = output,
              .output_height = convolution->output_height,
              .output_width = convolution->output_width,
              .output_row_stride =
                  convolution->output_width * output_pixel_stride,
              .output_col_increment =
                  (output_pixel_stride - groups) * sizeof(uint8_t),
              .quantization_params = conv_quantization_params,
              .unipass_ukernel = convolution->per_channel
                  ? pytorch_qnnp_params.q8dw9.updw_per_channel
                  : pytorch_qnnp_params.q8dw9.updw,
              .multipass_ukernel = convolution->per_channel
                  ? pytorch_qnnp_params.q8dw25.mpdw_per_channel
                  : pytorch_qnnp_params.q8dw25.mpdw,
          };
          pthreadpool_compute_2d(
              threadpool,
              (pthreadpool_function_2d_t)compute_dwconv2d_multiipass,
              &context,
              batch_size,
              convolution->output_height);
          break;
        }
        case 27: {
          struct q8dwconv3d_context context = {
              .groups = groups,
              .group_stride = group_stride,
              .indirection_buffer =
                  (const uint8_t**)convolution->indirection_buffer,
              .indirection_buffer_slice_stride =
                  step_height * convolution->output_height,
              .indirection_buffer_row_stride = step_height * sizeof(void*),
              .indirection_buffer_col_stride =
                  kernel_height * kernel_depth * step_width * sizeof(void*),
              .packed_weights = packed_weights,
              .output = output,
              .output_depth = convolution->output_depth,
              .output_height = convolution->output_height,
              .output_width = convolution->output_width,
              .output_slice_stride = convolution->output_height *
                  convolution->output_width * output_pixel_stride,
              .quantization_params = conv_quantization_params,
              .multipass_ukernel = pytorch_qnnp_params.q8dw27.mpdw,
          };
          pthreadpool_compute_2d(
              threadpool,
              (pthreadpool_function_2d_t)compute_dwconv3d_multiipass,
              &context,
              batch_size,
              convolution->output_depth);
          break;
        }
        default:
          PYTORCH_QNNP_UNREACHABLE;
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_xzp_gemm: {
      const size_t group_input_channels = convolution->group_input_channels;
      const size_t group_output_channels = convolution->group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv_xzp.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv_xzp.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv_xzp.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

      /* compute input row sum */
      const size_t input_size = input_depth * input_height * input_width;
      int32_t* a_sum = (int32_t*)realloc(
          convolution->a_sum,
          sizeof(int32_t) * batch_size * groups * input_size);
      if (a_sum == nullptr) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for row sum data",
            sizeof(int32_t) * batch_size * groups * input_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      convolution->a_sum = a_sum;
      struct q8sum_rows_context context = {
          .a = input,
          .groups = groups,
          .m = input_size,
          .k = convolution->group_input_channels,
          .a_stride = input_pixel_stride,
          // XZP kernels are not supporting per channel quant.
          // We dont really use XZP kernels ATM.
          // Thus assigning the zero point of first channel.
          .multiplier = (int32_t)-kernel_zero_points[0],
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
          .k = convolution->group_input_channels,
          .k_stride = k_stride,
          .n = convolution->group_output_channels,
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
      const size_t group_input_channels = convolution->group_input_channels;
      const size_t group_output_channels = convolution->group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

      struct q8gemm_context q8gemm_context = {
          .k = convolution->group_input_channels,
          .k_stride = k_stride,
          .n = convolution->group_output_channels,
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
      const size_t group_input_channels = convolution->group_input_channels;
      const size_t group_output_channels = convolution->group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;
      const size_t m_stride = round_up(output_size, mr);

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
