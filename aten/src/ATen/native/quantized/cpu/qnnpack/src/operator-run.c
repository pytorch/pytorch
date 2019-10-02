/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/common.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/params.h>

#ifdef _MSC_VER
#include <malloc.h>
#endif

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
    const struct q8gemm_context context[RESTRICT_STATIC 1],
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
  const uint8_t* restrict a = context->a;
  const size_t a_stride = context->a_stride;
  const void* restrict packed_w = context->packed_w;
  uint8_t* restrict c = context->c;
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
    const struct q8sum_rows_context context[RESTRICT_STATIC 1],
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
    const struct q8gemm_xzp_context context[RESTRICT_STATIC 1],
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
  const uint8_t* restrict a = context->a;
  const size_t a_stride = context->a_stride;
  const void* restrict packed_w = context->packed_w;
  uint8_t* restrict c = context->c;
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
    const struct q8conv_context context[RESTRICT_STATIC 1],
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
  const uint8_t** restrict indirect_a = context->indirect_a;
  const void* restrict packed_w = context->packed_w;
  uint8_t* restrict c = context->c;
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
  union {
    const pytorch_q8dwconv_up_ukernel_function unipass_ukernel;
    const pytorch_q8dwconv_mp_ukernel_function multipass_ukernel;
  };
};

static void compute_dwconv_unipass(
    const struct q8dwconv_context context[RESTRICT_STATIC 1],
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
    const struct q8dwconv_context context[RESTRICT_STATIC 1],
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

struct max_pooling_context {
  const void** indirect_input;
  size_t indirect_input_batch_stride;
  size_t indirect_input_height_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  size_t pooling_size;
  size_t channels;
  size_t input_increment;
  size_t output_increment;
  union pytorch_qnnp_u8_clamping_params params;
  pytorch_u8maxpool_ukernel_function ukernel;
};

static void compute_max_pooling(
    const struct max_pooling_context context[RESTRICT_STATIC 1],
    size_t batch_index,
    size_t output_y) {
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input +
      batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->ukernel(
      context->output_width,
      context->pooling_size,
      context->channels,
      (const uint8_t**)indirect_input,
      output,
      context->input_increment,
      context->output_increment,
      &context->params);
}

struct average_pooling_context {
  const void** indirect_input;
  size_t indirect_input_batch_stride;
  size_t indirect_input_height_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  size_t pooling_size;
  size_t channels;
  size_t packed_channels;
  const void* zero;
  size_t input_increment;
  size_t output_increment;
  union pytorch_qnnp_avgpool_quantization_params quantization_params;
  union {
    pytorch_q8avgpool_up_ukernel_function unipass_ukernel;
    pytorch_q8avgpool_mp_ukernel_function multipass_ukernel;
  };
};

static void compute_average_pooling_unipass(
    const struct average_pooling_context context[RESTRICT_STATIC 1],
    size_t batch_index,
    size_t output_y) {
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input +
      batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->unipass_ukernel(
      context->output_width,
      context->pooling_size,
      context->channels,
      (const uint8_t**)indirect_input,
      context->zero,
      output,
      context->input_increment,
      context->output_increment,
      &context->quantization_params);
}

static void compute_average_pooling_multipass(
    const struct average_pooling_context context[RESTRICT_STATIC 1],
    size_t batch_index,
    size_t output_y) {
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input +
      batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  int32_t* multipass_buffer =
      _malloca(sizeof(int32_t) * context->packed_channels);
#else
  int32_t multipass_buffer[context->packed_channels];
#endif

  context->multipass_ukernel(
      context->output_width,
      context->pooling_size,
      context->channels,
      (const uint8_t**)indirect_input,
      context->zero,
      multipass_buffer,
      output,
      context->input_increment,
      context->output_increment,
      &context->quantization_params);

#ifdef _MSC_VER
  _freea(multipass_buffer);
#endif
}

struct global_average_pooling_context {
  const void* input;
  const void* zero;
  size_t input_pixel_stride;
  size_t input_batch_stride;
  size_t input_elements;
  size_t channels;
  size_t packed_channels;
  void* output;
  size_t output_batch_stride;
  union pytorch_qnnp_avgpool_quantization_params quantization_params;
  union {
    pytorch_q8gavgpool_up_ukernel_function unipass_ukernel;
    pytorch_q8gavgpool_mp_ukernel_function multipass_ukernel;
  };
};

static void compute_global_average_pooling_unipass(
    const struct global_average_pooling_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  const void* input =
      (const void*)((uintptr_t)context->input + batch_index * context->input_batch_stride);
  void* output =
      (void*)((uintptr_t)context->output + batch_index * context->output_batch_stride);

  context->unipass_ukernel(
      context->input_elements,
      context->channels,
      input,
      context->input_pixel_stride,
      context->zero,
      output,
      &context->quantization_params);
}

static void compute_global_average_pooling_multipass(
    const struct global_average_pooling_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  const void* input =
      (const void*)((uintptr_t)context->input + batch_index * context->input_batch_stride);
  void* output =
      (void*)((uintptr_t)context->output + batch_index * context->output_batch_stride);
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  int32_t* multipass_buffer =
      _malloca(sizeof(int32_t) * context->packed_channels);
#else
  int32_t multipass_buffer[context->packed_channels];
#endif

  context->multipass_ukernel(
      context->input_elements,
      context->channels,
      input,
      context->input_pixel_stride,
      context->zero,
      multipass_buffer,
      output,
      &context->quantization_params);

#ifdef _MSC_VER
  _freea(multipass_buffer);
#endif
}

struct q8add_strided_context {
  size_t n;
  const uint8_t* a;
  size_t a_stride;
  const uint8_t* b;
  size_t b_stride;
  const uint8_t* y;
  size_t y_stride;
  union pytorch_qnnp_add_quantization_params quantization_params;
  pytorch_q8vadd_ukernel_function ukernel;
};

static void compute_q8add_strided(
    const struct q8add_strided_context context[RESTRICT_STATIC 1],
    size_t batch_offset,
    size_t batch_range /* always 1 */) {
  assert(batch_range == 1);

  const size_t n = context->n;
  const size_t a_stride = context->a_stride;
  const size_t b_stride = context->b_stride;
  const size_t y_stride = context->y_stride;
  const void* a =
      (const void*)((uintptr_t)context->a + a_stride * batch_offset);
  const void* b =
      (const void*)((uintptr_t)context->b + b_stride * batch_offset);
  void* y = (void*)((uintptr_t)context->y + y_stride * batch_offset);

  context->ukernel(n, a, b, y, &context->quantization_params);
}

struct q8add_contiguous_context {
  const uint8_t* a;
  const uint8_t* b;
  uint8_t* y;
  union pytorch_qnnp_add_quantization_params quantization_params;
  pytorch_q8vadd_ukernel_function ukernel;
};

static void compute_q8add_contiguous(
    const struct q8add_contiguous_context context[RESTRICT_STATIC 1],
    size_t offset,
    size_t size) {
  const void* a = (const void*)((uintptr_t)context->a + offset);
  const void* b = (const void*)((uintptr_t)context->b + offset);
  void* y = (void*)((uintptr_t)context->y + offset);
  context->ukernel(size, a, b, y, &context->quantization_params);
}

struct channel_shuffle_context {
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  size_t n;
  size_t m;
  union {
    pytorch_xzipc_ukernel_function fixed_ukernel;
    pytorch_xzipv_ukernel_function variable_ukernel;
  };
};

static void compute_channel_shuffle_fixed(
    const struct channel_shuffle_context context[RESTRICT_STATIC 1],
    size_t index) {
  const void* x =
      (const void*)((uintptr_t)context->x + index * context->x_stride);
  void* y = (void*)((uintptr_t)context->y + index * context->y_stride);

  context->fixed_ukernel(context->n, x, y);
}

static void compute_channel_shuffle_variable(
    const struct channel_shuffle_context context[RESTRICT_STATIC 1],
    size_t index) {
  const void* x =
      (const void*)((uintptr_t)context->x + index * context->x_stride);
  void* y = (void*)((uintptr_t)context->y + index * context->y_stride);

  context->variable_ukernel(context->n, context->m, x, y);
}

struct lut_strided_context {
  size_t n;
  const void* x;
  size_t x_stride;
  const void* t;
  void* y;
  size_t y_stride;
  pytorch_x8lut_ukernel_function ukernel;
};

static void compute_lut_strided(
    const struct lut_strided_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  const void* x =
      (const void*)((uintptr_t)context->x + context->x_stride * batch_index);
  void* y = (void*)((uintptr_t)context->y + context->y_stride * batch_index);

  context->ukernel(context->n, x, context->t, y);
}

struct lut_contiguous_context {
  const void* x;
  size_t x_stride;
  const void* t;
  void* y;
  size_t y_stride;
  pytorch_x8lut_ukernel_function ukernel;
};

static void compute_lut_contiguous(
    const struct lut_contiguous_context context[RESTRICT_STATIC 1],
    size_t offset,
    size_t size) {
  const void* x = (const void*)((uintptr_t)context->x + offset);
  void* y = (void*)((uintptr_t)context->y + offset);

  context->ukernel(size, x, context->t, y);
}

struct clamp_strided_context {
  size_t n;
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  pytorch_u8clamp_ukernel_function ukernel;
  union pytorch_qnnp_u8_clamping_params params;
};

static void compute_clamp_strided(
    const struct clamp_strided_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  const void* x =
      (const void*)((uintptr_t)context->x + context->x_stride * batch_index);
  void* y = (void*)((uintptr_t)context->y + context->y_stride * batch_index);
  context->ukernel(context->n, x, y, &context->params);
}

struct clamp_contiguous_context {
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  pytorch_u8clamp_ukernel_function ukernel;
  union pytorch_qnnp_u8_clamping_params params;
};

static void compute_clamp_contiguous(
    const struct clamp_contiguous_context context[RESTRICT_STATIC 1],
    size_t offset,
    size_t size) {
  const void* x = (const void*)((uintptr_t)context->x + offset);
  void* y = (void*)((uintptr_t)context->y + offset);
  context->ukernel(size, x, y, &context->params);
}

struct u8softargmax_context {
  size_t n;
  const uint8_t* x;
  size_t x_stride;
  const uint32_t* t;
  uint8_t* y;
  size_t y_stride;
  pytorch_u8rmax_ukernel_function rmax_ukernel;
  pytorch_u8lut32norm_ukernel_function lut_norm_ukernel;
};

static void compute_u8softargmax(
    const struct u8softargmax_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  const uint8_t* x =
      (const uint8_t*)((uintptr_t)context->x + context->x_stride * batch_index);
  uint8_t* y =
      (uint8_t*)((uintptr_t)context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  const uint8_t x_max = context->rmax_ukernel(n, x);
  const size_t adjustment = x_max ^ 255;
  const uint32_t* t = (const uint32_t*)context->t + adjustment;
  context->lut_norm_ukernel(n, x, t, y);
}

enum pytorch_qnnp_status pytorch_qnnp_run_operator(
    pytorch_qnnp_operator_t op,
    pthreadpool_t threadpool) {
  // For any ukernel type, there is no work to do if the batch size is 0.
  if (op->batch_size == 0) {
    return pytorch_qnnp_status_success;
  }

  switch (op->ukernel_type) {
    case pytorch_qnnp_ukernel_type_dwconv: {
      const size_t batch_size = op->batch_size;
      const size_t groups = op->groups;
      const size_t kernel_height = op->kernel_height;
      const size_t kernel_width = op->kernel_width;
      const size_t kernel_size = kernel_height * kernel_width;
      const size_t width_step =
          op->dilation_width == 1 ? op->stride_width : op->kernel_width;
      const size_t output_height = op->output_height;
      const size_t output_width = op->output_width;

      switch (kernel_size) {
        case 9: {
          struct q8dwconv_context context = {
              .groups = groups,
              .indirection_buffer = (const uint8_t**)op->indirection_buffer,
              .indirection_buffer_row_stride =
                  kernel_size + (output_width * width_step - 1) * kernel_height,
              .indirection_buffer_col_stride =
                  kernel_height * width_step * sizeof(void*),
              .packed_weights = op->packed_weights,
              .output = op->output,
              .output_height = output_height,
              .output_width = output_width,
              .output_row_stride = output_width * op->output_pixel_stride,
              .output_col_increment =
                  (op->output_pixel_stride - groups) * sizeof(uint8_t),
              .quantization_params = op->conv_quantization_params,
              .unipass_ukernel = pytorch_qnnp_params.q8dw9.updw,
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
              .group_stride = op->group_stride,
              .indirection_buffer = (const uint8_t**)op->indirection_buffer,
              .indirection_buffer_row_stride =
                  kernel_size + (output_width * width_step - 1) * kernel_height,
              .indirection_buffer_col_stride =
                  kernel_height * width_step * sizeof(void*),
              .packed_weights = op->packed_weights,
              .output = op->output,
              .output_height = output_height,
              .output_width = output_width,
              .output_row_stride = output_width * op->output_pixel_stride,
              .output_col_increment =
                  (op->output_pixel_stride - groups) * sizeof(uint8_t),
              .quantization_params = op->conv_quantization_params,
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
      const size_t batch_size = op->batch_size;
      const size_t groups = op->groups;
      const size_t group_input_channels = op->group_input_channels;
      const size_t group_output_channels = op->group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv_xzp.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv_xzp.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv_xzp.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

      /* compute input row sum */
      const size_t input_size = op->input_height * op->input_width;
      int32_t* a_sum = (int32_t*)op->a_sum;

      struct q8sum_rows_context context = {
          .a = op->input,
          .groups = groups,
          .m = input_size,
          .k = group_input_channels,
          .a_stride = op->input_pixel_stride,
          .multiplier = (int32_t)-op->kernel_zero_point,
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
          .k = group_input_channels,
          .k_stride = k_stride,
          .n = group_output_channels,
          .n_stride = n_stride,
          .a = op->input,
          .a_stride = op->input_pixel_stride,
          .packed_w = op->packed_weights,
          .c = op->output,
          .c_stride = op->output_pixel_stride,
          .a_sum = a_sum,
          .groups = op->groups,
          .batch_size = batch_size,
          .a_sum_stride = input_size,
          .requantization_params = op->requantization_params,
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
      const size_t batch_size = op->batch_size;
      const size_t groups = op->groups;
      const size_t group_input_channels = op->group_input_channels;
      const size_t group_output_channels = op->group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

      const size_t output_size = op->output_height * op->output_width;
      struct q8gemm_context q8gemm_context = {
          .k = group_input_channels,
          .k_stride = k_stride,
          .n = group_output_channels,
          .n_stride = n_stride,
          .a = op->input,
          .a_stride = op->input_pixel_stride,
          .packed_w = op->packed_weights,
          .c = op->output,
          .c_stride = op->output_pixel_stride,
          .quantization_params = op->conv_quantization_params,
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
      const size_t batch_size = op->batch_size;
      const size_t groups = op->groups;
      const size_t group_input_channels = op->group_input_channels;
      const size_t group_output_channels = op->group_output_channels;
      const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

      const size_t output_size = op->output_height * op->output_width;
      const size_t kernel_size = op->kernel_height * op->kernel_width;
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
          .indirect_a = (const uint8_t**)op->indirection_buffer,
          .packed_w = op->packed_weights,
          .c = op->output,
          .c_stride = op->output_pixel_stride,
          .quantization_params = op->conv_quantization_params,
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
    case pytorch_qnnp_ukernel_type_average_pooling: {
      const uint32_t kr = pytorch_qnnp_params.q8avgpool.kr;
      const uint32_t mr = pytorch_qnnp_params.q8avgpool.mr;
      const uint32_t qr = pytorch_qnnp_params.q8avgpool.qr;
      const size_t channels = op->channels;
      const size_t output_width = op->output_width;
      const size_t output_height = op->output_height;
      const size_t pooling_height = op->kernel_height;
      const size_t pooling_width = op->kernel_width;
      const size_t pooling_size = pooling_height * pooling_width;

      const size_t width_step = min(op->stride_width, pooling_width);
      const size_t indirect_input_height_stride =
          (pooling_size + (output_width * width_step - 1) * pooling_height) *
          sizeof(void*);
      const size_t output_height_stride =
          output_width * op->output_pixel_stride;

      size_t multipass_adjustment = 0;
      if (channels >= kr && pooling_size > mr) {
        multipass_adjustment = round_up(pooling_size - mr, qr) + mr - qr;
      }
      struct average_pooling_context context = {
          .indirect_input = op->indirection_buffer,
          .indirect_input_batch_stride =
              output_height * indirect_input_height_stride,
          .indirect_input_height_stride = indirect_input_height_stride,
          .output = op->output,
          .output_batch_stride = output_height * output_height_stride,
          .output_height_stride = output_height_stride,
          .output_width = output_width,
          .pooling_size = pooling_size,
          .channels = channels,
          .packed_channels = (channels + (kr - 1)) & -kr,
          .zero = op->zero_pointer,
          .input_increment =
              (pooling_height * width_step - multipass_adjustment) *
              sizeof(void*),
          .output_increment =
              (op->output_pixel_stride - channels) * sizeof(uint8_t),
          .quantization_params = op->avgpool_quantization_params,
      };

      pthreadpool_function_2d_t compute_function = NULL;
      if (channels < kr) {
        compute_function =
            (pthreadpool_function_2d_t)compute_average_pooling_unipass;
        context.unipass_ukernel = pytorch_qnnp_params.q8avgpool.ltkr;
      } else {
        if (pooling_size <= mr) {
          compute_function =
              (pthreadpool_function_2d_t)compute_average_pooling_unipass;
          context.unipass_ukernel = pytorch_qnnp_params.q8avgpool.gekr_lemr;
        } else {
          compute_function =
              (pthreadpool_function_2d_t)compute_average_pooling_multipass;
          context.multipass_ukernel = pytorch_qnnp_params.q8avgpool.gekr_gtmr;
        }
      }

      pthreadpool_compute_2d(
          threadpool,
          compute_function,
          &context,
          op->batch_size,
          output_height);
      break;
    }
    case pytorch_qnnp_ukernel_type_max_pooling: {
      const uint32_t kr = pytorch_qnnp_params.u8maxpool.kr;
      const uint32_t mr = pytorch_qnnp_params.u8maxpool.mr;
      const uint32_t qr = pytorch_qnnp_params.u8maxpool.qr;
      const size_t channels = op->channels;
      const size_t output_width = op->output_width;
      const size_t output_height = op->output_height;
      const size_t pooling_height = op->kernel_height;
      const size_t pooling_width = op->kernel_width;
      const size_t pooling_size = pooling_height * pooling_width;

      const size_t width_step = op->dilation_width > 1
          ? pooling_width
          : min(op->stride_width, pooling_width);
      const size_t indirect_input_height_stride =
          (pooling_size + (output_width * width_step - 1) * pooling_height) *
          sizeof(void*);
      const size_t output_height_stride =
          output_width * op->output_pixel_stride;

      size_t multipass_adjustment = pooling_size;
      if (channels >= kr) {
        multipass_adjustment = round_up(doz(pooling_size, mr), qr) + mr;
      }
      struct max_pooling_context context = {
          .indirect_input = op->indirection_buffer,
          .indirect_input_batch_stride =
              output_height * indirect_input_height_stride,
          .indirect_input_height_stride = indirect_input_height_stride,
          .output = op->output,
          .output_batch_stride = output_height * output_height_stride,
          .output_height_stride = output_height_stride,
          .output_width = output_width,
          .pooling_size = pooling_size,
          .channels = channels,
          .input_increment =
              (pooling_height * width_step - multipass_adjustment) *
              sizeof(void*),
          .output_increment =
              (op->output_pixel_stride - channels) * sizeof(uint8_t),
          .params = op->u8_clamping_params,
          .ukernel = channels < kr ? pytorch_qnnp_params.u8maxpool.ltkr
                                   : pytorch_qnnp_params.u8maxpool.gekr,
      };

      pthreadpool_compute_2d(
          threadpool,
          (pthreadpool_function_2d_t)compute_max_pooling,
          &context,
          op->batch_size,
          output_height);
      break;
    };
    case pytorch_qnnp_ukernel_type_add: {
      const size_t batch_size = op->batch_size;
      const size_t channels = op->channels;
      const size_t a_stride = op->input_pixel_stride;
      const size_t b_stride = op->input2_pixel_stride;
      const size_t y_stride = op->output_pixel_stride;
      if ((((a_stride ^ channels) | (b_stride ^ channels) |
            (y_stride ^ channels)) == 0) ||
          batch_size == 1) {
        const size_t block_size = 4096;
        struct q8add_contiguous_context add_context = {
            .a = op->input,
            .b = op->input2,
            .y = op->output,
            .quantization_params = op->add_quantization_params,
            .ukernel = pytorch_qnnp_params.q8vadd,
        };
        pthreadpool_compute_1d_tiled(
            threadpool,
            (pthreadpool_function_1d_tiled_t)compute_q8add_contiguous,
            &add_context,
            batch_size * channels * sizeof(uint8_t),
            block_size);
      } else {
        struct q8add_strided_context add_context = {
            .a = op->input,
            .a_stride = a_stride * sizeof(uint8_t),
            .b = op->input2,
            .b_stride = b_stride * sizeof(uint8_t),
            .y = op->output,
            .y_stride = y_stride * sizeof(uint8_t),
            .n = channels,
            .quantization_params = op->add_quantization_params,
            .ukernel = pytorch_qnnp_params.q8vadd,
        };
        pthreadpool_compute_1d_tiled(
            threadpool,
            (pthreadpool_function_1d_tiled_t)compute_q8add_strided,
            &add_context,
            batch_size,
            1);
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_global_average_pooling: {
      const uint32_t nr = pytorch_qnnp_params.q8gavgpool.nr;
      const uint32_t mr = pytorch_qnnp_params.q8gavgpool.mr;
      const size_t input_pixel_stride =
          op->input_pixel_stride * sizeof(uint8_t);
      const size_t input_width = op->input_width;
      const size_t channels = op->channels;
      struct global_average_pooling_context context = {
          .input = op->input,
          .zero = op->zero_pointer,
          .input_pixel_stride = input_pixel_stride,
          .input_batch_stride = input_pixel_stride * input_width,
          .input_elements = input_width,
          .channels = channels,
          .packed_channels = (channels + (nr - 1)) & -nr,
          .output = op->output,
          .output_batch_stride = op->output_pixel_stride * sizeof(uint8_t),
          .quantization_params = op->avgpool_quantization_params,
      };
      pthreadpool_function_1d_t compute_function = NULL;
      if (channels < nr) {
        compute_function =
            (pthreadpool_function_1d_t)compute_global_average_pooling_unipass;
        context.unipass_ukernel = pytorch_qnnp_params.q8gavgpool.ltnr;
      } else {
        if (input_width <= mr) {
          compute_function =
              (pthreadpool_function_1d_t)compute_global_average_pooling_unipass;
          context.unipass_ukernel = pytorch_qnnp_params.q8gavgpool.genr_lemr;
        } else {
          compute_function = (pthreadpool_function_1d_t)
              compute_global_average_pooling_multipass;
          context.multipass_ukernel = pytorch_qnnp_params.q8gavgpool.genr_gtmr;
        }
      }

      pthreadpool_compute_1d(
          threadpool, compute_function, &context, op->batch_size);
      break;
    }
    case pytorch_qnnp_ukernel_type_lut: {
      const size_t batch_size = op->batch_size;
      const size_t channels = op->channels;
      const size_t x_stride = op->input_pixel_stride;
      const size_t y_stride = op->output_pixel_stride;
      if ((((x_stride ^ channels) | (y_stride ^ channels)) == 0) ||
          batch_size == 1) {
        const size_t block_size = 1024;
        struct lut_contiguous_context context = {
            .x = op->input,
            .x_stride = x_stride * sizeof(uint8_t),
            .t = op->lookup_table,
            .y = op->output,
            .y_stride = y_stride * sizeof(uint8_t),
            .ukernel = pytorch_qnnp_params.x8lut,
        };
        pthreadpool_compute_1d_tiled(
            threadpool,
            (pthreadpool_function_1d_tiled_t)compute_lut_contiguous,
            &context,
            batch_size * channels * sizeof(uint8_t),
            block_size);
      } else {
        struct lut_strided_context context = {
            .n = channels,
            .x = op->input,
            .x_stride = x_stride * sizeof(uint8_t),
            .t = op->lookup_table,
            .y = op->output,
            .y_stride = y_stride * sizeof(uint8_t),
            .ukernel = pytorch_qnnp_params.x8lut,
        };
        pthreadpool_compute_1d(
            threadpool,
            (pthreadpool_function_1d_t)compute_lut_strided,
            &context,
            batch_size);
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_clamp: {
      const size_t batch_size = op->batch_size;
      const size_t channels = op->channels;
      const size_t x_stride = op->input_pixel_stride;
      const size_t y_stride = op->output_pixel_stride;
      if ((((x_stride ^ channels) | (y_stride ^ channels)) == 0) ||
          batch_size == 1) {
        const size_t block_size = 4096;
        struct clamp_contiguous_context context = {
            .x = op->input,
            .x_stride = x_stride * sizeof(uint8_t),
            .y = op->output,
            .y_stride = y_stride * sizeof(uint8_t),
            .ukernel = pytorch_qnnp_params.u8clamp,
            .params = op->u8_clamping_params,
        };
        pthreadpool_compute_1d_tiled(
            threadpool,
            (pthreadpool_function_1d_tiled_t)compute_clamp_contiguous,
            &context,
            batch_size * channels * sizeof(uint8_t),
            block_size);
      } else {
        struct clamp_strided_context context = {
            .n = channels,
            .x = op->input,
            .x_stride = x_stride * sizeof(uint8_t),
            .y = op->output,
            .y_stride = y_stride * sizeof(uint8_t),
            .ukernel = pytorch_qnnp_params.u8clamp,
            .params = op->u8_clamping_params,
        };
        pthreadpool_compute_1d(
            threadpool,
            (pthreadpool_function_1d_t)compute_clamp_strided,
            &context,
            batch_size);
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_softargmax: {
      struct u8softargmax_context context = {
          .n = op->channels,
          .x = op->input,
          .x_stride = op->input_pixel_stride * sizeof(uint8_t),
          .t = op->lookup_table,
          .y = op->output,
          .y_stride = op->output_pixel_stride * sizeof(uint8_t),
          .rmax_ukernel = pytorch_qnnp_params.u8rmax,
          .lut_norm_ukernel = pytorch_qnnp_params.u8lut32norm,
      };
      pthreadpool_compute_1d(
          threadpool,
          (pthreadpool_function_1d_t)compute_u8softargmax,
          &context,
          op->batch_size);
      break;
    }
    case pytorch_qnnp_ukernel_type_channel_shuffle: {
      const size_t groups = op->groups;
      struct channel_shuffle_context channel_shuffle_context = {
          .x = op->input,
          .x_stride = op->input_pixel_stride * sizeof(uint8_t),
          .y = op->output,
          .y_stride = op->output_pixel_stride * sizeof(uint8_t),
          .n = op->group_channels * sizeof(uint8_t),
          .m = groups,
      };
      pthreadpool_function_1d_t compute_function = NULL;
      switch (groups) {
        case 2:
          compute_function =
              (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
          channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x2;
          break;
        case 3:
          compute_function =
              (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
          channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x3;
          break;
        case 4:
          compute_function =
              (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
          channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x4;
          break;
        default:
          compute_function =
              (pthreadpool_function_1d_t)compute_channel_shuffle_variable;
          channel_shuffle_context.variable_ukernel =
              pytorch_qnnp_params.x8zip.xm;
          break;
        case 0:
        case 1:
          PYTORCH_QNNP_UNREACHABLE;
      }
      pthreadpool_compute_1d(
          threadpool,
          compute_function,
          &channel_shuffle_context,
          op->batch_size);
      break;
    }
    default:
      PYTORCH_QNNP_UNREACHABLE;
  }
  return pytorch_qnnp_status_success;
}
