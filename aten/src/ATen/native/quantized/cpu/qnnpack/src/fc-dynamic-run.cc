#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <cstring>

namespace qnnpack {
struct q8gemm_dq_context {
  size_t k;
  size_t k_stride;
  size_t n;
  size_t n_stride;
  const uint8_t* a;
  size_t a_stride;
  const uint8_t* packed_w;
  const float* bias;
  float* c;
  size_t c_stride;
  struct pytorch_qnnp_conv_dynamic_quantization_params quantization_params;
  const pytorch_q8gemm_dq_ukernel_function ukernel;
};

static void compute_q8gemm_dq(
    const struct q8gemm_dq_context* context,
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
  float* c = context->c;
  const size_t c_stride = context->c_stride;
  const float* bias = context->bias;

  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,
      a_stride,
      (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(uint8_t) + sizeof(int32_t))),
      bias + nr_block_start,
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start +
          group_index * n,
      c_stride,
      &context->quantization_params);
}

enum pytorch_qnnp_status qnnpackLinearDynamic(
    const size_t batch_size,
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t input_zero_point,
    const float input_scale,
    const uint8_t kernel_zero_point,
    const float kernel_scale,
    const uint8_t* input,
    const size_t input_stride,
    void* packed_weights,
    const float* bias,
    float* output,
    const size_t output_stride,
    pthreadpool_t threadpool) {
  const size_t groups = 1;
  const size_t group_input_channels = input_channels;
  const size_t group_output_channels = output_channels;
  const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
  const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
  const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

  const size_t output_size = batch_size * 1;

  const struct pytorch_qnnp_conv_dynamic_quantization_params
      quantizationParams {
    input_zero_point, kernel_zero_point, input_scale *kernel_scale,
  };

  struct q8gemm_dq_context q8gemm_dq_context = {
      .k = group_input_channels,
      .k_stride = k_stride,
      .n = group_output_channels,
      .n_stride = n_stride,
      .a = input,
      .a_stride = input_stride,
      .packed_w = (uint8_t*)packed_weights,
      .bias = bias,
      .c = output,
      .c_stride = output_stride,
      .quantization_params = quantizationParams,
      .ukernel = pytorch_qnnp_params.q8conv.gemm_dq,
  };

  pthreadpool_compute_4d_tiled(
      threadpool,
      (pthreadpool_function_4d_tiled_t)compute_q8gemm_dq,
      &q8gemm_dq_context,
      groups,
      1 * output_size,
      output_size,
      group_output_channels,
      1,
      output_size,
      mr,
      nr);

  return pytorch_qnnp_status_success;
}
} // namespace qnnpack
