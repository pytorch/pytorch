#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <cstring>

namespace qnnpack {
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
    size_t nr_block_size)
{
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
      (const void*) ((uintptr_t) packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(uint8_t) + sizeof(int32_t))),
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start + group_index * n,
      c_stride,
      &context->quantization_params);
}

enum pytorch_qnnp_status qnnpackLinear(
    const size_t batch_size,
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t input_zero_point,
    const float input_scale,
    const uint8_t kernel_zero_point,
    const float kernel_scale,
    const uint8_t output_zero_point,
    const float output_scale,
    const uint8_t output_min,
    const uint8_t output_max,
    const uint8_t* input,
    const size_t input_stride,
    void* packed_weights,
    uint8_t* output,
    const size_t output_stride,
    pthreadpool_t threadpool)
{
  const size_t groups = 1;
  const size_t group_input_channels = input_channels;
  const size_t group_output_channels = output_channels;
  const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
  const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
  const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

  const size_t output_size = batch_size * 1;
  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 1.0f) {
    pytorch_qnnp_log_error(
        "failed to create fully connected operator with %.7g input scale, %.7g "
        "kernel scale, and %.7g output scale: "
        "requantization scale %.7g is greater or equal to 1.0",
        input_scale,
        kernel_scale,
        output_scale,
        requantization_scale);
    return pytorch_qnnp_status_unsupported_parameter;
  }
  union pytorch_qnnp_conv_quantization_params conv_quantization_params = pytorch_qnnp_compute_conv_quantization_params(
      input_zero_point, kernel_zero_point, requantization_scale, output_zero_point, output_min, output_max);

  struct q8gemm_context q8gemm_context = {
      .k = group_input_channels,
      .k_stride = k_stride,
      .n = group_output_channels,
      .n_stride = n_stride,
      .a = input,
      .a_stride = input_stride,
      .packed_w = (uint8_t*) packed_weights,
      .c = output,
      .c_stride = output_stride,
      .quantization_params = conv_quantization_params,
      .ukernel = pytorch_qnnp_params.q8conv.gemm,
  };

  pthreadpool_compute_4d_tiled(
      threadpool,
      (pthreadpool_function_4d_tiled_t) compute_q8gemm,
      &q8gemm_context,
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
