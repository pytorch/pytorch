#include <ATen/native/mps/kernels/AdaptivePooling.h>
#include <metal_stdlib>

using namespace metal;

template <typename T, typename index_t>
kernel void adaptive_avg_pool2d_forward(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant AdaptiveAvgPool2DParams& params [[buffer(2)]],
    uint output_index [[thread_position_in_grid]]) {
  const index_t output_width = index_t(params.output_width);
  const index_t output_height = index_t(params.output_height);
  const index_t output_plane = output_height * output_width;
  const index_t channel_plane = index_t(output_index) / output_plane;
  const index_t output_offset = index_t(output_index) % output_plane;
  const index_t batch = channel_plane / index_t(params.C);
  const index_t channel = channel_plane % index_t(params.C);
  const index_t output_y = output_offset / output_width;
  const index_t output_x = output_offset % output_width;

  const index_t input_y_start =
      output_y * index_t(params.input_height) / output_height;
  const index_t input_y_end =
      ((output_y + 1) * index_t(params.input_height) + output_height - 1) /
      output_height;
  const index_t input_x_start =
      output_x * index_t(params.input_width) / output_width;
  const index_t input_x_end =
      ((output_x + 1) * index_t(params.input_width) + output_width - 1) /
      output_width;

  const index_t input_base = batch * index_t(params.input_strides[0]) +
      channel * index_t(params.input_strides[1]);
  float sum = 0.0f;
  for (index_t input_y = input_y_start; input_y < input_y_end; ++input_y) {
    for (index_t input_x = input_x_start; input_x < input_x_end; ++input_x) {
      sum +=
          float(input
                    [input_base + input_y * index_t(params.input_strides[2]) +
                     input_x * index_t(params.input_strides[3])]);
    }
  }
  const float count =
      float((input_y_end - input_y_start) * (input_x_end - input_x_start));
  const index_t output_storage_index =
      batch * index_t(params.output_strides[0]) +
      channel * index_t(params.output_strides[1]) +
      output_y * index_t(params.output_strides[2]) +
      output_x * index_t(params.output_strides[3]);
  output[output_storage_index] = T(sum / count);
}

template <typename T, typename index_t>
kernel void adaptive_avg_pool2d_backward(
    constant T* grad_output [[buffer(0)]],
    device T* grad_input [[buffer(1)]],
    constant AdaptiveAvgPool2DParams& params [[buffer(2)]],
    uint input_index [[thread_position_in_grid]]) {
  const index_t input_width = index_t(params.input_width);
  const index_t input_height = index_t(params.input_height);
  const index_t input_plane = input_height * input_width;
  const index_t channel_plane = index_t(input_index) / input_plane;
  const index_t input_offset = index_t(input_index) % input_plane;
  const index_t batch = channel_plane / index_t(params.C);
  const index_t channel = channel_plane % index_t(params.C);
  const index_t input_y = input_offset / input_width;
  const index_t input_x = input_offset % input_width;

  const index_t output_y_start =
      input_y * index_t(params.output_height) / input_height;
  const index_t output_y_end =
      ((input_y + 1) * index_t(params.output_height) + input_height - 1) /
      input_height;
  const index_t output_x_start =
      input_x * index_t(params.output_width) / input_width;
  const index_t output_x_end =
      ((input_x + 1) * index_t(params.output_width) + input_width - 1) /
      input_width;

  const index_t output_base = batch * index_t(params.output_strides[0]) +
      channel * index_t(params.output_strides[1]);
  float sum = 0.0f;
  for (index_t output_y = output_y_start; output_y < output_y_end; ++output_y) {
    const index_t input_y_start =
        output_y * input_height / index_t(params.output_height);
    const index_t input_y_end =
        ((output_y + 1) * input_height + index_t(params.output_height) - 1) /
        index_t(params.output_height);
    for (index_t output_x = output_x_start; output_x < output_x_end;
         ++output_x) {
      const index_t input_x_start =
          output_x * input_width / index_t(params.output_width);
      const index_t input_x_end =
          ((output_x + 1) * input_width + index_t(params.output_width) - 1) /
          index_t(params.output_width);
      const float count =
          float((input_y_end - input_y_start) * (input_x_end - input_x_start));
      sum +=
          float(
              grad_output
                  [output_base + output_y * index_t(params.output_strides[2]) +
                   output_x * index_t(params.output_strides[3])]) /
          count;
    }
  }
  const index_t input_storage_index = batch * index_t(params.input_strides[0]) +
      channel * index_t(params.input_strides[1]) +
      input_y * index_t(params.input_strides[2]) +
      input_x * index_t(params.input_strides[3]);
  grad_input[input_storage_index] = T(sum);
}

#define REGISTER_ADAPTIVE_AVG_POOL2D(T, IT)                          \
  template [[host_name("adaptive_avg_pool2d_forward_" #T "_" #IT)]]  \
  kernel void adaptive_avg_pool2d_forward<T, IT>(                    \
      constant T * input [[buffer(0)]],                              \
      device T * output [[buffer(1)]],                               \
      constant AdaptiveAvgPool2DParams & params [[buffer(2)]],       \
      uint output_index [[thread_position_in_grid]]);                \
  template [[host_name("adaptive_avg_pool2d_backward_" #T "_" #IT)]] \
  kernel void adaptive_avg_pool2d_backward<T, IT>(                   \
      constant T * grad_output [[buffer(0)]],                        \
      device T * grad_input [[buffer(1)]],                           \
      constant AdaptiveAvgPool2DParams & params [[buffer(2)]],       \
      uint input_index [[thread_position_in_grid]]);

REGISTER_ADAPTIVE_AVG_POOL2D(float, uint)
REGISTER_ADAPTIVE_AVG_POOL2D(float, ulong)
REGISTER_ADAPTIVE_AVG_POOL2D(half, uint)
REGISTER_ADAPTIVE_AVG_POOL2D(half, ulong)
REGISTER_ADAPTIVE_AVG_POOL2D(bfloat, uint)
REGISTER_ADAPTIVE_AVG_POOL2D(bfloat, ulong)
