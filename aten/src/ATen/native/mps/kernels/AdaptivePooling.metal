#include <ATen/native/mps/kernels/AdaptivePooling.h>
#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void adaptive_avg_pool2d_forward(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant AdaptiveAvgPool2DParams& params [[buffer(2)]],
    uint output_index [[thread_position_in_grid]]) {
  const ulong output_width = ulong(params.output_width);
  const ulong output_height = ulong(params.output_height);
  const ulong output_plane = output_height * output_width;
  const ulong channel_plane = ulong(output_index) / output_plane;
  const ulong output_offset = ulong(output_index) % output_plane;
  const ulong batch = channel_plane / ulong(params.C);
  const ulong channel = channel_plane % ulong(params.C);
  const ulong output_y = output_offset / output_width;
  const ulong output_x = output_offset % output_width;

  const ulong input_y_start =
      output_y * ulong(params.input_height) / output_height;
  const ulong input_y_end =
      ((output_y + 1) * ulong(params.input_height) + output_height - 1) /
      output_height;
  const ulong input_x_start =
      output_x * ulong(params.input_width) / output_width;
  const ulong input_x_end =
      ((output_x + 1) * ulong(params.input_width) + output_width - 1) /
      output_width;

  const ulong input_base = batch * ulong(params.input_strides[0]) +
      channel * ulong(params.input_strides[1]);
  float sum = 0.0f;
  for (ulong input_y = input_y_start; input_y < input_y_end; ++input_y) {
    for (ulong input_x = input_x_start; input_x < input_x_end; ++input_x) {
      sum += float(input
                       [input_base + input_y * ulong(params.input_strides[2]) +
                        input_x * ulong(params.input_strides[3])]);
    }
  }
  const float count =
      float((input_y_end - input_y_start) * (input_x_end - input_x_start));
  const ulong output_storage_index = batch * ulong(params.output_strides[0]) +
      channel * ulong(params.output_strides[1]) +
      output_y * ulong(params.output_strides[2]) +
      output_x * ulong(params.output_strides[3]);
  output[output_storage_index] = T(sum / count);
}

template <typename T>
kernel void adaptive_avg_pool2d_backward(
    constant T* grad_output [[buffer(0)]],
    device T* grad_input [[buffer(1)]],
    constant AdaptiveAvgPool2DParams& params [[buffer(2)]],
    uint input_index [[thread_position_in_grid]]) {
  const ulong input_width = ulong(params.input_width);
  const ulong input_height = ulong(params.input_height);
  const ulong input_plane = input_height * input_width;
  const ulong channel_plane = ulong(input_index) / input_plane;
  const ulong input_offset = ulong(input_index) % input_plane;
  const ulong batch = channel_plane / ulong(params.C);
  const ulong channel = channel_plane % ulong(params.C);
  const ulong input_y = input_offset / input_width;
  const ulong input_x = input_offset % input_width;

  const ulong output_y_start =
      input_y * ulong(params.output_height) / input_height;
  const ulong output_y_end =
      ((input_y + 1) * ulong(params.output_height) + input_height - 1) /
      input_height;
  const ulong output_x_start =
      input_x * ulong(params.output_width) / input_width;
  const ulong output_x_end =
      ((input_x + 1) * ulong(params.output_width) + input_width - 1) /
      input_width;

  const ulong output_base = batch * ulong(params.output_strides[0]) +
      channel * ulong(params.output_strides[1]);
  float sum = 0.0f;
  for (ulong output_y = output_y_start; output_y < output_y_end; ++output_y) {
    const ulong input_y_start =
        output_y * input_height / ulong(params.output_height);
    const ulong input_y_end =
        ((output_y + 1) * input_height + ulong(params.output_height) - 1) /
        ulong(params.output_height);
    for (ulong output_x = output_x_start; output_x < output_x_end; ++output_x) {
      const ulong input_x_start =
          output_x * input_width / ulong(params.output_width);
      const ulong input_x_end =
          ((output_x + 1) * input_width + ulong(params.output_width) - 1) /
          ulong(params.output_width);
      const float count =
          float((input_y_end - input_y_start) * (input_x_end - input_x_start));
      sum +=
          float(grad_output
                    [output_base + output_y * ulong(params.output_strides[2]) +
                     output_x * ulong(params.output_strides[3])]) /
          count;
    }
  }
  const ulong input_storage_index = batch * ulong(params.input_strides[0]) +
      channel * ulong(params.input_strides[1]) +
      input_y * ulong(params.input_strides[2]) +
      input_x * ulong(params.input_strides[3]);
  grad_input[input_storage_index] = T(sum);
}

#define REGISTER_ADAPTIVE_AVG_POOL2D(T)                        \
  template [[host_name("adaptive_avg_pool2d_forward_" #T)]]    \
  kernel void adaptive_avg_pool2d_forward<T>(                  \
      constant T * input [[buffer(0)]],                        \
      device T * output [[buffer(1)]],                         \
      constant AdaptiveAvgPool2DParams & params [[buffer(2)]], \
      uint output_index [[thread_position_in_grid]]);          \
  template [[host_name("adaptive_avg_pool2d_backward_" #T)]]   \
  kernel void adaptive_avg_pool2d_backward<T>(                 \
      constant T * grad_output [[buffer(0)]],                  \
      device T * grad_input [[buffer(1)]],                     \
      constant AdaptiveAvgPool2DParams & params [[buffer(2)]], \
      uint input_index [[thread_position_in_grid]]);

REGISTER_ADAPTIVE_AVG_POOL2D(float)
REGISTER_ADAPTIVE_AVG_POOL2D(half)
REGISTER_ADAPTIVE_AVG_POOL2D(bfloat)
