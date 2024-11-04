// Heavily inspired by
// https://github.com/pytorch/pytorch/blob/09519eb19/aten/src/ATen/native/cuda/im2col.cuh#L51
template <typename T>
void im2col_kernel(
    constant T* input,
    device T* output,
    uint2 kernel_size,
    long2 input_offset,
    long2 input_size,
    long2 dilation,
    ulong2 input_strides,
    ulong output_stride) {
  for (ulong i = 0; i < kernel_size.y; ++i) {
    for (ulong j = 0; j < kernel_size.x; ++j) {
      auto input_pos = input_offset + long2(j, i) * dilation;
      if (input_pos.x < 0 || input_pos.y < 0 || input_pos.x >= input_size.x ||
          input_pos.y >= input_size.y) {
        *output = T(0);
      } else {
        auto offset =
            input_pos.x * input_strides.x + input_pos.y * input_strides.y;
        *output = input[offset];
      }
      output += output_stride;
    }
  }
}

template <typename T>
kernel void im2col(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant uint4& kernel_dilation [[buffer(2)]],
    constant int4& padding_stride [[buffer(3)]],
    constant ulong4& input_strides [[buffer(4)]],
    constant ulong4& output_strides [[buffer(5)]],
    constant long4& input_sizes [[buffer(6)]],
    uint3 thread_index [[thread_position_in_grid]]) {
  // thread_index is (output_length, input_channels, input_batch)
  const auto N = thread_index.z;
  const auto C = thread_index.y;
  const auto L = thread_index.x;
  const auto output_width = output_strides.w;
  const auto o_x = L % output_width;
  const auto o_y = L / output_width;
  auto i_x = o_x * padding_stride.z - padding_stride.x;
  auto i_y = o_y * padding_stride.w - padding_stride.y;
  ulong kernel_size = kernel_dilation.x * kernel_dilation.y;
  outputData += N * output_strides.z + C * kernel_size * output_strides.y +
      L * output_strides.x;
  inputData += N * input_strides.w + C * input_strides.z;
  im2col_kernel(
      inputData,
      outputData,
      kernel_dilation.xy,
      long2(i_x, i_y),
      input_sizes.xy,
      long2(kernel_dilation.zw),
      input_strides.xy,
      output_strides.y);
}

#define INSTANTIATE_IM2COL(DTYPE)                                     \
  template [[host_name("im2col_" #DTYPE)]] kernel void im2col<DTYPE>( \
      constant DTYPE * inputData [[buffer(0)]],                       \
      device DTYPE * outputData [[buffer(1)]],                        \
      constant uint4 & kernel_dilation [[buffer(2)]],                 \
      constant int4 & padding_stride [[buffer(3)]],                   \
      constant ulong4 & input_strides [[buffer(4)]],                  \
      constant ulong4 & output_strides [[buffer(5)]],                 \
      constant long4 & input_sizes [[buffer(6)]],                     \
      uint3 thread_index [[thread_position_in_grid]])

INSTANTIATE_IM2COL(bool);
INSTANTIATE_IM2COL(float);
INSTANTIATE_IM2COL(float2);
INSTANTIATE_IM2COL(half);
INSTANTIATE_IM2COL(half2);
#if __METAL_VERSION__ >= 310
INSTANTIATE_IM2COL(bfloat);
#endif
