// Heavily inspired by
// https://github.com/pytorch/pytorch/blob/09519eb19/aten/src/ATen/native/cuda/im2col.cuh#L51
#include <metal_stdlib>
using namespace metal;

template <typename T, typename IdxT>
void im2col_kernel(
    constant T* input,
    device T* output,
    uint2 kernel_size,
    vec<make_signed_t<IdxT>, 2> input_offset,
    vec<make_signed_t<IdxT>, 2> input_size,
    vec<make_signed_t<IdxT>, 2> dilation,
    vec<IdxT, 2> input_strides,
    IdxT output_stride) {
  using SIdxT = make_signed_t<IdxT>;
  for (uint i = 0; i < kernel_size.y; ++i) {
    for (uint j = 0; j < kernel_size.x; ++j) {
      auto input_pos = input_offset + vec<SIdxT, 2>(j, i) * dilation;
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

template <typename T, typename IdxT>
kernel void im2col(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant uint4& kernel_dilation [[buffer(2)]],
    constant int4& padding_stride [[buffer(3)]],
    constant vec<IdxT, 4>& input_strides [[buffer(4)]],
    constant vec<IdxT, 4>& output_strides [[buffer(5)]],
    constant vec<make_signed_t<IdxT>, 4>& input_sizes [[buffer(6)]],
    uint3 thread_index [[thread_position_in_grid]]) {
  using SIdxT = make_signed_t<IdxT>;
  // thread_index is (output_length, input_channels, input_batch)
  const auto N = thread_index.z;
  const auto C = thread_index.y;
  const auto L = thread_index.x;
  const IdxT output_width = output_strides.w;
  const IdxT o_x = L % output_width;
  const IdxT o_y = L / output_width;
  SIdxT i_x = SIdxT(o_x) * padding_stride.z - padding_stride.x;
  SIdxT i_y = SIdxT(o_y) * padding_stride.w - padding_stride.y;
  IdxT kernel_size = kernel_dilation.x * kernel_dilation.y;
  outputData += IdxT(N) * output_strides.z +
      IdxT(C) * kernel_size * output_strides.y + IdxT(L) * output_strides.x;
  inputData += IdxT(N) * input_strides.w + IdxT(C) * input_strides.z;
  im2col_kernel<T, IdxT>(
      inputData,
      outputData,
      kernel_dilation.xy,
      vec<SIdxT, 2>(i_x, i_y),
      input_sizes.xy,
      vec<SIdxT, 2>(kernel_dilation.zw),
      input_strides.xy,
      output_strides.y);
}

#define INSTANTIATE_IM2COL_IDX(DTYPE, IDX, SUFFIX)                     \
  template [[host_name("im2col_" #DTYPE "_" #SUFFIX)]] kernel void     \
  im2col<DTYPE, IDX>(                                                  \
      constant DTYPE * inputData [[buffer(0)]],                        \
      device DTYPE * outputData [[buffer(1)]],                         \
      constant uint4 & kernel_dilation [[buffer(2)]],                  \
      constant int4 & padding_stride [[buffer(3)]],                    \
      constant vec<IDX, 4> & input_strides [[buffer(4)]],              \
      constant vec<IDX, 4> & output_strides [[buffer(5)]],             \
      constant vec<make_signed_t<IDX>, 4> & input_sizes [[buffer(6)]], \
      uint3 thread_index [[thread_position_in_grid]])

#define INSTANTIATE_IM2COL(DTYPE)           \
  INSTANTIATE_IM2COL_IDX(DTYPE, uint, u32); \
  INSTANTIATE_IM2COL_IDX(DTYPE, ulong, u64)

INSTANTIATE_IM2COL(bool);
INSTANTIATE_IM2COL(float);
INSTANTIATE_IM2COL(float2);
INSTANTIATE_IM2COL(half);
INSTANTIATE_IM2COL(half2);
INSTANTIATE_IM2COL(bfloat);
